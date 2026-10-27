"""Collect runs into the paper's Table I and turbine-bootstrap intervals.

Runs live at ``<root>/<experiment>/s<seed>/`` with ``results.json`` and
``per_event.csv`` (the layout written by ``python -m windturbine run``).
A run counts only when its ``COMPLETE`` marker exists, or when it is a committed
copy under ``results/paper`` (which keeps only the small files).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from windturbine.scoring import aggregate_event_metrics

COMPONENTS = ("care", "coverage", "earliness", "reliability", "accuracy")


def find_runs(root):
    rows = []
    for results in sorted(Path(root).glob("*/s*/results.json")):
        run = results.parent
        if not (run / "per_event.csv").is_file():
            continue
        data = json.loads(results.read_text(encoding="utf-8"))
        rows.append({"experiment": run.parent.name, "seed": int(run.name[1:]), "path": run,
                     **{k: data[k] for k in COMPONENTS}, "detected": data["tp"], "false_alarms": data["fp"]})
    return pd.DataFrame(rows)


def seed_table(runs, order=None):
    """Mean, SD and seed count of CARE, its components, detections and false alarms."""
    if runs.empty:
        return pd.DataFrame()
    metrics = [*COMPONENTS, "detected", "false_alarms"]
    grouped = runs.groupby("experiment")[metrics]
    table = grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_sd"))
    table["n_seeds"] = runs.groupby("experiment").size()
    table["seeds"] = runs.groupby("experiment")["seed"].apply(lambda s: " ".join(map(str, sorted(s))))
    if order:
        table = table.reindex([e for e in order if e in table.index] +
                              [e for e in table.index if e not in order])
    return table


def _care(per_event, options):
    return aggregate_event_metrics(per_event, **options)["CARE"]


def _turbine_sums(frame, turbines):
    """Per-turbine sufficient statistics for CARE: component sums and counts, and event confusion."""
    rows = []
    for aid in turbines:
        part = frame[frame.asset_id == aid]
        anomaly, detected = part.is_anomaly.to_numpy() == 1, part.detected.to_numpy() == 1
        rows.append([part.coverage.sum(skipna=True), part.coverage.notna().sum(),
                     part.earliness.sum(skipna=True), part.earliness.notna().sum(),
                     part.accuracy.sum(skipna=True), part.accuracy.notna().sum(),
                     (anomaly & detected).sum(), (~anomaly & detected).sum(), (anomaly & ~detected).sum()])
    return np.asarray(rows, dtype=float)


def _care_from_counts(weights, sums, beta, care_weights):
    """CARE for many resamples at once; ``weights`` is (draws, turbines) of resample counts.

    Same rules as aggregate_event_metrics: component means over supported events,
    F-beta reliability from event counts, 0 with no detection, Accuracy when it is below 0.5.
    """
    totals = weights @ sums
    with np.errstate(invalid="ignore", divide="ignore"):
        c, e, a = totals[:, 0] / totals[:, 1], totals[:, 2] / totals[:, 3], totals[:, 4] / totals[:, 5]
        tp, fp, fn = totals[:, 6], totals[:, 7], totals[:, 8]
        den = (1 + beta**2) * tp + beta**2 * fn + fp
        r = np.where(den > 0, (1 + beta**2) * tp / den, 0.0)
    w = np.asarray(care_weights, dtype=float)
    care = (w[0] * c + w[1] * e + w[2] * r + w[3] * a) / w.sum()
    care = np.where(a < 0.5, a, care)
    care = np.where(tp + fp == 0, 0.0, care)
    return np.where(np.isfinite(c) & np.isfinite(e) & np.isfinite(a), care, np.nan)


def bootstrap_difference(runs, a, b, *, n_boot=2000, seed=0, beta=0.5, care_weights=(1, 1, 1, 2)):
    """95% interval for mean CARE(a) - mean CARE(b), resampling turbines, not events.

    The 58 events come from 22 turbines, so events of one turbine are not
    independent. Each draw resamples turbines with replacement, keeps every event
    of a drawn turbine, computes CARE for each seed's run on the same draw and
    averages over seeds, so the two experiments are compared on identical draws.
    """
    options = {"beta": beta, "care_weights": care_weights}
    tables = {name: [pd.read_csv(p / "per_event.csv") for p in runs.loc[runs.experiment == name, "path"]]
              for name in (a, b)}
    if not tables[a] or not tables[b]:
        raise ValueError(f"Need at least one run of {a} and of {b}")
    turbines = np.array(sorted(tables[a][0]["asset_id"].unique()))
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(turbines), np.full(len(turbines), 1 / len(turbines)), size=n_boot)

    def mean_care(frames):
        return np.nanmean([_care_from_counts(draws, _turbine_sums(f, turbines), beta, care_weights)
                           for f in frames], axis=0)

    observed = (np.mean([_care(f, options) for f in tables[a]]) - np.mean([_care(f, options) for f in tables[b]]))
    diffs = mean_care(tables[a]) - mean_care(tables[b])
    low, high = np.nanpercentile(diffs, [2.5, 97.5])
    return {"a": a, "b": b, "difference": float(observed), "ci_low": float(low), "ci_high": float(high),
            "excludes_zero": bool(low > 0 or high < 0), "n_boot": n_boot,
            "n_seeds_a": len(tables[a]), "n_seeds_b": len(tables[b])}


MU_EXPERIMENTS = {"F3_mu0": 0.0, "F3_mu0.001": 0.001, "F3_mu0.01": 0.01, "F3_mu0.1": 0.1, "F3_mu1": 1.0,
                  "F3_mu10": 10.0}
CURVES = {"F1": ("fedavg/rounds.csv", "FedAvg"), "central": ("centralized/epochs.csv", "Centralised (pooled)"),
          "central_sequential": ("centralized/epochs.csv", "Centralised (sequential)"),
          "F3_mu0": ("fedprox/rounds.csv", "FedProx μ=0"), "F3_mu0.1": ("fedprox/rounds.csv", "FedProx μ=0.1"),
          "F3_mu1": ("fedprox/rounds.csv", "FedProx μ=1"), "F3_mu10": ("fedprox/rounds.csv", "FedProx μ=10")}


def plot_figure4(root, path):
    """Fig. 4: (a) validation MSE per round, mean over seeds; (b) CARE against μ, mean ± SD."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    runs = find_runs(root)
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, (relative, label) in CURVES.items():
        curves = [pd.read_csv(p / relative).set_index("epoch")["validation_mse"]
                  for p in runs.loc[runs.experiment == name, "path"] if (p / relative).is_file()]
        if curves:
            mean = pd.concat(curves, axis=1).mean(axis=1)
            left.plot(mean.index, mean.values, label=f"{label} (n={len(curves)})",
                      linestyle="--" if name.startswith("central") else "-")
    left.set_yscale("log")
    left.set_xlabel("Round (epoch for the centralised comparator)")
    left.set_ylabel("Validation error (MSE)")
    left.set_title("(a) Validation error during training")
    left.legend(fontsize=7)
    mu = runs[runs.experiment.isin(MU_EXPERIMENTS)].assign(mu=lambda f: f.experiment.map(MU_EXPERIMENTS))
    if not mu.empty:
        stats = mu.groupby("mu")["care"].agg(["mean", "std"]).reset_index()
        positions = np.arange(len(stats))
        right.errorbar(positions, stats["mean"], yerr=stats["std"].fillna(0), marker="s", capsize=3, label="FedProx")
        right.set_xticks(positions, [f"{m:g}" for m in stats["mu"]])
        for name, style in (("D", ":"), ("central", "-."), ("F1", "--")):
            values = runs.loc[runs.experiment == name, "care"]
            if len(values):
                right.axhline(values.mean(), linestyle=style, color="gray",
                              label=f"{name} ({values.mean():.3f}, n={len(values)})")
        right.set_xlabel("Proximal strength μ")
        right.set_ylabel("CARE")
        right.set_title("(b) CARE against proximal strength")
        right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
