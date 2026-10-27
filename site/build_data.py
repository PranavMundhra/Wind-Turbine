"""Build site/data.js from the committed run files in results/paper/.

The demo site reads this one file, so every number on the page comes from Table I,
bootstrap_vs_D.csv and the per-event decisions that ship with the repository. The file is
plain JavaScript (`window.WT_DATA = {...}`) so the page also works when opened from disk.

    python site/build_data.py            # rewrite site/data.js
    python site/build_data.py --check    # exit 1 if site/data.js is stale

Uses only the standard library, so the Pages workflow needs no installation.
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"
OUT = Path(__file__).resolve().parent / "data.js"

REFERENCE = "D"
# (experiment id, label, note) in display order, per explorer tab.
GROUPS = {
    "arms": [
        ("D", "Local training (D)",
         "Each turbine trains alone. Seed 42 was trained on Kaggle with TensorFlow; the federated runs used PyTorch."),
        ("central_sequential", "Centralised comparator",
         "The paper's comparator trains one turbine at a time within each epoch and drifted toward the last turbine."),
        ("F1", "FedAvg", "One global model, local scalers and thresholds."),
        ("F2", "Personalised (FedPer)", "Shared encoder, a decoder kept on each turbine."),
    ],
    "mu": [
        ("F3_mu0", "μ = 0", "No proximal term, so this is FedAvg."),
        ("F3_mu0.001", "μ = 0.001", ""),
        ("F3_mu0.01", "μ = 0.01", ""),
        ("F3_mu0.1", "μ = 0.1", ""),
        ("F3_mu1", "μ = 1", "A strong pull toward the global model reduced Coverage and Earliness."),
        ("F3_mu10", "μ = 10", "The proximal term dominates training."),
    ],
    "priv": [
        ("F4_sigma0", "σ = 0", "Secure aggregation only, no differential-privacy noise."),
        ("F4_sigma0.5", "σ = 0.5", "With noise the models barely trained."),
        ("F4_sigma1", "σ = 1", "With noise the models barely trained."),
        ("F4_sigma2", "σ = 2", "Training was unstable and checkpoint selection kept the initial weights."),
        ("F4_sigma4", "σ = 4", "Training was unstable and checkpoint selection kept the initial weights."),
    ],
}
METRICS = ("care", "coverage", "earliness", "reliability", "accuracy")


def _text(path):
    return path.read_bytes().replace(b"\r\n", b"\n")


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _num(value):
    return round(float(value), 6) if value not in ("", None) else None


def _seed_dirs(exp):
    dirs = sorted((d for d in (PAPER / exp).glob("s*") if d.is_dir()), key=lambda d: int(d.name[1:]))
    if not dirs:
        raise SystemExit(f"no seed folder for {exp} under {PAPER}")
    return dirs


def build():
    table = {r["experiment"]: r for r in _read_csv(PAPER / "table_I.csv")}
    boot = {r["a"]: r for r in _read_csv(PAPER / "bootstrap_vs_D.csv") if r["b"] == REFERENCE}
    digest = hashlib.sha256()
    digest.update(_text(PAPER / "table_I.csv"))
    digest.update(_text(PAPER / "bootstrap_vs_D.csv"))

    experiments, events = {}, {}
    for group, items in GROUPS.items():
        for exp, label, note in items:
            if exp not in table:
                raise SystemExit(f"{exp} is missing from results/paper/table_I.csv")
            row, first = table[exp], _seed_dirs(exp)[0]
            per_event = _read_csv(first / "per_event.csv")
            digest.update(_text(first / "per_event.csv"))
            for e in per_event:
                events[int(e["event_id"])] = {"id": int(e["event_id"]), "asset": int(e["asset_id"]),
                                              "anomaly": e["is_anomaly"] == "1"}
            privacy = sorted(first.glob("*/privacy.json"))
            epsilon = None
            if privacy:
                eps = json.loads(privacy[0].read_text(encoding="utf-8"))["epsilon_by_client"]
                epsilon = round(max(eps.values()), 3)
            sd = {m: _num(row[f"{m}_sd"]) for m in METRICS}
            ci = boot.get(exp)
            experiments[exp] = {
                "label": label, "group": group, "note": note,
                "seeds": [int(s) for s in re.findall(r"\d+", row["seeds"])],
                "metrics": {m: _num(row[f"{m}_mean"]) for m in METRICS},
                "sd": sd if any(v is not None for v in sd.values()) else None,
                "detected": _num(row["detected_mean"]), "false_alarms": _num(row["false_alarms_mean"]),
                "ci": None if ci is None else {
                    "diff": _num(ci["difference"]), "lo": _num(ci["ci_low"]), "hi": _num(ci["ci_high"]),
                    "excludes_zero": ci["excludes_zero"] == "True", "draws": int(ci["n_boot"])},
                "epsilon": epsilon,
                "path": f"results/paper/{exp}/{first.name}/",
                "detected_events": sorted(int(e["event_id"]) for e in per_event if e["detected"] == "1"),
            }
    ordered = [events[k] for k in sorted(events)]
    data = {
        "meta": {
            "reference": REFERENCE,
            "dataset": "CARE to Compare, Wind Farm C (version 6)",
            "n_events": len(ordered),
            "n_anomaly": sum(e["anomaly"] for e in ordered),
            "n_normal": sum(not e["anomaly"] for e in ordered),
            "n_turbines": len({e["asset"] for e in ordered}),
            "fingerprint": digest.hexdigest()[:12],
        },
        "groups": {g: [exp for exp, _, _ in items] for g, items in GROUPS.items()},
        "events": ordered,
        "experiments": experiments,
    }
    return "window.WT_DATA = " + json.dumps(data, ensure_ascii=False, indent=1) + ";\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if site/data.js is out of date")
    args = parser.parse_args(argv)
    text = build()
    if args.check:
        current = OUT.read_bytes().replace(b"\r\n", b"\n").decode("utf-8") if OUT.exists() else ""
        if current != text:
            print("site/data.js is out of date: run `python site/build_data.py`", file=sys.stderr)
            return 1
        print("site/data.js is up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
