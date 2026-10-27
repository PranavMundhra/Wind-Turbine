"""Command line for the paper's experiments.

    python -m windturbine list
    python -m windturbine build-clients --raw DATA_ROOT --out CLIENTS
    python -m windturbine run D F1 central --seeds 42 43 44 --clients CLIENTS [--jobs 3]
    python -m windturbine summarise --runs results/runs --out results/paper
    python -m windturbine figures

``run`` writes ``<out>/<experiment>/s<seed>/`` and skips any run that already
exists there, so an interrupted batch resumes by running the same command again.
With ``--jobs N`` it runs N experiments at once in separate processes (the
seed-42 runs used three lanes on one RTX 5070 Ti).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from windturbine.experiments import DEFAULT_CONFIG_DIR, experiment_names, load_experiment, run_experiment

PAPER_ORDER = ["D", "central", "central_sequential", "F1", "F2", "F3_mu0", "F3_mu0.001", "F3_mu0.01",
               "F3_mu0.1", "F3_mu1", "F3_mu10", "F4_sigma0", "F4_sigma0.5", "F4_sigma1", "F4_sigma2", "F4_sigma4"]
SMALL_FILES = ("results.json", "per_event.csv", "manifest.json", "rounds.csv", "epochs.csv", "thresholds.csv",
               "privacy.json", "fit_log.csv", "metric_parity.json")


def _source_identity():
    from windturbine.runs import git_info
    return {"git": git_info(Path(__file__).resolve().parents[2])}


def _expand(names, config_dir):
    known = experiment_names(config_dir)
    out = []
    for name in names:
        if name == "all":
            out += [n for n in PAPER_ORDER if n in known and n != "central_sequential"]
        elif name.endswith("*"):
            out += [n for n in PAPER_ORDER if n.startswith(name[:-1]) and n in known]
        elif name in known:
            out.append(name)
        else:
            raise SystemExit(f"Unknown experiment {name!r}. Known: {', '.join(known)}")
    return list(dict.fromkeys(out))


def _overrides(args):
    import yaml

    top, extra = {}, {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        target, key = (extra, key[6:]) if key.startswith("extra.") else (top, key)
        target[key] = yaml.safe_load(value)
    if args.dataset_sha:
        if not args.dataset_reason:
            raise SystemExit("--dataset-sha needs --dataset-reason")
        extra.update(expected_dataset_manifest_sha256=args.dataset_sha, dataset_override_reason=args.dataset_reason)
    if args.rounds:
        extra["rounds"] = args.rounds
    return {**top, "extra": extra} if top or extra else None


def cmd_list(args):
    for name in experiment_names(args.config_dir):
        cfg = load_experiment(name, config_dir=args.config_dir)
        detail = {k: cfg.extra[k] for k in ("strategy", "mu", "dp_noise_multiplier", "central_order",
                                              "rounds", "local_epochs") if k in cfg.extra}
        print(f"{name:<20} runner={cfg.extra['runner']:<10} {detail}")


def cmd_build_clients(args):
    from windturbine.config import load_config
    from windturbine.data.clients import build_client_dataset
    from windturbine.runs import file_hash

    cfg = load_config(Path(args.config_dir) / "base.yaml", Path(args.config_dir) / "clients.yaml")
    build_client_dataset(args.raw, args.out, cfg, source_identity=_source_identity())
    print("Client export:", args.out)
    print("dataset_manifest.json SHA-256:", file_hash(Path(args.out) / "dataset_manifest.json"))


def cmd_run(args):
    names = _expand(args.experiments, args.config_dir)
    jobs = [(name, seed) for seed in args.seeds for name in names]
    todo = [(n, s) for n, s in jobs if not (Path(args.out) / n / f"s{s}" / "COMPLETE").is_file()]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already complete, {len(todo)} to run")
    for name, seed in todo:  # fail fast on a bad config before any training starts
        load_experiment(name, seed=seed, config_dir=args.config_dir, overrides=_overrides(args))
    if args.jobs <= 1:
        for name, seed in todo:
            _run_one(args, name, seed)
        return
    logs = Path(args.out) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running, failed = [], []
    queue = list(todo)
    while queue or running:
        while queue and len(running) < args.jobs:
            name, seed = queue.pop(0)
            command = [sys.executable, "-m", "windturbine", "run", name, "--seeds", str(seed), "--jobs", "1",
                       "--clients", str(args.clients), "--out", str(args.out), "--config-dir", str(args.config_dir)]
            if args.dataset_sha:
                command += ["--dataset-sha", args.dataset_sha, "--dataset-reason", args.dataset_reason]
            if args.rounds:
                command += ["--rounds", str(args.rounds)]
            for item in args.set or []:
                command += ["--set", item]
            log = open(logs / f"{name}_s{seed}.log", "a", encoding="utf-8")  # noqa: SIM115 - closed below
            running.append((name, seed, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
            print(f"[start] {name} seed {seed}  (log: {logs / f'{name}_s{seed}.log'})", flush=True)
        time.sleep(5)
        for item in list(running):
            name, seed, process, log = item
            if process.poll() is not None:
                log.close()
                running.remove(item)
                ok = process.returncode == 0
                print(f"[{'done' if ok else 'FAILED'}] {name} seed {seed}", flush=True)
                if not ok:
                    failed.append((name, seed))
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")
    print("ALL DONE")


def _run_one(args, name, seed):
    output = Path(args.out) / name / f"s{seed}"
    if (output / "COMPLETE").is_file():
        return
    print(f"=== {name} seed {seed} -> {output}", flush=True)
    manifest = run_experiment(name, seed=seed, clients=args.clients, output_root=output,
                              config_dir=args.config_dir, source_identity=_source_identity(),
                              overrides=_overrides(args))
    print(f"=== {name} seed {seed} done in {manifest['runtime_seconds'] / 3600:.2f} h", flush=True)


def cmd_summarise(args):
    import shutil

    import pandas as pd

    from windturbine.summary import bootstrap_difference, find_runs, seed_table

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if Path(args.runs).resolve() != out.resolve():
        # Keep the small, citable files of every complete run; weights and scores stay out of git.
        for complete in sorted(Path(args.runs).glob("*/s*/COMPLETE")):
            run = complete.parent
            target = out / run.parent.name / run.name
            for path in run.rglob("*"):
                if path.is_file() and path.name in SMALL_FILES:
                    destination = target / path.relative_to(run)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    runs = find_runs(out)
    if runs.empty:
        raise SystemExit(f"No runs with results.json and per_event.csv under {out}")
    table = seed_table(runs, PAPER_ORDER)
    table.to_csv(out / "table_I.csv")
    lines = ["| Experiment | Seeds | CARE | Coverage | Earliness | Reliability | Accuracy | Det. | FA |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, row in table.iterrows():
        def cell(metric, digits=3, row=row):
            sd = row[f"{metric}_sd"]
            return f"{row[f'{metric}_mean']:.{digits}f}" + ("" if pd.isna(sd) else f" ± {sd:.{digits}f}")
        lines.append(f"| {name} | {int(row.n_seeds)} | " + " | ".join(
            [cell(m) for m in ("care", "coverage", "earliness", "reliability", "accuracy")]
            + [cell("detected", 1), cell("false_alarms", 1)]) + " |")
    (out / "table_I.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if args.reference and args.reference in set(runs.experiment) and args.boot > 0:
        rows = [bootstrap_difference(runs, name, args.reference, n_boot=args.boot)
                for name in table.index if name != args.reference]
        frame = pd.DataFrame(rows)
        frame.to_csv(out / f"bootstrap_vs_{args.reference}.csv", index=False)
        print(f"\nCARE difference vs {args.reference}, 95% turbine-bootstrap interval:")
        print(frame[["a", "difference", "ci_low", "ci_high", "excludes_zero", "n_seeds_a"]].to_string(index=False))


def cmd_figures(args):
    from windturbine.summary import plot_figure4

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("Wrote", plot_figure4(args.runs, out / "fig4_validation_and_care.pdf"))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m windturbine", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list the paper experiments").set_defaults(func=cmd_list)
    build = sub.add_parser("build-clients", help="build the leak-free per-turbine client export (E03)")
    build.add_argument("--raw", required=True, help="folder containing 'Wind Farm C' from the Zenodo record")
    build.add_argument("--out", required=True)
    build.set_defaults(func=cmd_build_clients)
    run = sub.add_parser("run", help="train and score experiments at one or more seeds")
    run.add_argument("experiments", nargs="+", help="names from 'list', a prefix such as F3*, or 'all'")
    run.add_argument("--seeds", nargs="+", type=int, default=[42])
    run.add_argument("--clients", required=True, help="client export folder containing dataset_manifest.json")
    run.add_argument("--out", default="results/runs")
    run.add_argument("--jobs", type=int, default=1, help="experiments to run at once (separate processes)")
    run.add_argument("--rounds", type=int, help="override the round budget, e.g. 2 for a smoke run")
    run.add_argument("--dataset-sha", help="pin a different client export")
    run.add_argument("--dataset-reason", help="why a different export is used (recorded in the manifest)")
    run.add_argument("--set", action="append", metavar="KEY=VALUE",
                     help="override a config value (YAML syntax), e.g. --set max_epochs=3 --set extra.rounds=2")
    run.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    run.set_defaults(func=cmd_run)
    summary = sub.add_parser("summarise", help="Table I as mean ± SD over seeds, plus bootstrap intervals")
    summary.add_argument("--runs", default="results/runs")
    summary.add_argument("--out", default="results/paper")
    summary.add_argument("--reference", default="D")
    summary.add_argument("--boot", type=int, default=2000)
    summary.set_defaults(func=cmd_summarise)
    figures = sub.add_parser("figures", help="regenerate Fig. 4 from run files")
    figures.add_argument("--runs", default="results/paper")
    figures.add_argument("--out", default="results/paper/figures")
    figures.set_defaults(func=cmd_figures)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
