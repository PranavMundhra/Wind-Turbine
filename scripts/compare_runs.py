"""Compare experiment runs from results/runs/ (replaces cross-notebook /kaggle/working reads).

    python scripts/compare_runs.py                     # mean, sd, n per experiment
    python scripts/compare_runs.py --pairs B-A C-B F2-D # differences of means
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

METRICS = ["care", "coverage", "earliness", "reliability", "accuracy"]


def load_runs(root: str | Path = "results/runs") -> pd.DataFrame:
    rows = []
    for res in sorted(Path(root).glob("*/*/results.json")):
        man_path = res.parent / "manifest.json"
        man = json.loads(man_path.read_text()) if man_path.exists() else {}
        r = json.loads(res.read_text())
        rows.append({"experiment": res.parent.parent.name, "run_id": res.parent.name,
                     "seed": man.get("seed"), "git_dirty": man.get("git", {}).get("dirty"),
                     **{m: r.get(m) for m in METRICS}})
    return pd.DataFrame(rows)


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    g = runs.groupby("experiment")
    out = g[METRICS].mean().round(3)
    out["care_sd"] = g["care"].std().round(3)
    out["n_runs"] = g.size()
    return out


def pair_diffs(summary: pd.DataFrame, pairs) -> pd.DataFrame:
    rows = []
    for p in pairs:
        a, b = p.split("-")
        if a in summary.index and b in summary.index:
            rows.append({"pair": f"{a} - {b}", "care_diff": round(summary.loc[a, "care"] - summary.loc[b, "care"], 3)})
        else:
            rows.append({"pair": f"{a} - {b}", "care_diff": None})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/runs")
    ap.add_argument("--pairs", nargs="*", default=[])
    args = ap.parse_args(argv)
    runs = load_runs(args.root)
    if runs.empty:
        print(f"No runs under {args.root}.")
        return 0
    if runs["git_dirty"].fillna(False).any():
        print("WARNING: some runs came from a dirty working tree; they are not exactly reproducible.")
    s = summarise(runs)
    print(s.to_string())
    if args.pairs:
        print(pair_diffs(s, args.pairs).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
