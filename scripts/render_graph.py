"""Generate the experiment graph and index table from experiments/registry.yaml.

    python scripts/render_graph.py          # rewrite the generated blocks
    python scripts/render_graph.py --check  # exit 1 if they are stale (used in CI)

Writes between <!-- generated:start --> / <!-- generated:end --> markers in
docs/experiments/EXPERIMENT_GRAPH.md (Mermaid) and docs/experiments/README.md (table).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import ROOT, by_id, descendants, load

GRAPH_MD = ROOT / "docs" / "experiments" / "EXPERIMENT_GRAPH.md"
INDEX_MD = ROOT / "docs" / "experiments" / "README.md"
START, END = "<!-- generated:start -->", "<!-- generated:end -->"

LANES = [
    ("lane_r", "Phase R: data and evidence", {"R"}),
    ("lane_done", "Done: baselines", {"done"}),
    ("lane_0", "Phase 0: clean baselines, client data, noise", {"0", "1"}),
    ("lane_fl", "Phases 2-5: federated arms", {"2", "3", "4", "5"}),
    ("lane_any", "Any phase", {"any"}),
]
CLASSES = {
    "gate": "fill:#fff4e5,stroke:#d97706,stroke-width:3px,color:#111",
    "headline": "fill:#e0ecff,stroke:#2563eb,stroke-width:3px,color:#111",
    "baseline": "fill:#eeeeee,stroke:#555,stroke-width:1.5px,color:#111",
    "supporting": "fill:#ffffff,stroke:#999,stroke-width:1px,color:#111",
}


def mermaid(exps: list[dict]) -> str:
    unblocks = descendants(exps)
    idx = by_id(exps)
    lines = ["```mermaid", "flowchart LR"]
    for lane_id, title, phases in LANES:
        members = [e for e in exps if str(e["phase"]) in phases]
        if not members:
            continue
        lines.append(f'  subgraph {lane_id}["{title}"]')
        for e in members:
            n = len(unblocks[e["id"]])
            label = f'{e["id"]}: {e["title"]}<br/>{e["status"]} · unblocks {n}'
            lines.append(f'    {e["id"]}["{label}"]')
        lines.append("  end")
    for e in exps:
        for d in e["depends_on"]:
            arrow = "==>" if idx[d]["importance"] == "gate" else "-->"
            lines.append(f"  {d} {arrow} {e['id']}")
    for e in exps:
        for c in e["compares_with"]:
            if c not in e["depends_on"]:
                lines.append(f"  {e['id']} -.-|vs| {c}")
    for name, style in CLASSES.items():
        lines.append(f"  classDef {name} {style}")
    for name in CLASSES:
        members = [e["id"] for e in exps if e["importance"] == name]
        if members:
            lines.append(f"  class {','.join(members)} {name}")
    lines.append("```")
    return "\n".join(lines)


def index_table(exps: list[dict]) -> str:
    unblocks = descendants(exps)
    rows = ["| ID | Experiment | Phase | Status | Importance | Depends on | Unblocks | Doc |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in exps:
        doc = Path(e["doc"]).name
        deps = ", ".join(e["depends_on"]) or "none"
        rows.append(f'| {e["id"]} | {e["title"]} | {e["phase"]} | {e["status"]} | {e["importance"]} '
                    f'| {deps} | {len(unblocks[e["id"]])} | [{doc}]({doc}) |')
    return "\n".join(rows)


def doc_header(e: dict, exps: list[dict]) -> str:
    idx = by_id(exps)
    unblocks = sorted(descendants(exps)[e["id"]])

    def links(ids):
        return ", ".join(f"[{i}]({Path(idx[i]['doc']).name})" for i in ids) or "none"

    def path(p):
        return f"[`{p}`](../../{p})" if p else "none yet"

    return "\n".join([
        "| Registry field | Value |",
        "| --- | --- |",
        f"| Status | {e['status']} |",
        f"| Phase | {e['phase']} |",
        f"| Importance | {e['importance']} |",
        f"| Depends on | {links(e['depends_on'])} |",
        f"| Judged against | {links(e['compares_with'])} |",
        f"| Unblocks | {links(unblocks)} |",
        f"| Notebook | {path(e.get('notebook'))} |",
        f"| Config | {path(e.get('config'))} |",
        f"| Runs | `results/runs/{e['id']}/` |",
    ])


def splice(text: str, block: str) -> str:
    pat = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if not pat.search(text):
        raise ValueError("generated markers not found")
    return pat.sub(lambda _m: f"{START}\n{block}\n{END}", text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    exps = load()
    stale = []
    targets = [(GRAPH_MD, mermaid(exps)), (INDEX_MD, index_table(exps))]
    targets += [(ROOT / e["doc"], doc_header(e, exps)) for e in exps]
    for path, block in targets:
        old = path.read_text()
        new = splice(old, block)
        if new != old:
            stale.append(path.relative_to(ROOT).as_posix())
            if not args.check:
                path.write_text(new)
    if args.check and stale:
        print("stale generated blocks:", stale, "- run `make graph`")
        return 1
    print("updated:" if stale else "up to date", stale or "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
