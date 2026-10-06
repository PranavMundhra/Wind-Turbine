"""Fail if the repo and experiments/registry.yaml disagree, or a markdown link is broken.

Checks: required fields and allowed values; unique ids; depends_on / compares_with point
at real experiments; no dependency cycle; every notebook, config, experiment doc exists
and is registered; each registered notebook's first cell names its experiment id; every
relative link in README.md and docs/**/*.md resolves.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import IMPORTANCE, REQUIRED, ROOT, STATUSES, find_cycle, load

LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
DOC_EXEMPT = {"README.md", "EXPERIMENT_GRAPH.md", "_TEMPLATE.md"}


def check(root: Path = ROOT) -> list[str]:
    errs: list[str] = []
    exps = load(root / "experiments" / "registry.yaml")
    ids = [e.get("id") for e in exps]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        errs.append(f"duplicate ids: {sorted(dupes)}")
    known = set(ids)
    for e in exps:
        eid = e.get("id", "?")
        for f in REQUIRED:
            if f not in e:
                errs.append(f"{eid}: missing field '{f}'")
        if e.get("status") not in STATUSES:
            errs.append(f"{eid}: status {e.get('status')!r} not in {sorted(STATUSES)}")
        if e.get("importance") not in IMPORTANCE:
            errs.append(f"{eid}: importance {e.get('importance')!r} not in {sorted(IMPORTANCE)}")
        for f in ("depends_on", "compares_with"):
            for ref in e.get(f, []):
                if ref not in known:
                    errs.append(f"{eid}: {f} references unknown experiment {ref!r}")
        for f in ("notebook", "config", "doc"):
            p = e.get(f)
            if p and not (root / p).exists():
                errs.append(f"{eid}: {f} {p} does not exist")
        nb = e.get("notebook")
        if nb and (root / nb).exists():
            cells = json.loads((root / nb).read_text())["cells"]
            first = "".join(cells[0]["source"]) if cells else ""
            if f"Experiment {eid}" not in first:
                errs.append(f"{eid}: first cell of {nb} must name 'Experiment {eid}' and link its doc")
    cyc = find_cycle(exps)
    if cyc:
        errs.append(f"dependency cycle: {' -> '.join(cyc)}")

    registered = {e.get(f) for e in exps for f in ("notebook", "config", "doc") if e.get(f)}
    for nb in sorted((root / "notebooks").glob("*.ipynb")):
        rel = nb.relative_to(root).as_posix()
        if rel not in registered:
            errs.append(f"unregistered notebook {rel} (register it, or move it to notebooks/archive/)")
    for cfg in sorted((root / "configs").glob("*.yaml")):
        rel = cfg.relative_to(root).as_posix()
        if cfg.name != "base.yaml" and rel not in registered:
            errs.append(f"unregistered config {rel}")
    for doc in sorted((root / "docs" / "experiments").glob("*.md")):
        rel = doc.relative_to(root).as_posix()
        if doc.name not in DOC_EXEMPT and rel not in registered:
            errs.append(f"unregistered experiment doc {rel}")

    md_files = [root / "README.md", *sorted((root / "docs").rglob("*.md")),
                *sorted((root / "results").rglob("*.md"))]
    for md in md_files:
        if not md.exists():
            continue
        text = re.sub(r"```.*?```", "", md.read_text(), flags=re.DOTALL)  # ignore code blocks
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            path = target.split("#")[0]
            if path and not (md.parent / path).exists():
                errs.append(f"broken link in {md.relative_to(root)}: {target}")
    return errs


def main() -> int:
    errs = check()
    for e in errs:
        print("ERROR:", e)
    print(f"registry check: {len(errs)} problem(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
