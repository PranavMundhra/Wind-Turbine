"""Shared helpers for experiments/registry.yaml (used by check_registry and render_graph)."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.yaml"
REQUIRED = ("id", "title", "kind", "phase", "status", "importance", "question", "doc",
            "depends_on", "compares_with")
STATUSES = {"done", "done-unverified", "partial", "planned"}
IMPORTANCE = {"gate", "headline", "baseline", "supporting"}


def load(path: Path = REGISTRY) -> list[dict]:
    return yaml.safe_load(path.read_text())["experiments"]


def by_id(exps: list[dict]) -> dict[str, dict]:
    return {e["id"]: e for e in exps}


def descendants(exps: list[dict]) -> dict[str, set[str]]:
    """Experiments that (transitively) depend on each experiment: what it unblocks."""
    children: dict[str, set[str]] = {e["id"]: set() for e in exps}
    for e in exps:
        for d in e.get("depends_on", []):
            children.setdefault(d, set()).add(e["id"])
    out = {}
    for eid, kids in children.items():
        seen, stack = set(), list(kids)
        while stack:
            n = stack.pop()
            if n not in seen:
                seen.add(n)
                stack.extend(children.get(n, ()))
        out[eid] = seen
    return out


def find_cycle(exps: list[dict]) -> list[str] | None:
    deps = {e["id"]: list(e.get("depends_on", [])) for e in exps}
    state: dict[str, int] = {}

    def visit(n, path):
        state[n] = 1
        for d in deps.get(n, []):
            if state.get(d) == 1:
                return [*path, n, d]
            if state.get(d) is None:
                cyc = visit(d, [*path, n])
                if cyc:
                    return cyc
        state[n] = 2
        return None

    for n in deps:
        if state.get(n) is None:
            cyc = visit(n, [])
            if cyc:
                return cyc
    return None
