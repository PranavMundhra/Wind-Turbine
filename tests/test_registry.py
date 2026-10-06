import json
import shutil
from pathlib import Path

import pytest
import yaml

from check_registry import check
from registry import descendants, find_cycle, load
from render_graph import index_table, mermaid

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path):
    dst = tmp_path / "repo"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.egg-info",
                                                              ".pytest_cache", "*.png", "*.pdf"))
    # keep link targets that were skipped by the ignore pattern
    for p in ROOT.rglob("*.png"):
        q = dst / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.touch()
    for p in ROOT.rglob("*.pdf"):
        (dst / p.relative_to(ROOT)).touch()
    return dst


def test_real_repo_is_consistent():
    assert check(ROOT) == []


def test_unregistered_notebook_is_caught(repo):
    (repo / "notebooks" / "99_scratch.ipynb").write_text(json.dumps({"cells": []}))
    assert any("unregistered notebook" in e for e in check(repo))


def test_broken_link_is_caught(repo):
    p = repo / "docs" / "experiments" / "C_pooled_dedup.md"
    p.write_text(p.read_text() + "\nSee [missing](nowhere.md).\n")
    assert any("broken link" in e for e in check(repo))


def test_notebook_without_header_cell_is_caught(repo):
    nb = repo / "notebooks" / "10_arm_isolated.ipynb"
    d = json.loads(nb.read_text())
    d["cells"] = d["cells"][1:]
    nb.write_text(json.dumps(d))
    assert any("must name 'Experiment A'" in e for e in check(repo))


def test_cycle_and_unknown_dependency_are_caught(repo):
    reg = repo / "experiments" / "registry.yaml"
    data = yaml.safe_load(reg.read_text())
    for e in data["experiments"]:
        if e["id"] == "B":  # F4 -> F2 -> F1 -> C -> B, so B -> F4 closes a cycle
            e["depends_on"] = ["F4"]
        if e["id"] == "A":
            e["depends_on"] = ["NOPE"]
    reg.write_text(yaml.safe_dump(data))
    errs = check(repo)
    assert any("dependency cycle" in e for e in errs)
    assert any("unknown experiment 'NOPE'" in e for e in errs)


def test_graph_shape():
    exps = load()
    assert find_cycle(exps) is None
    down = descendants(exps)
    # the gates must sit upstream of the headline experiments
    for gate in ("E02", "E03", "C", "D", "F1"):
        assert {"F2", "F3"} <= down[gate], gate
    assert down["F4"] == set()
    m = mermaid(exps)
    assert m.startswith("```mermaid") and "E02 ==> C" in m and "F2 -.-|vs| C" in m
    assert index_table(exps).count("\n") == len(exps) + 1
