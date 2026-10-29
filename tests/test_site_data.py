"""The demo site reads site/data.js, which is built from the committed files in results/paper/."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


def _load_builder():
    spec = importlib.util.spec_from_file_location("site_build_data", SITE / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data(text):
    return json.loads(re.sub(r"^window\.WT_DATA = |;\n$", "", text))


@pytest.fixture(scope="module")
def data():
    return _data(_load_builder().build())


def test_data_js_matches_the_committed_run_files():
    built = _load_builder().build()
    committed = (SITE / "data.js").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert committed == built, "site/data.js is stale: run `python site/build_data.py`"


def test_data_covers_every_experiment_and_event(data):
    ids = [exp for group in data["groups"].values() for exp in group]
    assert set(ids) == set(data["experiments"])
    assert data["meta"]["reference"] in data["experiments"]
    assert data["meta"]["n_events"] == len(data["events"]) == 58
    assert data["meta"]["n_anomaly"] + data["meta"]["n_normal"] == data["meta"]["n_events"]
    for exp, row in data["experiments"].items():
        assert len(row["hits"]) == data["meta"]["n_events"], exp
        assert all(0 <= h <= len(row["seeds"]) for h in row["hits"]), exp
        assert [p["seed"] for p in row["per_seed"]] == row["seeds"], exp
        if exp != data["meta"]["reference"]:
            assert row["ci"] is not None, f"{exp} has no interval against the reference"


def test_care_formula_on_the_page_reproduces_every_published_score(data):
    """CARE = (C + E + R + 2A) / 5 must give the published score for each run and each experiment mean."""
    w = data["meta"]["care"]["weights"]
    assert w == [1, 1, 1, 2]
    for exp, row in data["experiments"].items():
        for result in [row["metrics"]] + row["per_seed"]:
            c, e, r, a = (result[k] for k in ("coverage", "earliness", "reliability", "accuracy"))
            assert a >= 0.5, f"{exp}: the Accuracy < 0.5 rule would apply, so the plain formula does not"
            recomputed = (w[0] * c + w[1] * e + w[2] * r + w[3] * a) / sum(w)
            assert recomputed == pytest.approx(result["care"], abs=2e-6), exp


def test_reliability_is_f_beta_of_the_event_counts(data):
    beta = data["meta"]["care"]["beta"]
    for exp, row in data["experiments"].items():
        for p in row["per_seed"]:
            f = (1 + beta**2) * p["tp"] / ((1 + beta**2) * p["tp"] + beta**2 * p["fn"] + p["fp"])
            assert f == pytest.approx(p["reliability"], abs=2e-6), f"{exp} seed {p['seed']}"


def test_constants_on_the_page_come_from_the_scoring_config(data):
    care = data["meta"]["care"]
    assert care == {"beta": 0.5, "score_quantile": 0.99, "criticality_threshold": 72, "weights": [1, 1, 1, 2]}


def test_turbine_simulator_scores_the_same_statuses_as_the_runs():
    """The Test a turbine tab scores only the normal statuses; every committed run must agree."""
    js = (SITE / "assets" / "simulator.js").read_text(encoding="utf-8")
    normal = json.loads(re.search(r"NORMAL_STATUS = (\[[\d, ]*\])", js).group(1))
    manifests = sorted(ROOT.glob("results/paper/*/s*/manifest.json"))
    assert manifests
    for path in manifests:
        assert json.loads(path.read_text(encoding="utf-8"))["config"]["normal_status_ids"] == normal, path


def test_page_only_references_files_that_exist():
    html = (SITE / "index.html").read_text(encoding="utf-8")
    local = re.findall(r'(?:src|href)="((?!https?:|data:|#)[^"]+)"', html)
    assert {"data.js", "assets/story.js", "assets/maths.js", "assets/simulator.js", "assets/app.js", "assets/style.css"} <= set(local)
    for ref in local:
        assert (SITE / ref).is_file(), f"site/index.html references missing file {ref}"
