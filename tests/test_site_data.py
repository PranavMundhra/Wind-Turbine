"""The demo site reads site/data.js, which is built from the committed files in results/paper/."""
import importlib.util
import json
import re
from pathlib import Path

SITE = Path(__file__).resolve().parents[1] / "site"


def _load_builder():
    spec = importlib.util.spec_from_file_location("site_build_data", SITE / "build_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _data(text):
    return json.loads(re.sub(r"^window\.WT_DATA = |;\n$", "", text))


def test_data_js_matches_the_committed_run_files():
    built = _load_builder().build()
    committed = (SITE / "data.js").read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    assert committed == built, "site/data.js is stale: run `python site/build_data.py`"


def test_data_covers_every_experiment_and_event():
    data = _data(_load_builder().build())
    ids = [exp for group in data["groups"].values() for exp in group]
    assert set(ids) == set(data["experiments"])
    assert data["meta"]["reference"] in data["experiments"]
    assert data["meta"]["n_events"] == len(data["events"]) == 58
    assert data["meta"]["n_anomaly"] + data["meta"]["n_normal"] == data["meta"]["n_events"]
    event_ids = {e["id"] for e in data["events"]}
    for exp, row in data["experiments"].items():
        assert set(row["detected_events"]) <= event_ids, exp
        if exp != data["meta"]["reference"]:
            assert row["ci"] is not None, f"{exp} has no interval against the reference"
