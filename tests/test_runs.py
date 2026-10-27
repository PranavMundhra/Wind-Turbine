import json

import pandas as pd
import pytest

from windturbine.config import Config
from windturbine.paths import resolve_data_root
from windturbine.runs import (
    RESULT_KEYS,
    config_hash,
    from_evaluate_care,
    new_run_dir,
    set_seeds,
    write_manifest,
    write_results,
)


def _results(care):
    r = dict.fromkeys(RESULT_KEYS, 0.5)
    r["care"] = care
    return r


def test_config_hash_is_stable_and_sensitive():
    assert config_hash(Config()) == config_hash(Config())
    assert config_hash(Config()) != config_hash(Config(seed=7))


def test_run_dir_manifest_and_results(tmp_path):
    cfg = Config(seed=3)
    run = new_run_dir("C", cfg, root=tmp_path)
    assert run.parent.name == "C" and run.name.endswith("_s3")
    data_manifest = tmp_path / "cache.json"
    data_manifest.write_text("{}")
    man = write_manifest(run, "C", cfg, upstream={"B": "run-b"}, data_manifest=data_manifest)
    assert man["upstream"] == {"B": "run-b"} and len(man["data_manifest_sha256"]) == 64
    assert (run / "config.yaml").exists()
    per_event = pd.DataFrame({"event_id": [1], "asset_id": [53], "event_label": ["anomaly"],
                              "detected": [1], "max_criticality": [80]})
    write_results(run, _results(0.57), per_event)
    assert json.loads((run / "results.json").read_text())["care"] == 0.57


def test_result_contract_is_enforced(tmp_path):
    with pytest.raises(KeyError):
        write_results(tmp_path, {"care": 0.6})
    with pytest.raises(KeyError):
        write_results(tmp_path, _results(0.6), pd.DataFrame({"event_id": [1]}))


def test_evaluate_care_output_meets_the_contract(tmp_path):
    # the shape the arm notebooks' evaluate_care() returns
    results = {"Coverage": 0.4, "Earliness": 0.3, "Reliability": 0.6, "Accuracy": 0.9, "CARE": 0.62,
               "event_confusion": {"tp": 1, "fp": 0, "fn": 1, "tn": 2}}
    per_event = pd.DataFrame({"event_id": [1, 2], "asset_id": [10, 10], "is_anomaly": [1, 0],
                              "detected": [1, 0], "peak_criticality": [90, 3]})
    res, pe = from_evaluate_care(results, per_event)
    write_results(tmp_path, res, pe)  # raises if a contract key or column is missing
    assert res["care"] == 0.62 and res["tp"] == 1
    assert pe.event_label.tolist() == ["anomaly", "normal"] and pe.max_criticality.tolist() == [90, 3]


def test_set_seeds_without_tensorflow():
    applied = set_seeds(5)
    assert applied["seed"] == 5


def test_resolve_data_root_order(tmp_path, monkeypatch):
    env_dir, cfg_dir = tmp_path / "env", tmp_path / "cfg"
    env_dir.mkdir()
    cfg_dir.mkdir()
    monkeypatch.setenv("WT_DATA_ROOT", str(env_dir))
    assert resolve_data_root(cfg=Config(dataset_root=str(cfg_dir))) == env_dir
    monkeypatch.delenv("WT_DATA_ROOT")
    assert resolve_data_root(cfg=Config(dataset_root=str(cfg_dir))) == cfg_dir
    kaggle = tmp_path / "kaggle"
    (kaggle / "some-slug" / "Wind Farm C").mkdir(parents=True)
    assert resolve_data_root(kaggle_input=kaggle) == kaggle / "some-slug"
    with pytest.raises(FileNotFoundError):
        resolve_data_root(kaggle_input=tmp_path / "missing")
