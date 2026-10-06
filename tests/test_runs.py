import json

import pandas as pd
import pytest

from compare_runs import load_runs, pair_diffs, summarise
from windturbine.config import Config
from windturbine.paths import resolve_data_root
from windturbine.runs import (
    RESULT_KEYS,
    config_hash,
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


def test_compare_runs_summarises_and_diffs(tmp_path):
    for exp, cares in {"A": [0.62, 0.63], "B": [0.58]}.items():
        for i, c in enumerate(cares):
            run = new_run_dir(exp, Config(seed=i), root=tmp_path)
            write_manifest(run, exp, Config(seed=i))
            write_results(run, _results(c))
    s = summarise(load_runs(tmp_path))
    assert s.loc["A", "n_runs"] == 2 and s.loc["A", "care"] == pytest.approx(0.625)
    d = pair_diffs(s, ["B-A", "C-A"])
    assert d.loc[0, "care_diff"] == pytest.approx(-0.045, abs=1e-3)
    assert pd.isna(d.loc[1, "care_diff"])  # C has no runs yet


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
