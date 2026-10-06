from pathlib import Path

import pytest

from windturbine.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_match_the_paper_and_notebook_values():
    c = Config()
    assert c.hidden_layers == (133, 83, 20, 83, 133)
    assert c.learning_rate == 0.0056 and c.criticality_threshold == 72
    assert c.care_weights == (1, 1, 1, 2) and c.score_quantile == 0.99


def test_arm_configs_load_and_differ_only_where_intended():
    iso = load_config(CONFIGS / "base.yaml", CONFIGS / "arm_isolated.yaml")
    pooled = load_config(CONFIGS / "base.yaml", CONFIGS / "arm_pooled.yaml")
    assert (iso.arm, pooled.arm) == ("isolated", "pooled")
    assert iso.hidden_layers == pooled.hidden_layers
    # In the notebooks the arm alone decides the split level: isolated = blocked in
    # time within an event; pooled = whole turbines held out (val_fraction of 22 -> 3).
    assert iso.val_fraction == pooled.val_fraction == 0.15


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("learnng_rate: 0.1\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_every_arm_config_loads():
    for p in sorted(CONFIGS.glob("*.yaml")):
        if p.name != "base.yaml":
            assert load_config(CONFIGS / "base.yaml", p).arm
