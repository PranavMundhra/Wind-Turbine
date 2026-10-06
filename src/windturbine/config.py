"""Run configuration.

Defaults mirror the constants at the top of ``notebooks/10_arm_isolated.ipynb`` and
``notebooks/11_arm_pooled.ipynb`` (both arms use the same values). YAML files in
``configs/`` override them, so a run is described by a file rather than by edited
notebook cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    # data
    farm: str = "Wind Farm C"
    dataset_root: str = "data/raw"  # the notebooks hard-code a Kaggle path; set this per environment
    cache_dir: str = "data/cache"
    use_avg_only: bool = True
    drop_reactive: bool = False
    angle_transform: bool = True
    max_nan_fraction: float = 0.30
    normal_status_ids: tuple = (0, 2)
    wind_speed_cutin: float = 3.0
    wind_speed_rated: float = 12.0
    power_near_zero_thresh: float = 0.02
    # model
    hidden_layers: tuple = (133, 83, 20, 83, 133)
    learning_rate: float = 0.0056
    use_lr_plateau: bool = True
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 4
    noise_std: float = 0.0
    batch_size: int = 64
    max_epochs: int = 200
    early_stop_patience: int = 10
    seed: int = 42
    # validation / threshold
    val_fraction: float = 0.15
    val_split_mode: str = "blocked"  # blocked | chronological | random
    val_block_hours: int = 120
    score_quantile: float = 0.99
    # scoring
    criticality_threshold: int = 72
    beta: float = 0.5
    care_weights: tuple = (1, 1, 1, 2)  # coverage, earliness, reliability, accuracy
    # run
    arm: str = "isolated"  # isolated | pooled | pooled_dedup | federated
    output_dir: str = "results/runs"
    extra: dict = field(default_factory=dict)

    def validate(self) -> Config:
        if self.val_split_mode not in {"blocked", "chronological", "random"}:
            raise ValueError(f"val_split_mode must be blocked|chronological|random, got {self.val_split_mode!r}")
        if not 0.0 < self.score_quantile < 1.0:
            raise ValueError("score_quantile must be in (0, 1)")
        if len(self.care_weights) != 4:
            raise ValueError("care_weights must have 4 entries")
        return self


def _coerce(name: str, value: Any) -> Any:
    # YAML lists -> tuples for the fields that are tuples in the dataclass.
    if name in {"hidden_layers", "care_weights", "normal_status_ids"} and isinstance(value, list):
        return tuple(value)
    return value


def load_config(*paths: str | Path) -> Config:
    """Load defaults, then apply each YAML file in order (later files win)."""
    cfg = Config()
    known = {f.name for f in fields(Config)}
    for p in paths:
        data = yaml.safe_load(Path(p).read_text()) or {}
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"{p}: unknown config keys {sorted(unknown)}")
        cfg = replace(cfg, **{k: _coerce(k, v) for k, v in data.items()})
    return cfg.validate()
