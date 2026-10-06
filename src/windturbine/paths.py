"""Resolve where the data lives, so no notebook has to hard-code a Kaggle path.

Order: explicit argument > ``WT_DATA_ROOT`` environment variable > ``Config.dataset_root``
(if it exists) > first directory under ``/kaggle/input`` that contains the farm folder.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "WT_DATA_ROOT"
KAGGLE_INPUT = Path("/kaggle/input")


def resolve_data_root(explicit: str | Path | None = None, cfg=None, farm: str = "Wind Farm C",
                      kaggle_input: Path = KAGGLE_INPUT) -> Path:
    candidates: list[tuple[str, Path]] = []
    if explicit:
        candidates.append(("argument", Path(explicit)))
    if os.environ.get(ENV_VAR):
        candidates.append((ENV_VAR, Path(os.environ[ENV_VAR])))
    if cfg is not None and getattr(cfg, "dataset_root", None):
        candidates.append(("config", Path(cfg.dataset_root)))
    for _source, p in candidates:
        if p.exists():
            return p
    if kaggle_input.exists():
        for hit in sorted(kaggle_input.rglob(farm)):
            if hit.is_dir():
                return hit.parent
    tried = ", ".join(f"{s}={p}" for s, p in candidates) or "nothing configured"
    raise FileNotFoundError(
        f"Could not find the dataset ({tried}; no '{farm}' under {kaggle_input}). "
        f"Set {ENV_VAR} or dataset_root in configs/base.yaml."
    )
