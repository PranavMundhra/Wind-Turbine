"""Run directories, manifests and the result contract.

Every experiment run writes one directory::

    results/runs/<experiment id>/<UTC time>_<config hash>_s<seed>/
        manifest.json   who/what/when: experiment id, git SHA, dirty flag, config hash,
                        seed, package versions, upstream run ids, data manifest hash
        config.yaml     the exact resolved config
        results.json    the CARE summary (RESULT_KEYS, validated)
        per_event.csv   one row per event (PER_EVENT_COLUMNS, validated)
        scores.parquet  optional: per-row anomaly scores, so thresholds and earliness
                        can be re-scored without retraining (needed by ablation X)

``scripts/compare_runs.py`` reads these directories, which replaces the old pattern of
the pooled notebook reading ``/kaggle/working/results_isolated.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

RESULT_KEYS = ("coverage", "earliness", "reliability", "accuracy", "care", "tp", "fp", "fn", "tn")
PER_EVENT_COLUMNS = ("event_id", "asset_id", "event_label", "detected", "max_criticality")
TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "tensorflow", "keras", "flwr", "pyarrow")


def _plain(obj):
    """Tuples -> lists recursively, so configs serialise identically everywhere."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def config_dict(cfg) -> dict:
    return _plain(asdict(cfg))


def config_hash(cfg) -> str:
    blob = json.dumps(config_dict(cfg), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root: str | Path = ".") -> dict:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def package_versions(names=TRACKED_PACKAGES) -> dict:
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def set_seeds(seed: int, deterministic: bool = True) -> dict:
    """Seed Python, NumPy and (if installed) TensorFlow.

    With ``deterministic=True`` also asks TensorFlow for deterministic ops, which is
    what the documented cuDNN run-to-run noise (CARE 0.570 vs 0.583) calls for. It can
    slow GPU training; record the flag in the manifest either way.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    applied = {"seed": seed, "tensorflow": False, "op_determinism": False}
    try:
        import tensorflow as tf
    except ImportError:
        return applied
    tf.random.set_seed(seed)
    applied["tensorflow"] = True
    if deterministic and hasattr(tf.config.experimental, "enable_op_determinism"):
        tf.config.experimental.enable_op_determinism()
        applied["op_determinism"] = True
    return applied


def new_run_dir(exp_id: str, cfg, root: str | Path = "results/runs") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(root) / exp_id / f"{stamp}_{config_hash(cfg)[:8]}_s{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(run_dir: str | Path, exp_id: str, cfg, upstream: dict | None = None,
                   data_manifest: str | Path | None = None, seeding: dict | None = None,
                   repo_root: str | Path = ".") -> dict:
    """Write manifest.json and config.yaml. ``upstream`` maps experiment id -> run id
    this run consumed (e.g. ``{"E03": "20261004T...", "C": "..."}``)."""
    run_dir = Path(run_dir)
    manifest = {
        "experiment": exp_id,
        "run_id": run_dir.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(repo_root),
        "config_hash": config_hash(cfg),
        "seed": cfg.seed,
        "seeding": seeding,
        "upstream": upstream or {},
        "data_manifest_sha256": file_hash(data_manifest) if data_manifest else None,
        "python": platform.python_version(),
        "packages": package_versions(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config_dict(cfg), sort_keys=True))
    return manifest


def write_results(run_dir: str | Path, results: dict, per_event=None) -> None:
    """Validate and write results.json (and per_event.csv if given)."""
    missing = [k for k in RESULT_KEYS if k not in results]
    if missing:
        raise KeyError(f"results.json is missing {missing}")
    run_dir = Path(run_dir)
    (run_dir / "results.json").write_text(json.dumps(_plain(results), indent=2, default=float))
    if per_event is not None:
        cols = [c for c in PER_EVENT_COLUMNS if c not in per_event.columns]
        if cols:
            raise KeyError(f"per_event is missing columns {cols}")
        per_event.to_csv(run_dir / "per_event.csv", index=False)
