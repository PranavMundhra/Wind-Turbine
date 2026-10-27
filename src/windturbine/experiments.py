"""The paper's experiments: one name per row of Table I, one config file per name.

``configs/paper/<name>.yaml`` may list ``inherits: [file, ...]`` (paths relative to
``configs/paper``). Files are applied in order on top of ``configs/base.yaml``;
later files win, and ``extra`` is merged key by key rather than replaced.
``extra.runner`` picks the training code: ``local`` (D), ``fedavg`` (F1),
``central`` (the matched centralised comparator) or ``federated`` (F2-F4).
"""
from __future__ import annotations

from dataclasses import fields, replace
from pathlib import Path

import yaml

from windturbine.config import Config, _coerce

RUNNERS = ("local", "fedavg", "central", "federated")
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def experiment_names(config_dir=DEFAULT_CONFIG_DIR):
    return sorted(p.stem for p in (Path(config_dir) / "paper").glob("*.yaml") if not p.stem.startswith("_"))


def _layers(path, seen=()):
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular inherits: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = data.pop("inherits", []) or []
    out = []
    for parent in parents:
        out += _layers(path.parent / parent, (*seen, path))
    return out + [(path, data)]


def load_experiment(name, *, seed=None, config_dir=DEFAULT_CONFIG_DIR, overrides=None):
    """Config for one named experiment; ``seed`` and ``overrides['extra']`` win last."""
    config_dir = Path(config_dir)
    path = config_dir / "paper" / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown experiment {name!r}; known: {', '.join(experiment_names(config_dir))}")
    known = {f.name for f in fields(Config)}
    merged, extra = {}, {}
    for source, data in [(config_dir / "base.yaml",
                          yaml.safe_load((config_dir / "base.yaml").read_text(encoding="utf-8")) or {}),
                         *_layers(path), ("overrides", dict(overrides or {}))]:
        unknown = set(data) - known
        if unknown:
            raise KeyError(f"{source}: unknown config keys {sorted(unknown)}")
        extra.update(data.get("extra") or {})
        merged.update({k: _coerce(k, v) for k, v in data.items() if k != "extra"})
    if seed is not None:
        merged["seed"] = int(seed)
    cfg = replace(Config(), **merged, extra=extra).validate()
    if cfg.extra.get("runner") not in RUNNERS:
        raise ValueError(f"{path}: extra.runner must be one of {RUNNERS}")
    return cfg


def run_experiment(name, *, seed, clients, output_root, config_dir=DEFAULT_CONFIG_DIR, source_identity=None,
                   overrides=None):
    """Train and score one experiment at one seed; returns the run manifest."""
    cfg = load_experiment(name, seed=seed, config_dir=config_dir, overrides=overrides)
    runner = cfg.extra["runner"]
    output = Path(output_root)
    if runner == "local":
        from windturbine.local import run_local
        return run_local(clients, output, cfg, source_identity=source_identity, experiment=name)
    if runner in ("fedavg", "central"):
        from windturbine.federated import run_fedavg
        return run_fedavg(clients, output, cfg, source_identity=source_identity, experiment=name,
                          run_centralized=runner == "central", run_federated_branch=runner == "fedavg")
    from windturbine.federated import run_federated
    return run_federated(clients, output, cfg, experiment=name, source_identity=source_identity)
