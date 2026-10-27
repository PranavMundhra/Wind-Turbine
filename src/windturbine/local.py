"""Local training (D): one autoencoder per turbine, no collaboration.

Each turbine trains on its own fitting rows from the frozen client export with
the same minibatch step as F1-F4 (``train_epochs``), so D and the federated
approaches share one training implementation and one backend. The schedule is
the one the Kaggle D notebook used through Keras callbacks: Adam at the base
learning rate, multiplied by ``lr_plateau_factor`` after ``lr_plateau_patience``
epochs without a validation improvement of at least 1e-4 (Keras
ReduceLROnPlateau defaults, floor 1e-6), stopped after ``early_stop_patience``
epochs without improvement (Keras EarlyStopping), at most ``max_epochs``, and
the best-validation weights restored. Scoring, thresholds and
publication reuse the federated code path, so D's ``results.json`` and
``per_event.csv`` have exactly the same format and CARE computation as F1-F4.
"""
from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile

import pandas as pd

from windturbine.federated.fedavg import (
    _matrix,
    _plain,
    _prepare_dataset,
    _score_selected,
    _verify_frozen_inputs,
    _write_json,
    save_checkpoint,
)
from windturbine.models import build_autoencoder, mean_squared_error, train_epochs
from windturbine.runs import config_dict, config_hash, file_hash, package_versions


def _fit_with_plateau(model, Xtr, Xva, cfg, *, seed_base):
    """Epoch loop equivalent to Keras EarlyStopping + ReduceLROnPlateau on val MSE."""
    import keras
    import numpy as np

    from windturbine.federated.fedavg import _seed

    history = {"loss": [], "val_loss": [], "lr": []}
    best, best_weights, since_best = np.inf, model.get_weights(), 0
    plateau_best, since_plateau = np.inf, 0
    for epoch in range(1, cfg.max_epochs + 1):
        lr = float(keras.ops.convert_to_numpy(model.optimizer.learning_rate))
        loss = train_epochs(model, Xtr, epochs=1, batch_size=cfg.batch_size, seed=_seed(cfg.seed, epoch, seed_base, 3))
        val = mean_squared_error(model, Xva)
        history["loss"].append(loss)
        history["val_loss"].append(val)
        history["lr"].append(lr)
        if val < best:
            best, best_weights, since_best = val, model.get_weights(), 0
        else:
            since_best += 1
        if cfg.use_lr_plateau:
            if val < plateau_best - 1e-4:
                plateau_best, since_plateau = val, 0
            else:
                since_plateau += 1
                if since_plateau >= cfg.lr_plateau_patience and lr > 1e-6:
                    model.optimizer.learning_rate.assign(max(lr * cfg.lr_plateau_factor, 1e-6))
                    since_plateau = 0
        if since_best >= cfg.early_stop_patience:
            break
    model.set_weights(best_weights)
    return history


def run_local(dataset_root, output_root, cfg, *, source_identity=None, experiment="D"):
    import keras
    import numpy as np

    if cfg.max_epochs <= 0 or cfg.early_stop_patience <= 0:
        raise ValueError("max_epochs and early_stop_patience must be positive")
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed or partial run: {output}")
    keras.utils.set_random_seed(cfg.seed)
    __import__("torch").use_deterministic_algorithms(True)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".local-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        dataset, frozen, cache, counts, val_counts, events = _prepare_dataset(root, stage, cfg)
        n_features = len(dataset["schema"])
        publish = stage / "publish"
        branch_dir = publish / "local"
        (branch_dir / "selected").mkdir(parents=True)
        weights, fit_rows, histories = {}, [], []
        for aid in counts:
            keras.backend.clear_session()
            model = build_autoencoder(n_features, cfg)  # same initial weights for every turbine, as before
            Xtr = np.asarray(_matrix(cache, aid, "train"))
            Xva = np.asarray(_matrix(cache, aid, "val"))
            history = _fit_with_plateau(model, Xtr, Xva, cfg, seed_base=int(aid))
            weights[aid] = model.get_weights()
            save_checkpoint(branch_dir / "selected" / f"client_{aid}.npz", weights[aid])
            best_epoch = int(np.argmin(history["val_loss"])) + 1  # restored by _fit_with_plateau
            fit_rows.append({"asset_id": aid, "n_train": counts[aid], "n_val": val_counts[aid],
                             "epochs_run": len(history["loss"]), "selected_epoch": best_epoch,
                             "selected_validation_mse": mean_squared_error(model, Xva)})
            histories += [{"asset_id": aid, "epoch": i + 1, "loss": loss, "val_loss": val}
                          for i, (loss, val) in enumerate(zip(history["loss"], history["val_loss"]))]
            print(f"Turbine {aid}: {len(history['loss'])} epochs, best epoch {best_epoch}")
            del model, Xtr, Xva
        pd.DataFrame(fit_rows).to_csv(branch_dir / "fit_log.csv", index=False)
        pd.DataFrame(histories).to_csv(branch_dir / "history.csv", index=False)
        selected_mse = sum(r["selected_validation_mse"] * r["n_val"] for r in fit_rows) / sum(val_counts.values())
        info = {"selected_validation_mse": selected_mse, "checkpoint": "local/selected",
                "history": "local/history.csv", "fit_log": "local/fit_log.csv"}
        info.update(_score_selected(root, branch_dir, cfg, weights, dataset["schema"], events))
        _verify_frozen_inputs(root, frozen)
        copyfile(branch_dir / "results.json", publish / "results.json")
        copyfile(branch_dir / "per_event.csv", publish / "per_event.csv")
        manifest = {"experiment": experiment, "run_id": output.name, "config_hash": config_hash(cfg),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "algorithm": "one Keras autoencoder per turbine; early stopping and learning-rate halving",
                    "config": config_dict(cfg), "source_identity": source_identity,
                    "dataset_manifest_sha256": frozen["dataset_manifest.json"],
                    "dataset_source_identity": dataset.get("source_identity"),
                    "dataset_override_reason": cfg.extra.get("dataset_override_reason"),
                    "dataset_hashes_rechecked": True, "clients": list(counts), "n_events": len(events),
                    "fit_counts": counts, "val_counts": val_counts, "schema": dataset["schema"],
                    "seed": cfg.seed, "model_seed": cfg.seed,
                    "partition_seed": dataset["preprocessing_config"]["seed"], "torch_deterministic_algorithms": True,
                    "schema_hash": dataset["schema_hash"], "branches": {"local": info},
                    "packages": package_versions(), "runtime_seconds": time.perf_counter() - started,
                    "implementation_sha256": file_hash(__file__)}
        manifest["artifacts"] = {p.relative_to(publish).as_posix(): file_hash(p)
                                 for p in sorted(publish.rglob("*")) if p.is_file()}
        _write_json(publish / "manifest.json", manifest)
        (publish / "COMPLETE").write_text(file_hash(publish / "manifest.json"), encoding="ascii")
        publish.rename(output)
    return _plain(manifest)
