"""Sequential, in-process Flower FedAvg with a matched centralized comparator.

This is an algorithm simulation: all clients and their data reside in one process.
It provides no network isolation, secure aggregation, or privacy guarantee.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from windturbine.data import clients as client_data
from windturbine.models import build_autoencoder, mean_squared_error, parameter_contract, train_epochs
from windturbine.runs import config_dict, config_hash, file_hash, from_evaluate_care, package_versions
from windturbine.scoring import aggregate_event_metrics, evaluate_care, pick_threshold, reconstruction_error

ACCEPTED_DATASET_MANIFEST_SHA256 = "bb2ed3ffc84adbf79bbba624941090acd91831097f427b38b2e20885d4b0b1d1"


@dataclass(frozen=True)
class ClientUpdate:
    client_id: int
    num_examples: int
    parameters: list
    schema_hash: str


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _check_parameters(parameters, reference):
    if len(parameters) != len(reference):
        raise ValueError("Parameter count does not match the model contract")
    for index, (array, ref) in enumerate(zip(parameters, reference, strict=True)):
        array = np.asarray(array)
        if array.shape != ref.shape or array.dtype != ref.dtype or not np.isfinite(array).all():
            raise ValueError(f"Parameter {index}: wrong shape/dtype or nonfinite values")


def aggregate_updates(updates, *, expected_counts, schema_hash, reference, server_round):
    """Validate a full round, then call Flower's real FedAvg.aggregate_fit path.

    FitRes.num_examples is the actual retained fitting row count, not the event
    count or the number of unique fingerprints. Missing/failed clients abort.
    """
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server.strategy import FedAvg

    _positive_int(server_round, "server_round")
    for count in expected_counts.values():
        _positive_int(count, "expected fitting row count")
    if len(updates) != len(expected_counts) or len({u.client_id for u in updates}) != len(updates):
        raise ValueError("A round requires exactly one update per expected client")
    if {u.client_id for u in updates} != set(expected_counts):
        raise ValueError("Unexpected or missing client IDs")
    results = []
    for update in updates:
        count = _positive_int(update.num_examples, "num_examples")
        if count != expected_counts[update.client_id]:
            raise ValueError(f"Client {update.client_id}: reported count differs from verified fitting rows")
        if update.schema_hash != schema_hash:
            raise ValueError("Client feature schema does not match the broadcast contract")
        _check_parameters(update.parameters, reference)
        response = FitRes(status=Status(code=Code.OK, message=""),
                          parameters=ndarrays_to_parameters(update.parameters), num_examples=count, metrics={})
        # FedAvg only needs the FitRes here; cid identifies the in-process peer.
        results.append((SimpleNamespace(cid=str(update.client_id)), response))
    strategy = FedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
                      min_fit_clients=len(expected_counts), min_available_clients=len(expected_counts),
                      accept_failures=False, inplace=False, fit_metrics_aggregation_fn=lambda _: {})
    parameters, _ = strategy.aggregate_fit(server_round, results, failures=[])
    if parameters is None:
        raise RuntimeError("Flower rejected the full round")
    output = parameters_to_ndarrays(parameters)
    _check_parameters(output, reference)
    return output


def save_checkpoint(path, weights):
    if not weights or not all(np.isfinite(w).all() for w in weights):
        raise ValueError("Checkpoint parameters must be finite and nonempty")
    np.savez_compressed(path, **{f"parameter_{index:04d}": value for index, value in enumerate(weights)})


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as stored:
        names = [f"parameter_{index:04d}" for index in range(len(stored.files))]
        if sorted(stored.files) != names:
            raise ValueError("Invalid checkpoint parameter order")
        weights = [stored[name] for name in names]
    if not weights or not all(np.isfinite(w).all() for w in weights):
        raise ValueError("Invalid checkpoint parameters")
    return weights


def _assert_metrics_equal(left, right):
    for key in ("Coverage", "Earliness", "Reliability", "Accuracy", "CARE"):
        np.testing.assert_allclose(left[key], right[key], rtol=0, atol=1e-12, equal_nan=True)
    for key in ("support", "event_confusion", "n_anomaly_events", "n_normal_events"):
        if left[key] != right[key]:
            raise ValueError(f"Direct/server CARE disagreement: {key}")


def run_preflight():
    """Check actual Flower aggregation against full-batch centralized SGD.

    Equivalence holds for one full-batch SGD step from the same broadcast and
    sample-weighted squared loss. It does not claim Adam/local-epoch equivalence.
    """
    rng = np.random.default_rng(810)
    counts = {1: 3, 2: 5, 3: 8}
    data = {aid: (rng.normal(size=(n, 3)), rng.normal(size=n)) for aid, n in counts.items()}
    global_parameters = [np.array([0.2, -0.1, 0.4], dtype=np.float64), np.array([0.05], dtype=np.float64)]
    central_parameters = [p.copy() for p in global_parameters]
    errors = []
    for server_round in range(1, 4):
        updates = []
        for aid, (x, y) in data.items():
            residual = x @ global_parameters[0] + global_parameters[1][0] - y
            gradients = [2 * x.T @ residual / len(x), np.array([2 * residual.mean()])]
            updated = [p - 0.03 * gradient for p, gradient in zip(global_parameters, gradients, strict=True)]
            updates.append(ClientUpdate(aid, len(x), updated, "linear-v1"))
        global_parameters = aggregate_updates(updates, expected_counts=counts, schema_hash="linear-v1",
                                              reference=global_parameters, server_round=server_round)
        union_x = np.concatenate([part[0] for part in data.values()])
        union_y = np.concatenate([part[1] for part in data.values()])
        residual = union_x @ central_parameters[0] + central_parameters[1][0] - union_y
        gradients = [2 * union_x.T @ residual / len(union_x), np.array([2 * residual.mean()])]
        central_parameters = [p - 0.03 * gradient for p, gradient in zip(central_parameters, gradients, strict=True)]
        for global_weight, central_weight in zip(global_parameters, central_parameters, strict=True):
            errors.append(float(np.max(np.abs(global_weight - central_weight))))
            np.testing.assert_allclose(global_weight, central_weight, rtol=0, atol=1e-12)
    times = pd.date_range("2020-01-01", periods=8, freq="10min").to_numpy()
    meta, records, reports = {}, {}, []
    for eid, anomaly in ((1, False), (2, True), (3, False)):
        info = {"asset_id": eid, "is_anomaly": anomaly,
                "event_start": pd.Timestamp(times[0]), "event_end": pd.Timestamp(times[-1])}
        rec = {"time_stamp": times, "normal_mask": np.ones(8, dtype=bool),
               "prediction": np.full(8, int(anomaly), dtype=np.int8)}
        meta[eid], records[eid] = info, rec
        _, report = evaluate_care({eid: info}, {eid: rec}, criticality_threshold=2, verbose=False)
        reports.append(report)
    direct, _ = evaluate_care(meta, records, criticality_threshold=2, verbose=False)
    combined = aggregate_event_metrics(pd.concat(reports), expected_event_ids=set(meta))
    _assert_metrics_equal(direct, combined)
    return {"passed": True,
            "sgd_equivalence": {"passed": True, "rounds": 3, "client_rows": list(counts.values()),
                                "parameter_arrays": 2, "max_abs_error": max(errors), "tolerance": 1e-12},
            "care_parity": {"passed": True, "events": 3, "results": direct},
            "tensorflow_compatibility": _tensorflow_compatibility_gate()}


def _tensorflow_compatibility_gate():
    import keras

    from windturbine.config import Config

    cfg = Config(hidden_layers=(3, 2, 3), learning_rate=0.001, batch_size=8, use_lr_plateau=False)
    rng = np.random.default_rng(915)
    counts = {1: 3, 2: 5, 3: 8}
    keras.backend.clear_session()
    model = build_autoencoder(2, cfg)
    initial = model.get_weights()
    del model
    updates = []
    for aid, count in counts.items():
        keras.backend.clear_session()
        model = build_autoencoder(2, cfg, seed=aid)
        model.set_weights(initial)
        x = rng.normal(size=(count, 2)).astype(np.float32)
        train_epochs(model, x, epochs=1, batch_size=8, seed=aid)
        updates.append(ClientUpdate(aid, count, model.get_weights(), "tensorflow-gate"))
        del model
    aggregated = aggregate_updates(updates, expected_counts=counts, schema_hash="tensorflow-gate",
                                   reference=initial, server_round=1)
    with tempfile.TemporaryDirectory(prefix="fedavg-preflight-") as temporary:
        path = Path(temporary) / "weights.npz"
        save_checkpoint(path, aggregated)
        restored = load_checkpoint(path)
    for left, right in zip(aggregated, restored, strict=True):
        np.testing.assert_array_equal(left, right)
    keras.backend.clear_session()
    model = build_autoencoder(2, cfg)
    model.set_weights(restored)
    loss = mean_squared_error(model, x)
    if not any(not np.array_equal(one, two) for one, two in zip(initial, restored, strict=True)):
        raise AssertionError("TensorFlow compatibility gate did not update any model parameter")
    del model
    keras.backend.clear_session()
    return {"passed": True, "clients": len(counts), "validation_mse": loss,
            "checkpoint_roundtrip": True, "keras_backend": f"{keras.backend.backend()} {__import__('torch').__version__}"}


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path, data):
    Path(path).write_text(json.dumps(_plain(data), sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")


def _seed(base, step, client, branch=0):
    return int(np.random.SeedSequence([base, step, client, branch]).generate_state(1)[0])


def _verify_frozen_inputs(root, frozen):
    for relative, expected in frozen.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_hash(path) != expected:
            raise ValueError(f"Dataset changed during the run: {relative}")


def _prepare_dataset(root, stage, cfg):
    root = Path(root).resolve()
    manifest_path = root / "dataset_manifest.json"
    expected = cfg.extra.get("expected_dataset_manifest_sha256")
    actual = file_hash(manifest_path)
    reason = cfg.extra.get("dataset_override_reason")
    if expected:
        if actual != expected:
            raise ValueError("Dataset manifest does not match expected_dataset_manifest_sha256")
    elif not isinstance(reason, str) or not reason.strip():
        raise ValueError("Pin expected_dataset_manifest_sha256 or provide an explicit dataset_override_reason")
    if actual != ACCEPTED_DATASET_MANIFEST_SHA256 and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("Alternative dataset requires nonempty dataset_override_reason, even when explicitly pinned")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    requested_preprocessing = client_data.preprocessing_config(cfg)
    # The immutable export fixes its own split seed. A model-seed study does
    # not rebuild partitions or move rows between training and validation.
    requested_preprocessing["seed"] = manifest["preprocessing_config"]["seed"]
    if manifest["preprocessing_config"] != requested_preprocessing:
        raise ValueError("Dataset preprocessing configuration differs from the requested experiment")
    if manifest.get("implementation_sha256") != file_hash(client_data.__file__):
        raise ValueError("Dataset preprocessing source hash differs from this installed implementation")
    expected_clients = _positive_int(cfg.extra.get("expected_clients", 22), "expected_clients")
    expected_events = _positive_int(cfg.extra.get("expected_events", 58), "expected_events")
    ids = manifest["clients"]
    if len(ids) != expected_clients or len(set(ids)) != len(ids) or manifest["n_events"] != expected_events:
        raise ValueError("Dataset cohort does not match the requested client/event counts")
    if cfg.extra.get("n_clients", expected_clients) != expected_clients:
        raise ValueError("n_clients differs from expected_clients")
    frozen = {"dataset_manifest.json": actual, "COMPLETE": file_hash(root / "COMPLETE"), **manifest["artifacts"]}
    # Freeze before checking any clients so a concurrent change cannot become the
    # accepted baseline after an earlier verification pass.
    _verify_frozen_inputs(root, frozen)
    cache = stage / "matrix_cache"
    cache.mkdir()
    counts, val_counts, events = {}, {}, {}
    for aid in ids:
        client = client_data.load_client_dataset(root, aid, verify=True)
        if client["schema"] != manifest["schema"]:
            raise ValueError("Client feature order differs from the shared schema")
        counts[aid], val_counts[aid] = len(client["train"]["X"]), len(client["val"]["X"])
        _positive_int(counts[aid], "retained fitting rows")
        for kind in ("train", "val"):
            np.save(cache / f"{aid}_{kind}.npy", client[kind]["X"], allow_pickle=False)
        for event in client["events"]:
            eid = event["event_id"]
            if eid in events:
                raise ValueError("An event belongs to more than one client")
            events[eid] = event
    if len(events) != expected_events:
        raise ValueError("Exported event metadata does not match the complete evaluation cohort")
    _verify_frozen_inputs(root, frozen)
    return manifest, frozen, cache, counts, val_counts, events


def _matrix(cache, aid, kind):
    return np.load(cache / f"{aid}_{kind}.npy", mmap_mode="r", allow_pickle=False)


def _validation(model, cache, val_counts, epoch, history):
    losses = []
    for aid, count in val_counts.items():
        value = mean_squared_error(model, _matrix(cache, aid, "val"))
        losses.append(value * count)
        history.append({"epoch": epoch, "asset_id": aid, "phase": "global_validation",
                        "loss": value, "num_examples": count})
    return math.fsum(losses) / sum(val_counts.values())


def _score_selected(root, branch_dir, cfg, weights, schema, events):
    """Score every client; ``weights`` is one global model or a dict of per-client models."""
    import keras

    personal = isinstance(weights, dict)
    keras.backend.clear_session()
    model = build_autoencoder(len(schema), cfg)
    if not personal:
        model.set_weights(weights)
    score_dir = branch_dir / "scores"
    score_dir.mkdir()
    records, reports, threshold_rows = {}, [], []
    options = {"beta": cfg.beta, "criticality_threshold": cfg.criticality_threshold,
               "care_weights": cfg.care_weights, "verbose": False}
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for aid in manifest["clients"]:
        if personal:
            model.set_weights(weights[aid])
        client = client_data.load_client_dataset(root, aid, verify=False)
        val_scores = reconstruction_error(model, client["val"]["X"])
        if not np.isfinite(val_scores).all():
            raise ValueError("Nonfinite calibration scores")
        threshold, details = pick_threshold(val_scores, cfg.score_quantile, tag=f"client {aid}")
        threshold_rows.append({"asset_id": aid, "n_val": len(val_scores), **details})
        np.savez_compressed(score_dir / f"client_{aid}_validation.npz", scores=val_scores,
                            event_id=client["val"]["event_id"], time_stamp=client["val"]["time_stamp"])
        part = client["prediction"]
        scores = reconstruction_error(model, part["X"])
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite prediction scores")
        prediction = (scores > threshold).astype(np.int8)
        np.savez_compressed(score_dir / f"client_{aid}_prediction.npz", scores=scores, prediction=prediction,
                            event_id=part["event_id"], time_stamp=part["time_stamp"],
                            eval_mask=part["eval_mask"], ground_truth=part["ground_truth"],
                            status_type_id=part["status_type_id"])
        local_records, local_meta = {}, {}
        for event in client["events"]:
            eid = event["event_id"]
            mask = part["event_id"] == eid
            if not mask.any() or eid in records:
                raise ValueError("Missing or repeated scored event")
            order = np.argsort(part["time_stamp"][mask], kind="stable")
            rec = {"time_stamp": part["time_stamp"][mask][order], "normal_mask": part["eval_mask"][mask][order],
                   "prediction": prediction[mask][order]}
            local_records[eid] = rec
            local_meta[eid] = events[eid]
            records[eid] = rec
        _, per_event = evaluate_care(local_meta, local_records, **options)
        reports.append(per_event)
    reports = pd.concat(reports, ignore_index=True)
    server = aggregate_event_metrics(reports, beta=cfg.beta, care_weights=cfg.care_weights,
                                     expected_event_ids=set(events))
    direct, direct_report = evaluate_care(events, records, **options)
    _assert_metrics_equal(server, direct)
    # Check the per-event terms, not only the final summary.
    pd.testing.assert_frame_equal(reports.sort_values("event_id").reset_index(drop=True),
                                  direct_report.sort_values("event_id").reset_index(drop=True))
    converted, converted_report = from_evaluate_care(server, reports)
    _write_json(branch_dir / "results.json", converted)
    converted_report.to_csv(branch_dir / "per_event.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(branch_dir / "thresholds.csv", index=False)
    _write_json(branch_dir / "metric_parity.json", {"passed": True, "expected_event_ids": sorted(events),
                                                   "server": server, "direct": direct})
    return {"results": server, "metric_parity": True, "n_events": len(events)}


def run_fedavg(dataset_root, output_root, cfg, *, source_identity=None, run_centralized=True):
    """Run full-participation FedAvg and optionally the matched central baseline.

    Local Adam is newly constructed for every client and round; central Adam is
    persistent. Both visit every retained fitting row rounds*local_epochs times.
    Fixed budgets run to completion, then choose the lowest row-weighted normal
    validation MSE. Prediction windows never enter optimization or selection.
    """
    import keras

    if not isinstance(run_centralized, bool):
        raise TypeError("run_centralized must be a boolean")
    rounds = _positive_int(cfg.extra.get("rounds", 100), "rounds")
    local_epochs = _positive_int(cfg.extra.get("local_epochs", 1), "local_epochs")
    _positive_int(cfg.batch_size, "batch_size")
    if cfg.extra.get("global_patience") is not None:
        raise ValueError("F1 uses fixed matched budgets; global_patience must be null")
    if cfg.use_lr_plateau:
        raise ValueError("F1 uses a fixed learning rate; set use_lr_plateau=false for both branches")
    if not 0 < cfg.score_quantile < 1:
        raise ValueError("score_quantile must be between zero and one")
    supported = {"framework": "flwr", "strategy": "fedavg", "fraction_fit": 1.0,
                 "aggregation_weight": "n_rows_after_leak_removal", "scaler": "local", "threshold": "local"}
    for key, value in supported.items():
        if cfg.extra.get(key, value) != value:
            raise ValueError(f"Unsupported F1 setting: {key} must be {value!r}")
    evaluate_care({}, {}, beta=cfg.beta, criticality_threshold=cfg.criticality_threshold,
                  care_weights=cfg.care_weights, verbose=False)
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed or partial run: {output}")
    keras.utils.set_random_seed(cfg.seed)
    __import__("torch").use_deterministic_algorithms(True)
    preflight = run_preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix=".fedavg-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        dataset, frozen, cache, counts, val_counts, events = _prepare_dataset(root, stage, cfg)
        publish = stage / "publish"
        publish.mkdir()
        keras.backend.clear_session()
        model = build_autoencoder(len(dataset["schema"]), cfg)
        initial = model.get_weights()
        contract = parameter_contract(model, dataset["schema"])
        save_checkpoint(publish / "initial_weights.npz", initial)
        _write_json(publish / "model_contract.json", contract)
        global_weights = [w.copy() for w in initial]
        del model
        branch_dir = publish / "fedavg"
        branch_dir.mkdir()
        history, round_rows = [], []
        model = build_autoencoder(len(dataset["schema"]), cfg)
        model.set_weights(initial)
        initial_validation_mse = _validation(model, cache, val_counts, 0, history)
        best_value, best_round = initial_validation_mse, 0
        save_checkpoint(branch_dir / "selected_weights.npz", initial)
        round_rows.append({"epoch": 0, "validation_mse": initial_validation_mse,
                           "participating_clients": 0, "fitting_rows": sum(counts.values()), "row_exposures": 0,
                           "parameters_sha256": _parameter_hash(initial)})
        del model
        fl_started, round_seconds = time.perf_counter(), []
        for server_round in range(1, rounds + 1):
            updates = []
            broadcast_hash = _parameter_hash(global_weights)
            for aid, count in counts.items():
                keras.backend.clear_session()
                seed = _seed(cfg.seed, server_round, aid)
                model = build_autoencoder(len(dataset["schema"]), cfg, seed=seed)
                model.set_weights(global_weights)
                if parameter_contract(model, dataset["schema"]) != contract:
                    raise ValueError("Local model variable contract differs from the shared model")
                if _parameter_hash(model.get_weights()) != broadcast_hash or int(model.optimizer.iterations.numpy()):
                    raise AssertionError("Client did not start from identical broadcast with fresh optimizer")
                loss = train_epochs(model, _matrix(cache, aid, "train"), epochs=local_epochs,
                                    batch_size=cfg.batch_size, seed=seed)
                local_val = mean_squared_error(model, _matrix(cache, aid, "val"))
                updates.append(ClientUpdate(aid, count, model.get_weights(), dataset["schema_hash"]))
                history.append({"epoch": server_round, "asset_id": aid, "phase": "local_fit",
                                "loss": loss, "local_val_loss": local_val, "num_examples": count,
                                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                                "seed": seed, "broadcast_sha256": broadcast_hash})
                del model
            global_weights = aggregate_updates(updates, expected_counts=counts, schema_hash=dataset["schema_hash"],
                                               reference=initial, server_round=server_round)
            keras.backend.clear_session()
            model = build_autoencoder(len(dataset["schema"]), cfg)
            model.set_weights(global_weights)
            value = _validation(model, cache, val_counts, server_round, history)
            round_rows.append({"epoch": server_round, "validation_mse": value,
                               "participating_clients": len(counts), "fitting_rows": sum(counts.values()),
                               "row_exposures": sum(counts.values()) * local_epochs,
                               "parameters_sha256": _parameter_hash(global_weights)})
            if value < best_value:
                best_value, best_round = value, server_round
                save_checkpoint(branch_dir / "selected_weights.npz", global_weights)
            round_seconds.append(time.perf_counter() - fl_started - sum(round_seconds))
            print(f"FedAvg round {server_round}/{rounds}: validation MSE {value:.6g} "
                  f"({round_seconds[-1]:.1f}s)")
            if server_round <= 3:
                _check_time_budget(cfg, rounds, round_seconds, run_centralized)
            del model, updates
        save_checkpoint(branch_dir / "final_weights.npz", global_weights)
        pd.DataFrame(history).to_csv(branch_dir / "history.csv", index=False)
        pd.DataFrame(round_rows).to_csv(branch_dir / "rounds.csv", index=False)
        branches = {"fedavg": {"selected_epoch": best_round, "selected_validation_mse": best_value,
                               "epochs_run": rounds, "checkpoint": "fedavg/selected_weights.npz",
                               "history": "fedavg/history.csv", "rounds": "fedavg/rounds.csv",
                               "initial_validation_mse": initial_validation_mse,
                               "optimizer_steps": sum(row.get("optimizer_steps", 0) for row in history)}}
        if run_centralized:
            central_dir = publish / "centralized"
            central_dir.mkdir()
            keras.backend.clear_session()
            model = build_autoencoder(len(dataset["schema"]), cfg)
            model.set_weights(initial)
            history, epoch_rows = [], [{"epoch": 0, "validation_mse": initial_validation_mse, "row_exposures": 0}]
            _validation(model, cache, val_counts, 0, history)
            best_value, best_epoch = initial_validation_mse, 0
            save_checkpoint(central_dir / "selected_weights.npz", initial)
            budget = rounds * local_epochs
            for epoch in range(1, budget + 1):
                rng = np.random.default_rng(_seed(cfg.seed, epoch, 0, 1))
                for index in rng.permutation(len(counts)):
                    aid = list(counts)[index]
                    loss = train_epochs(model, _matrix(cache, aid, "train"), epochs=1,
                                        batch_size=cfg.batch_size, seed=_seed(cfg.seed, epoch, aid, 1))
                    history.append({"epoch": epoch, "asset_id": aid, "phase": "central_fit",
                                    "loss": loss, "num_examples": counts[aid],
                                    "optimizer_steps": int(model.optimizer.iterations.numpy())})
                # Match the FedAvg selection opportunities when local_epochs>1.
                if epoch % local_epochs:
                    continue
                value = _validation(model, cache, val_counts, epoch, history)
                epoch_rows.append({"epoch": epoch, "validation_mse": value,
                                   "row_exposures": sum(counts.values()) * local_epochs})
                if value < best_value:
                    best_value, best_epoch = value, epoch
                    save_checkpoint(central_dir / "selected_weights.npz", model.get_weights())
                print(f"Centralized epoch {epoch}/{budget}: validation MSE {value:.6g}")
            save_checkpoint(central_dir / "final_weights.npz", model.get_weights())
            pd.DataFrame(history).to_csv(central_dir / "history.csv", index=False)
            pd.DataFrame(epoch_rows).to_csv(central_dir / "epochs.csv", index=False)
            branches["centralized"] = {"selected_epoch": best_epoch, "selected_validation_mse": best_value,
                                       "epochs_run": budget, "checkpoint": "centralized/selected_weights.npz",
                                       "history": "centralized/history.csv", "rounds": "centralized/epochs.csv",
                                       "initial_validation_mse": initial_validation_mse,
                                       "optimizer_steps": int(model.optimizer.iterations.numpy())}
            del model
        # Both training branches finish and freeze their validation-selected
        # checkpoint before any prediction score is computed.
        for name, branch in branches.items():
            weights = load_checkpoint(publish / branch["checkpoint"])
            _check_parameters(weights, initial)
            keras.backend.clear_session()
            model = build_autoencoder(len(dataset["schema"]), cfg)
            model.set_weights(weights)
            readback = _validation(model, cache, val_counts, branch["selected_epoch"], [])
            np.testing.assert_allclose(readback, branch["selected_validation_mse"], rtol=0, atol=1e-12)
            branch["checkpoint_readback_passed"] = True
            del model
            branch.update(_score_selected(root, publish / name, cfg, weights, dataset["schema"], events))
        _verify_frozen_inputs(root, frozen)
        from shutil import copyfile
        copyfile(publish / "fedavg/results.json", publish / "results.json")
        copyfile(publish / "fedavg/per_event.csv", publish / "per_event.csv")
        manifest = {"experiment": "F1", "run_id": output.name, "config_hash": config_hash(cfg),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "algorithm": "Flower server.strategy.FedAvg.aggregate_fit",
                    "config": config_dict(cfg), "source_identity": source_identity,
                    "dataset_manifest_sha256": frozen["dataset_manifest.json"],
                    "dataset_source_identity": dataset.get("source_identity"),
                    "dataset_override_reason": cfg.extra.get("dataset_override_reason"),
                    "dataset_hashes_rechecked": True, "clients": list(counts), "n_events": len(events),
                    "fit_counts": counts, "val_counts": val_counts, "schema": dataset["schema"],
                    "seed": cfg.seed, "model_seed": cfg.seed, "partition_seed": dataset["preprocessing_config"]["seed"],
                    "tensorflow_op_determinism": True,
                    "schema_hash": dataset["schema_hash"], "preflight": preflight, "branches": branches,
                    "initial_weights_sha256": file_hash(publish / "initial_weights.npz"),
                    "local_optimizer_policy": "fresh Adam for every client and round; fixed learning rate",
                    "central_optimizer_policy": "persistent Adam; randomized client order; per-client shuffled minibatches",
                    "seed_policy": "NumPy SeedSequence([base_seed,round_or_epoch,asset_id,branch]); branch 0 FL, 1 central",
                    "selection_policy": "initial checkpoint plus lowest retained-validation-row-weighted MSE; earliest tie; "
                                        "central selection every local_epochs passes; no prediction selection",
                    "budget": {"rounds": rounds, "local_epochs": local_epochs,
                               "union_epochs": rounds * local_epochs,
                               "row_exposures_per_branch": sum(counts.values()) * rounds * local_epochs},
                    "inactive_policy": dataset["inactive_policy"],
                    "simulation": "one process; sequential clients; no networking, isolation or privacy guarantee",
                    "packages": package_versions(), "flower_version": metadata.version("flwr"),
                    "runtime_seconds": time.perf_counter() - started,
                    "implementation_sha256": file_hash(__file__)}
        manifest["artifacts"] = {p.relative_to(publish).as_posix(): file_hash(p)
                                 for p in sorted(publish.rglob("*")) if p.is_file()}
        _write_json(publish / "manifest.json", manifest)
        (publish / "COMPLETE").write_text(file_hash(publish / "manifest.json"), encoding="ascii")
        publish.rename(output)
    return _plain(manifest)


def _check_time_budget(cfg, rounds, round_seconds, run_centralized):
    """Stop early if the measured pace cannot finish inside extra.max_hours.

    Nothing is published until both branches finish, so a Kaggle timeout would
    lose the whole run. A central epoch costs no more than a FedAvg round (it
    skips model rebuilds) and the central branch makes rounds*local_epochs of them,
    so the slowest round so far bounds the central branch at one more FedAvg pass.
    """
    max_hours = cfg.extra.get("max_hours")
    if max_hours is None:
        return
    # FedAvg rounds + central epochs + a fixed allowance for scoring and checkpoints.
    per_round = max(round_seconds) * (2 if run_centralized else 1)
    projected = per_round * rounds / 3600 + 0.25
    print(f"Projected total runtime: {projected:.2f} h (limit {max_hours} h)")
    if projected > max_hours:
        fit = int((max_hours - 0.25) * 3600 / per_round)
        raise RuntimeError(f"Projected {projected:.1f} h exceeds max_hours={max_hours}. "
                           f"Set WT_F1_ROUNDS to at most {max(fit, 1)} or raise max_hours.")


def _parameter_hash(weights):
    from hashlib import sha256

    digest = sha256()
    for weight in weights:
        digest.update(str(weight.dtype).encode())
        digest.update(str(weight.shape).encode())
        digest.update(weight.tobytes(order="C"))
    return digest.hexdigest()
