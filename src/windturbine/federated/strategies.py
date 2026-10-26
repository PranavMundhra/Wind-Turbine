"""FedPer, FedProx and private federated training on the exact E03 client export.

One sequential, in-process engine serves F2 (shared encoder, local decoder), F3
(FedProx) and F4 (DP-SGD plus simulated secure aggregation). It reuses F1's
dataset freezing, Flower aggregation, scoring and publication. Like F1 it is an
algorithm simulation: all clients run in one process, and the secure
aggregation here checks mask arithmetic, not a deployed protocol.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from shutil import copyfile

import numpy as np
import pandas as pd

from windturbine.models import build_autoencoder, mean_squared_error, parameter_contract, train_epochs
from windturbine.runs import config_dict, config_hash, file_hash, package_versions
from windturbine.scoring import evaluate_care

from .fedavg import (
    ClientUpdate,
    _check_parameters,
    _check_time_budget,
    _matrix,
    _parameter_hash,
    _plain,
    _positive_int,
    _prepare_dataset,
    _score_selected,
    _seed,
    _verify_frozen_inputs,
    _write_json,
    aggregate_updates,
    load_checkpoint,
    run_preflight,
    save_checkpoint,
)

STRATEGIES = ("fedavg", "fedper", "fedprox")
METRICS = ("CARE", "Coverage", "Earliness", "Reliability", "Accuracy")


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return float(value)


def strategy_options(cfg):
    """Validate and normalise the federated settings in ``cfg.extra``."""
    extra = cfg.extra
    strategy = extra.get("strategy")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    supported = {"framework": "flwr", "fraction_fit": 1.0, "aggregation_weight": "n_rows_after_leak_removal",
                 "scaler": "local", "threshold": "local"}
    for key, value in supported.items():
        if extra.get(key, value) != value:
            raise ValueError(f"Unsupported setting: {key} must be {value!r}")
    if extra.get("global_patience") is not None:
        raise ValueError("Fixed budgets only; global_patience must be null")
    if cfg.use_lr_plateau:
        raise ValueError("Use a fixed learning rate; set use_lr_plateau=false")
    if not 0 < cfg.score_quantile < 1:
        raise ValueError("score_quantile must be between zero and one")
    _positive_int(cfg.batch_size, "batch_size")
    mu = _nonnegative(extra.get("mu", 0.0), "mu")
    if strategy == "fedprox" and "mu" not in extra:
        raise ValueError("fedprox needs an explicit mu (choose one value from mu_grid)")
    if strategy != "fedprox" and mu:
        raise ValueError("mu is used only by fedprox")
    shared_layers, local_layers = extra.get("shared_layers"), extra.get("local_layers")
    if strategy == "fedper":
        if not isinstance(shared_layers, list) or not shared_layers:
            raise ValueError("fedper needs a nonempty shared_layers list")
    elif shared_layers is not None or local_layers is not None:
        raise ValueError("shared_layers/local_layers are used only by fedper")
    noise = _nonnegative(extra.get("dp_noise_multiplier", 0.0), "dp_noise_multiplier")
    clip, delta = extra.get("dp_clip_norm"), extra.get("dp_delta")
    if noise:
        if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0:
            raise ValueError("DP-SGD needs a positive dp_clip_norm")
        # YAML reads 1e-6 (no decimal point) as a string; write 1.0e-6.
        if isinstance(delta, bool) or not isinstance(delta, float) or not 0 < delta < 1:
            raise ValueError("DP-SGD needs a float dp_delta in (0, 1), e.g. 1.0e-6")
        if cfg.noise_std:
            raise ValueError("DP-SGD clips a plain Dense stack; set noise_std=0")
    secure = extra.get("secure_aggregation", False)
    if not isinstance(secure, bool):
        raise TypeError("secure_aggregation must be true or false")
    return {"strategy": strategy, "rounds": _positive_int(extra.get("rounds", 100), "rounds"),
            "local_epochs": _positive_int(extra.get("local_epochs", 1), "local_epochs"), "mu": mu,
            "shared_layers": shared_layers, "local_layers": local_layers, "dp_noise_multiplier": noise,
            "dp_clip_norm": float(clip) if noise else None, "dp_delta": delta if noise else None,
            "secure_aggregation": secure}


def shared_indices(model, shared_layers, local_layers=None):
    """Weight positions averaged by the server; None shares every weight."""
    if shared_layers is None:
        return list(range(len(model.weights)))
    position = {id(weight): index for index, weight in enumerate(model.weights)}
    names = [layer.name for layer in model.layers if layer.weights]
    unknown = set(shared_layers) - set(names)
    if unknown:
        raise ValueError(f"shared_layers not in the model: {sorted(unknown)}; model layers are {names}")
    if local_layers is not None and set(local_layers) != set(names) - set(shared_layers):
        raise ValueError("shared_layers and local_layers must partition the model's weighted layers")
    shared = sorted(position[id(weight)] for layer in model.layers if layer.name in shared_layers
                    for weight in layer.weights)
    if len(shared) == len(model.weights):
        raise ValueError("fedper must keep at least one layer local")
    return shared


def secure_aggregate(updates, *, expected_counts, schema_hash, reference, server_round, seed, scale_bits=24):
    """Pairwise-masked fixed-point FedAvg (simulated secure aggregation).

    Each client encodes num_examples*weights as 64-bit fixed point and adds one
    mask per other client (+ for the lower id, - for the higher). The server only
    sums masked vectors; the masks cancel exactly modulo 2**64, so the result is
    the row-weighted mean up to fixed-point rounding. Masks come from a seeded
    generator rather than key agreement, and there is no dropout recovery.
    """
    ids = sorted(expected_counts)
    if len(updates) != len(ids) or sorted(update.client_id for update in updates) != ids:
        raise ValueError("A round requires exactly one update per expected client")
    scale = 2.0 ** scale_bits
    bound = 2.0 ** 62 / len(ids)
    total = None
    for update in updates:
        if update.num_examples != expected_counts[update.client_id]:
            raise ValueError(f"Client {update.client_id}: reported count differs from verified fitting rows")
        if update.schema_hash != schema_hash:
            raise ValueError("Client feature schema does not match the broadcast contract")
        _check_parameters(update.parameters, reference)
        vector = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in update.parameters])
        vector *= update.num_examples * scale
        if np.abs(vector).max() >= bound:
            raise OverflowError("Update too large for 64-bit fixed-point aggregation")
        masked = np.rint(vector).astype(np.int64).view(np.uint64)
        for other in ids:
            if other != update.client_id:
                low, high = sorted((update.client_id, other))
                mask = np.random.default_rng([seed, server_round, low, high]).bit_generator.random_raw(vector.size)
                masked = masked + mask if update.client_id == low else masked - mask
        total = masked if total is None else total + masked
    mean = total.view(np.int64).astype(np.float64) / scale / sum(expected_counts.values())
    output, start = [], 0
    for ref in reference:
        output.append(mean[start:start + ref.size].reshape(ref.shape).astype(ref.dtype))
        start += ref.size
    return output


def run_strategy_preflight():
    """Gates for the F2-F4 additions, on small fixtures before any real training.

    Secure aggregation must equal the exact row-weighted mean and Flower's FedAvg.
    FedProx with mu=0 must equal plain training. DP-SGD without noise and with an
    unreachable clip norm must equal plain training, and a strong proximal pull
    must keep weights closer to the anchor than plain training does.
    """
    import tensorflow as tf

    from windturbine.config import Config

    rng = np.random.default_rng(1207)
    counts = {1: 3, 2: 5, 3: 8}
    reference = [np.zeros((4, 3), dtype=np.float32), np.zeros(3, dtype=np.float32)]
    updates = [ClientUpdate(aid, n, [rng.normal(size=r.shape).astype(np.float32) for r in reference], "gate")
               for aid, n in counts.items()]
    flower = aggregate_updates(updates, expected_counts=counts, schema_hash="gate", reference=reference,
                               server_round=1)
    secure = secure_aggregate(updates, expected_counts=counts, schema_hash="gate", reference=reference,
                              server_round=1, seed=7)
    exact = [sum(u.num_examples * u.parameters[k].astype(np.float64) for u in updates) / sum(counts.values())
             for k in range(len(reference))]
    secure_error = max(float(np.abs(s - e).max()) for s, e in zip(secure, exact, strict=True))
    flower_error = max(float(np.abs(s.astype(np.float64) - f).max()) for s, f in zip(secure, flower, strict=True))
    if secure_error > 1e-6 or flower_error > 1e-6:
        raise AssertionError("Secure aggregation differs from the row-weighted mean")

    cfg = Config(hidden_layers=(3, 2, 3), learning_rate=0.01, batch_size=8, use_lr_plateau=False)
    x = rng.normal(size=(20, 4)).astype(np.float32)

    def trained(**kwargs):
        tf.keras.backend.clear_session()
        model = build_autoencoder(4, cfg, seed=3)
        # Plain SGD: Adam's division by sqrt(v) would amplify float rounding in tiny gradients.
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05), loss="mse")
        start = model.get_weights()
        anchor = {index: weight for index, weight in enumerate(start)}
        train_epochs(model, x, epochs=2, batch_size=8, seed=5, anchor=anchor, **kwargs)
        weights = model.get_weights()
        del model
        return start, weights

    start, plain = trained()
    _, prox_zero = trained(mu=0.0)
    _, dp_open = trained(dp_clip_norm=1e9, dp_noise_multiplier=0.0)
    _, prox_strong = trained(mu=10.0)  # lr*mu = 0.5 < 1: pulls without overshooting
    tf.keras.backend.clear_session()
    for one, two in zip(plain, prox_zero, strict=True):
        np.testing.assert_array_equal(one, two)
    dp_error = max(float(np.abs(one - two).max()) for one, two in zip(plain, dp_open, strict=True))
    if dp_error > 1e-5:
        raise AssertionError("DP-SGD without noise or clipping differs from plain training")

    def distance(weights):
        return math.sqrt(sum(float(np.square(w - s).sum()) for w, s in zip(weights, start, strict=True)))

    if not distance(prox_strong) < distance(plain):
        raise AssertionError("The proximal term did not pull weights toward the anchor")
    return {"passed": True, "secure_aggregation_max_abs_error": secure_error,
            "secure_vs_flower_max_abs_error": flower_error, "fedprox_mu0_equals_plain": True,
            "dp_open_clip_max_abs_error": dp_error,
            "proximal_distance": {"plain": distance(plain), "mu_10": distance(prox_strong)}}


def privacy_report(counts, *, batch_size, rounds, local_epochs, noise_multiplier, delta):
    """Record-level (epsilon, delta) for each turbine from the RDP accountant.

    The accountant assumes Poisson sampling with q = batch_size / rows. Training
    uses shuffled fixed-size batches, so the reported epsilon is the common
    approximation for that sampler, not a proven bound.
    """
    import dp_accounting
    from dp_accounting import rdp

    rows = []
    for aid, n in counts.items():
        steps = rounds * local_epochs * math.ceil(n / batch_size)
        rate = min(1.0, batch_size / n)
        accountant = rdp.RdpAccountant()
        accountant.compose(dp_accounting.SelfComposedDpEvent(
            dp_accounting.PoissonSampledDpEvent(rate, dp_accounting.GaussianDpEvent(noise_multiplier)), steps))
        rows.append({"asset_id": aid, "n_train": n, "sampling_rate": rate, "steps": steps,
                     "epsilon": float(accountant.get_epsilon(delta))})
    return rows


def _personal_validation(model, cache, val_counts, weights, epoch, history):
    """Row-weighted validation MSE, each client scoring with its own model."""
    losses = []
    for aid, count in val_counts.items():
        model.set_weights(weights[aid])
        value = mean_squared_error(model, _matrix(cache, aid, "val"))
        losses.append(value * count)
        history.append({"epoch": epoch, "asset_id": aid, "phase": "personal_validation",
                        "loss": value, "num_examples": count})
    return math.fsum(losses) / sum(val_counts.values())


def run_federated(dataset_root, output_root, cfg, *, experiment, source_identity=None):
    """Run FedAvg, FedPer or FedProx, optionally with DP-SGD and secure aggregation.

    Every client trains every round from the broadcast shared weights (plus its
    own local layers for FedPer) with a fresh Adam optimizer. Only shared weights
    are aggregated, by retained fitting rows. The fixed budget runs to completion;
    the round with the lowest row-weighted validation MSE of the clients' own
    models is selected. Prediction windows never enter optimization or selection.
    """
    import tensorflow as tf

    options = strategy_options(cfg)
    rounds, local_epochs = options["rounds"], options["local_epochs"]
    private = options["dp_noise_multiplier"] > 0
    if private:
        import dp_accounting  # noqa: F401  fail before training if accounting is unavailable
    evaluate_care({}, {}, beta=cfg.beta, criticality_threshold=cfg.criticality_threshold,
                  care_weights=cfg.care_weights, verbose=False)
    root, output = Path(dataset_root).resolve(), Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite completed or partial run: {output}")
    tf.keras.utils.set_random_seed(cfg.seed)
    tf.config.experimental.enable_op_determinism()
    preflight = {"fedavg": run_preflight(), "strategy": run_strategy_preflight()}
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    branch = options["strategy"] + ("_private" if private or options["secure_aggregation"] else "")
    with tempfile.TemporaryDirectory(prefix=".federated-build-", dir=output.parent) as temporary:
        stage = Path(temporary)
        dataset, frozen, cache, counts, val_counts, events = _prepare_dataset(root, stage, cfg)
        n_features = len(dataset["schema"])
        publish = stage / "publish"
        publish.mkdir()
        tf.keras.backend.clear_session()
        model = build_autoencoder(n_features, cfg)
        initial = model.get_weights()
        contract = parameter_contract(model, dataset["schema"])
        shared = shared_indices(model, options["shared_layers"], options["local_layers"])
        del model
        save_checkpoint(publish / "initial_weights.npz", initial)
        _write_json(publish / "model_contract.json", {**contract, "shared_parameter_indices": shared})
        branch_dir = publish / branch
        branch_dir.mkdir()
        reference = [initial[index] for index in shared]
        personal = {aid: [w.copy() for w in initial] for aid in counts}
        history, round_rows = [], []
        tf.keras.backend.clear_session()
        model = build_autoencoder(n_features, cfg)
        initial_validation_mse = _personal_validation(model, cache, val_counts, personal, 0, history)
        best_value, best_round = initial_validation_mse, 0
        best = {aid: [w.copy() for w in weights] for aid, weights in personal.items()}
        round_rows.append({"epoch": 0, "validation_mse": initial_validation_mse, "participating_clients": 0,
                           "fitting_rows": sum(counts.values()), "row_exposures": 0,
                           "shared_parameters_sha256": _parameter_hash(reference),
                           "secure_aggregation_max_abs_error": None})
        del model
        rounds_started, round_seconds = time.perf_counter(), []
        for server_round in range(1, rounds + 1):
            updates, trained = [], {}
            broadcast = [personal[next(iter(counts))][index] for index in shared]
            broadcast_hash = _parameter_hash(broadcast)
            for aid, count in counts.items():
                tf.keras.backend.clear_session()
                seed = _seed(cfg.seed, server_round, aid)
                model = build_autoencoder(n_features, cfg, seed=seed)
                model.set_weights(personal[aid])
                if parameter_contract(model, dataset["schema"]) != contract:
                    raise ValueError("Local model variable contract differs from the shared model")
                start = model.get_weights()
                if (_parameter_hash([start[index] for index in shared]) != broadcast_hash
                        or int(model.optimizer.iterations.numpy())):
                    raise AssertionError("Client did not start from identical broadcast with fresh optimizer")
                loss = train_epochs(model, _matrix(cache, aid, "train"), epochs=local_epochs,
                                    batch_size=cfg.batch_size, seed=seed, mu=options["mu"],
                                    anchor={index: start[index] for index in shared},
                                    dp_clip_norm=options["dp_clip_norm"],
                                    dp_noise_multiplier=options["dp_noise_multiplier"])
                local_val = mean_squared_error(model, _matrix(cache, aid, "val"))
                trained[aid] = model.get_weights()
                updates.append(ClientUpdate(aid, count, [trained[aid][index] for index in shared],
                                            dataset["schema_hash"]))
                history.append({"epoch": server_round, "asset_id": aid, "phase": "local_fit",
                                "loss": loss, "local_val_loss": local_val, "num_examples": count,
                                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                                "seed": seed, "broadcast_sha256": broadcast_hash})
                del model
            aggregated = aggregate_updates(updates, expected_counts=counts, schema_hash=dataset["schema_hash"],
                                           reference=reference, server_round=server_round)
            secure_error = None
            if options["secure_aggregation"]:
                # Flower's plain aggregate is the test oracle; the masked sum is what the server uses.
                masked = secure_aggregate(updates, expected_counts=counts, schema_hash=dataset["schema_hash"],
                                          reference=reference, server_round=server_round,
                                          seed=_seed(cfg.seed, server_round, 0, 2))
                for one, two in zip(masked, aggregated, strict=True):
                    np.testing.assert_allclose(one, two, rtol=1e-5, atol=1e-6)
                secure_error = max(float(np.abs(one.astype(np.float64) - two).max())
                                   for one, two in zip(masked, aggregated, strict=True))
                aggregated = masked
            for aid in counts:
                personal[aid] = trained[aid]
                for index, value in zip(shared, aggregated, strict=True):
                    personal[aid][index] = value
            tf.keras.backend.clear_session()
            model = build_autoencoder(n_features, cfg)
            value = _personal_validation(model, cache, val_counts, personal, server_round, history)
            round_rows.append({"epoch": server_round, "validation_mse": value, "participating_clients": len(counts),
                               "fitting_rows": sum(counts.values()),
                               "row_exposures": sum(counts.values()) * local_epochs,
                               "shared_parameters_sha256": _parameter_hash(aggregated),
                               "secure_aggregation_max_abs_error": secure_error})
            if value < best_value:
                best_value, best_round = value, server_round
                best = {aid: [w.copy() for w in weights] for aid, weights in personal.items()}
            round_seconds.append(time.perf_counter() - rounds_started - sum(round_seconds))
            print(f"{branch} round {server_round}/{rounds}: validation MSE {value:.6g} ({round_seconds[-1]:.1f}s)")
            if server_round <= 3:
                _check_time_budget(cfg, rounds, round_seconds, False)
            del model, updates, trained
        for name, weights in (("selected", best), ("final", personal)):
            (branch_dir / name).mkdir()
            for aid in counts:
                save_checkpoint(branch_dir / name / f"client_{aid}.npz", weights[aid])
        pd.DataFrame(history).to_csv(branch_dir / "history.csv", index=False)
        pd.DataFrame(round_rows).to_csv(branch_dir / "rounds.csv", index=False)
        info = {"selected_epoch": best_round, "selected_validation_mse": best_value, "epochs_run": rounds,
                "checkpoint": f"{branch}/selected", "history": f"{branch}/history.csv",
                "rounds": f"{branch}/rounds.csv", "initial_validation_mse": initial_validation_mse,
                "optimizer_steps": sum(row.get("optimizer_steps", 0) for row in history)}
        privacy = None
        if private:
            per_client = privacy_report(counts, batch_size=cfg.batch_size, rounds=rounds, local_epochs=local_epochs,
                                        noise_multiplier=options["dp_noise_multiplier"], delta=options["dp_delta"])
            pd.DataFrame(per_client).to_csv(branch_dir / "privacy.csv", index=False)
            privacy = {"unit": "one 10-minute sensor record of one turbine (record-level DP-SGD)",
                       "mechanism": "per-example clipping and Gaussian noise on every local step",
                       "accountant": f"dp_accounting {metadata.version('dp-accounting')} RDP",
                       "sampling_assumption": "Poisson sampling at q=batch_size/rows; training uses shuffled "
                                              "fixed-size batches, so epsilon is the standard approximation",
                       "noise_multiplier": options["dp_noise_multiplier"], "clip_norm": options["dp_clip_norm"],
                       "delta": options["dp_delta"], "epsilon_max": max(row["epsilon"] for row in per_client),
                       "epsilon_by_client": {row["asset_id"]: row["epsilon"] for row in per_client},
                       "not_covered": ["server checkpoint selection sees each turbine's validation MSE",
                                       "local thresholds and scores use non-private validation rows"]}
            _write_json(branch_dir / "privacy.json", privacy)
        # Reload the frozen selection before any prediction score is computed.
        selected = {aid: load_checkpoint(branch_dir / "selected" / f"client_{aid}.npz") for aid in counts}
        for weights in selected.values():
            _check_parameters(weights, initial)
        tf.keras.backend.clear_session()
        model = build_autoencoder(n_features, cfg)
        readback = _personal_validation(model, cache, val_counts, selected, best_round, [])
        np.testing.assert_allclose(readback, best_value, rtol=0, atol=1e-12)
        info["checkpoint_readback_passed"] = True
        del model
        info.update(_score_selected(root, branch_dir, cfg, selected, dataset["schema"], events))
        _verify_frozen_inputs(root, frozen)
        copyfile(branch_dir / "results.json", publish / "results.json")
        copyfile(branch_dir / "per_event.csv", publish / "per_event.csv")
        manifest = {"experiment": experiment, "run_id": output.name, "config_hash": config_hash(cfg),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "algorithm": "Flower server.strategy.FedAvg.aggregate_fit over the shared parameters"
                                 + ("; pairwise-masked secure sum used as the global update"
                                    if options["secure_aggregation"] else ""),
                    "strategy": options, "config": config_dict(cfg), "source_identity": source_identity,
                    "dataset_manifest_sha256": frozen["dataset_manifest.json"],
                    "dataset_source_identity": dataset.get("source_identity"),
                    "dataset_override_reason": cfg.extra.get("dataset_override_reason"),
                    "dataset_hashes_rechecked": True, "clients": list(counts), "n_events": len(events),
                    "fit_counts": counts, "val_counts": val_counts, "schema": dataset["schema"],
                    "seed": cfg.seed, "model_seed": cfg.seed,
                    "partition_seed": dataset["preprocessing_config"]["seed"], "tensorflow_op_determinism": True,
                    "schema_hash": dataset["schema_hash"], "preflight": preflight, "branches": {branch: info},
                    "shared_parameter_indices": shared, "privacy": privacy,
                    "initial_weights_sha256": file_hash(publish / "initial_weights.npz"),
                    "local_optimizer_policy": "fresh Adam for every client and round; fixed learning rate",
                    "seed_policy": "NumPy SeedSequence([base_seed,round,asset_id,branch]); branch 2 secure masks",
                    "selection_policy": "initial checkpoint plus lowest retained-validation-row-weighted MSE of "
                                        "each client's own model; earliest tie; no prediction selection",
                    "budget": {"rounds": rounds, "local_epochs": local_epochs,
                               "row_exposures": sum(counts.values()) * rounds * local_epochs},
                    "inactive_policy": dataset["inactive_policy"],
                    "simulation": "one process; sequential clients; no networking or isolation",
                    "packages": package_versions(), "flower_version": metadata.version("flwr"),
                    "runtime_seconds": time.perf_counter() - started,
                    "implementation_sha256": file_hash(__file__)}
        manifest["artifacts"] = {p.relative_to(publish).as_posix(): file_hash(p)
                                 for p in sorted(publish.rglob("*")) if p.is_file()}
        _write_json(publish / "manifest.json", manifest)
        (publish / "COMPLETE").write_text(file_hash(publish / "manifest.json"), encoding="ascii")
        publish.rename(output)
    return _plain(manifest)


def summarize_runs(run_dirs):
    """One verified row per branch of each completed run, for CARE-vs-mu/epsilon tables.

    Accepts F1 runs too, so FedAvg and its centralized comparator can sit in the
    same table. Every artifact hash is checked; runs on different datasets fail.
    """
    rows = []
    for run in map(Path, run_dirs):
        manifest_path, marker = run / "manifest.json", run / "COMPLETE"
        if not marker.is_file() or marker.read_text(encoding="ascii") != file_hash(manifest_path):
            raise ValueError(f"Incomplete or modified run: {run}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, digest in manifest["artifacts"].items():
            if file_hash(run / relative) != digest:
                raise ValueError(f"Artifact hash mismatch in {run}: {relative}")
        strategy = manifest.get("strategy") or {}
        privacy = manifest.get("privacy") or {}
        for name, branch in manifest["branches"].items():
            rows.append({"run": run.name, "experiment": manifest["experiment"], "branch": name,
                         "mu": strategy.get("mu", 0.0),
                         "noise_multiplier": strategy.get("dp_noise_multiplier", 0.0),
                         "epsilon": privacy.get("epsilon_max", math.inf),
                         "secure_aggregation": strategy.get("secure_aggregation", False),
                         "seed": manifest["seed"], "rounds": manifest["budget"]["rounds"],
                         "local_epochs": manifest["budget"]["local_epochs"],
                         "selected_epoch": branch["selected_epoch"],
                         "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
                         **{key: branch["results"][key] for key in METRICS}})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No completed runs were given")
    if frame.dataset_manifest_sha256.nunique() > 1:
        raise ValueError("Runs use different datasets and cannot share one curve")
    return frame
