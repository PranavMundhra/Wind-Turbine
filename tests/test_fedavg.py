"""Real Flower aggregation and small, real TensorFlow end-to-end F1 checks."""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_clients import make_sources
from windturbine.config import Config
from windturbine.data import build_client_dataset, load_client_dataset
from windturbine.federated import ClientUpdate, aggregate_updates, load_checkpoint, run_fedavg, run_preflight
from windturbine.federated.fedavg import save_checkpoint
from windturbine.models import build_autoencoder, mean_squared_error
from windturbine.runs import file_hash
from windturbine.scoring import reconstruction_error


def test_preflight_uses_flower_multiple_rounds_and_unequal_counts():
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    result = run_preflight()
    assert result["passed"]
    assert result["sgd_equivalence"]["rounds"] == 3
    assert len(set(result["sgd_equivalence"]["client_rows"])) == 3
    assert result["sgd_equivalence"]["max_abs_error"] < 1e-12
    assert result["care_parity"]["passed"]
    assert result["sgd_equivalence"]["parameter_arrays"] == 2
    assert result["tensorflow_compatibility"]["checkpoint_roundtrip"]


def test_flower_aggregation_weights_actual_rows():
    pytest.importorskip("flwr")
    reference = [np.zeros((2,), dtype=np.float64)]
    updates = [ClientUpdate(1, 3, [np.array([1.0, 2.0])], "s"),
               ClientUpdate(2, 7, [np.array([11.0, 22.0])], "s")]
    result = aggregate_updates(updates, expected_counts={1: 3, 2: 7}, schema_hash="s",
                               reference=reference, server_round=1)
    np.testing.assert_allclose(result[0], [8, 16], rtol=0, atol=1e-15)
    assert not np.array_equal(result[0], [6, 12])  # client-mean averaging is wrong


@pytest.mark.parametrize("bad", ["count", "schema", "nonfinite", "shape", "dtype", "missing", "duplicate"])
def test_round_rejects_invalid_client_updates(bad):
    pytest.importorskip("flwr")
    array = np.array([1.0], dtype=np.float64)
    updates = [ClientUpdate(1, 3, [array.copy()], "s"), ClientUpdate(2, 7, [array.copy()], "s")]
    if bad == "count":
        updates[0] = replace(updates[0], num_examples=4)
    elif bad == "schema":
        updates[0] = replace(updates[0], schema_hash="wrong")
    elif bad == "nonfinite":
        updates[0].parameters[0][0] = np.nan
    elif bad == "shape":
        updates[0] = replace(updates[0], parameters=[np.ones(2)])
    elif bad == "dtype":
        updates[0] = replace(updates[0], parameters=[array.astype(np.float32)])
    elif bad == "missing":
        updates.pop()
    elif bad == "duplicate":
        updates[1] = updates[0]
    with pytest.raises(ValueError):
        aggregate_updates(updates, expected_counts={1: 3, 2: 7}, schema_hash="s",
                          reference=[array], server_round=1)


def test_checkpoint_order_and_finiteness(tmp_path):
    path = tmp_path / "weights.npz"
    weights = [np.arange(6, dtype=np.float32).reshape(2, 3), np.array([2.0], dtype=np.float32)]
    save_checkpoint(path, weights)
    for one, two in zip(weights, load_checkpoint(path), strict=True):
        np.testing.assert_array_equal(one, two)
    np.savez(path, parameter_0001=np.array([1.0]))
    with pytest.raises(ValueError, match="order"):
        load_checkpoint(path)
    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(path, [np.array([np.nan])])


def _make_export(tmp_path):
    farm, base = make_sources(tmp_path)
    # Unequal fitting/validation sizes distinguish sample weighting from client
    # averaging. Both turbines still have nonempty blocked partitions.
    for eid in (3, 4):
        path = farm / "datasets" / f"comma_{eid}.csv"
        frame = pd.read_csv(path)
        frame.loc[35:99, "status_type_id"] = 4
        frame.to_csv(path, index=False)
    dataset = tmp_path / "clients"
    build_client_dataset(farm, dataset, base, expected_clients=2, expected_events=4,
                         source_identity={"purpose": "synthetic test"})
    cfg = replace(base, hidden_layers=(6, 3, 6), learning_rate=0.001, batch_size=16,
                  use_lr_plateau=False, criticality_threshold=2, extra={
                      "expected_clients": 2, "expected_events": 4, "n_clients": 2,
                      "rounds": 2, "local_epochs": 2,
                      "expected_dataset_manifest_sha256": file_hash(dataset / "dataset_manifest.json"),
                      "dataset_override_reason": "Synthetic two-client integration test, not Farm C evidence",
                  })
    return dataset, cfg


def test_real_tensorflow_flower_end_to_end_checkpoint_calibration_and_metric_parity(tmp_path):
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    dataset, cfg = _make_export(tmp_path)
    cfg = replace(cfg, seed=43)  # immutable partition seed stays 42
    output = tmp_path / "F1"
    result = run_fedavg(dataset, output, cfg, source_identity={"purpose": "test"})
    assert result["preflight"]["passed"] and result["dataset_hashes_rechecked"]
    assert result["fit_counts"]["1"] != result["fit_counts"]["2"]
    assert result["budget"]["union_epochs"] == 4
    assert result["run_id"] == "F1"
    assert result["model_seed"] == 43 and result["partition_seed"] == 42
    assert (output / "COMPLETE").read_text(encoding="ascii") == file_hash(output / "manifest.json")
    assert result == json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert (output / "results.json").read_bytes() == (output / "fedavg/results.json").read_bytes()
    for relative, digest in result["artifacts"].items():
        assert file_hash(output / relative) == digest
    initial = load_checkpoint(output / "initial_weights.npz")
    rows = result["fit_counts"]
    expected_steps = sum(int(np.ceil(count / cfg.batch_size)) for count in rows.values()) * 4
    for name, branch in result["branches"].items():
        assert branch["metric_parity"] and branch["checkpoint_readback_passed"]
        assert branch["n_events"] == 4
        assert branch["optimizer_steps"] == expected_steps
        assert branch["initial_validation_mse"] == result["branches"]["fedavg"]["initial_validation_mse"]
        curve = pd.read_csv(output / branch["rounds"])
        assert curve.iloc[0].epoch == 0
        assert len(curve) == cfg.extra["rounds"] + 1
        assert branch["selected_validation_mse"] == pytest.approx(curve.validation_mse.min())
        assert branch["selected_epoch"] == int(curve.loc[curve.validation_mse.idxmin(), "epoch"])
        final = load_checkpoint(output / name / "final_weights.npz")
        assert any(not np.array_equal(a, b) for a, b in zip(initial, final, strict=True))
        weights = load_checkpoint(output / branch["checkpoint"])
        model = build_autoencoder(len(result["schema"]), cfg)
        model.set_weights(weights)
        threshold_table = pd.read_csv(output / name / "thresholds.csv").set_index("asset_id")
        val_terms = []
        for aid in (1, 2):
            client = load_client_dataset(dataset, aid)
            val = client["val"]["X"]
            val_terms.append((mean_squared_error(model, val), len(val)))
            expected_scores = reconstruction_error(model, val)
            assert threshold_table.loc[aid, "thr"] == pytest.approx(np.quantile(expected_scores, cfg.score_quantile))
            with np.load(output / name / "scores" / f"client_{aid}_validation.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["scores"], expected_scores)
        weighted_mse = sum(loss * n for loss, n in val_terms) / sum(n for _, n in val_terms)
        assert branch["selected_validation_mse"] == pytest.approx(weighted_mse)
        history = pd.read_csv(output / branch["history"])
        if name == "fedavg":
            local = history[history.phase == "local_fit"]
            assert len(local) == 4
            assert local.groupby("epoch").broadcast_sha256.nunique().eq(1).all()
            assert local.groupby("asset_id").optimizer_steps.nunique().eq(1).all()
    with pytest.raises(FileExistsError):
        run_fedavg(dataset, output, cfg)


@pytest.mark.parametrize("change, message", [
    ({"fraction_fit": 0.5}, "fraction_fit"), ({"strategy": "fedprox"}, "strategy"),
    ({"scaler": "global"}, "scaler"), ({"rounds": 0}, "rounds"),
    ({"global_patience": 3}, "fixed matched"),
])
def test_unsupported_experiment_options_fail_before_any_run(tmp_path, change, message):
    pytest.importorskip("tensorflow")
    cfg = replace(Config(), use_lr_plateau=False, extra=change)
    with pytest.raises(ValueError, match=message):
        run_fedavg(tmp_path / "missing", tmp_path / "output", cfg)
    assert not (tmp_path / "output").exists()


def test_source_manifest_pin_is_required_and_wrong_pin_fails_atomically(tmp_path):
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    dataset, cfg = _make_export(tmp_path)
    for extra, message in (({**cfg.extra, "expected_dataset_manifest_sha256": "bad"}, "does not match"),
                           ({**cfg.extra, "expected_dataset_manifest_sha256": None,
                             "dataset_override_reason": None}, "Pin expected")):
        with pytest.raises(ValueError, match=message):
            run_fedavg(dataset, tmp_path / "output", replace(cfg, extra=extra))
        assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("pinned", [True, False])
def test_alternative_dataset_requires_explanation_even_when_pinned(tmp_path, pinned):
    from windturbine.federated.fedavg import _prepare_dataset

    dataset, cfg = _make_export(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    extra = {**cfg.extra, "dataset_override_reason": None}
    if not pinned:
        extra["expected_dataset_manifest_sha256"] = None
    with pytest.raises(ValueError, match="dataset_override_reason"):
        _prepare_dataset(dataset, stage, replace(cfg, extra=extra))
    assert not (stage / "matrix_cache").exists()
    extra["dataset_override_reason"] = "Explicit alternate synthetic export for a small integration test"
    manifest, _, _, counts, _, _ = _prepare_dataset(dataset, stage, replace(cfg, extra=extra))
    assert manifest["n_events"] == 4 and len(counts) == 2
