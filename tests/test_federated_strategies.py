"""F2-F4 engine: secure aggregation, DP-SGD clipping, FedPer/FedProx/private end-to-end runs."""
import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from test_fedavg import _make_export
from windturbine.config import Config
from windturbine.federated import (
    ClientUpdate,
    aggregate_updates,
    load_checkpoint,
    run_fedavg,
    run_federated,
    run_strategy_preflight,
    secure_aggregate,
    strategy_options,
    summarize_runs,
)
from windturbine.runs import file_hash


def _updates(counts, reference, seed=3):
    rng = np.random.default_rng(seed)
    return [ClientUpdate(aid, n, [rng.normal(size=r.shape).astype(np.float32) for r in reference], "s")
            for aid, n in counts.items()]


def test_secure_aggregation_masks_cancel_to_the_row_weighted_mean():
    pytest.importorskip("flwr")
    counts = {4: 2, 9: 11, 12: 5, 20: 7}
    reference = [np.zeros((5, 2), dtype=np.float32), np.zeros(2, dtype=np.float32)]
    updates = _updates(counts, reference)
    secure = secure_aggregate(updates, expected_counts=counts, schema_hash="s", reference=reference,
                              server_round=3, seed=11)
    exact = [sum(u.num_examples * u.parameters[k].astype(np.float64) for u in updates) / sum(counts.values())
             for k in range(2)]
    flower = aggregate_updates(updates, expected_counts=counts, schema_hash="s", reference=reference,
                               server_round=3)
    for one, two, three in zip(secure, exact, flower, strict=True):
        assert one.dtype == np.float32
        np.testing.assert_allclose(one, two, rtol=0, atol=1e-7)
        np.testing.assert_allclose(one, three, rtol=1e-6, atol=1e-7)
    # A different mask seed changes every masked vector but not the sum.
    again = secure_aggregate(updates, expected_counts=counts, schema_hash="s", reference=reference,
                             server_round=3, seed=12)
    for one, two in zip(secure, again, strict=True):
        np.testing.assert_array_equal(one, two)


@pytest.mark.parametrize("bad", ["count", "schema", "missing", "nonfinite"])
def test_secure_aggregation_rejects_invalid_rounds(bad):
    counts = {1: 3, 2: 7}
    reference = [np.zeros(2, dtype=np.float32)]
    updates = _updates(counts, reference)
    if bad == "count":
        updates[0] = replace(updates[0], num_examples=4)
    elif bad == "schema":
        updates[0] = replace(updates[0], schema_hash="other")
    elif bad == "missing":
        updates.pop()
    else:
        updates[0].parameters[0][0] = np.inf
    with pytest.raises(ValueError):
        secure_aggregate(updates, expected_counts=counts, schema_hash="s", reference=reference,
                         server_round=1, seed=1)


def test_strategy_preflight_gates_pass():
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    result = run_strategy_preflight()
    assert result["passed"] and result["fedprox_mu0_equals_plain"]
    assert result["secure_aggregation_max_abs_error"] < 1e-6
    assert result["dp_open_clip_max_abs_error"] < 1e-5
    assert result["proximal_distance"]["mu_10"] < result["proximal_distance"]["plain"]


def test_dp_sgd_clipping_matches_per_example_jacobian():
    tf = pytest.importorskip("tensorflow")
    from windturbine.models import build_autoencoder
    from windturbine.models.autoencoder import _dense_stack

    cfg = Config(hidden_layers=(6, 3, 6), use_lr_plateau=False)
    model = build_autoencoder(5, cfg, seed=2)
    batch = tf.constant(np.random.default_rng(0).normal(size=(9, 5)).astype(np.float32))
    clip = 0.3
    with tf.GradientTape() as tape:
        losses = tf.reduce_mean(tf.square(batch - model(batch, training=True)), axis=1)
    jacobians = [j.numpy() for j in tape.jacobian(losses, model.trainable_variables)]
    norms = np.sqrt(sum(np.square(j.reshape(9, -1)).sum(axis=1) for j in jacobians))
    factors = np.minimum(1.0, clip / norms)
    assert factors.min() < 1  # the fixture actually clips
    expected = [np.tensordot(factors, j, axes=1) / 9 for j in jacobians]
    # One SGD step with lr=1 exposes the clipped gradient as the weight change.
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1.0), loss="mse")
    before = model.get_weights()
    from windturbine.models import train_epochs

    train_epochs(model, batch.numpy(), epochs=1, batch_size=9, seed=0, dp_clip_norm=clip, dp_noise_multiplier=0.0)
    assert len(_dense_stack(model)) == 4
    for start, end, gradient in zip(before, model.get_weights(), expected, strict=True):
        # Shuffling permutes rows, which does not change a sum over the batch.
        np.testing.assert_allclose(start - end, gradient, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("change, message", [
    ({"strategy": "scaffold"}, "strategy"), ({"strategy": "fedprox"}, "explicit mu"),
    ({"strategy": "fedavg", "mu": 0.1}, "only by fedprox"), ({"strategy": "fedper"}, "shared_layers"),
    ({"strategy": "fedavg", "shared_layers": ["encode1"]}, "only by fedper"),
    ({"strategy": "fedavg", "dp_noise_multiplier": 1.0}, "clip"),
    ({"strategy": "fedavg", "dp_noise_multiplier": 1.0, "dp_clip_norm": 1.0, "dp_delta": "1e-6"}, "dp_delta"),
    ({"strategy": "fedavg", "fraction_fit": 0.5}, "fraction_fit"),
    ({"strategy": "fedavg", "secure_aggregation": "yes"}, "secure_aggregation"),
])
def test_invalid_strategy_settings_fail_before_any_run(tmp_path, change, message):
    cfg = replace(Config(), use_lr_plateau=False, extra=change)
    with pytest.raises((ValueError, TypeError), match=message):
        strategy_options(cfg)
    pytest.importorskip("tensorflow")
    with pytest.raises((ValueError, TypeError), match=message):
        run_federated(tmp_path / "missing", tmp_path / "out", cfg, experiment="F2")
    assert not (tmp_path / "out").exists()


def _verify_published(output, result):
    assert (output / "COMPLETE").read_text(encoding="ascii") == file_hash(output / "manifest.json")
    assert result == json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for relative, digest in result["artifacts"].items():
        assert file_hash(output / relative) == digest
    (branch,) = result["branches"].values()
    assert branch["metric_parity"] and branch["checkpoint_readback_passed"] and branch["n_events"] == 4
    assert result["preflight"]["fedavg"]["passed"] and result["preflight"]["strategy"]["passed"]
    return branch


def _selected(output, branch, clients):
    return {aid: load_checkpoint(output / branch["checkpoint"] / f"client_{aid}.npz") for aid in clients}


def test_fedper_shares_encoder_and_keeps_decoders_local(tmp_path):
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    dataset, cfg = _make_export(tmp_path)
    cfg = replace(cfg, extra={**cfg.extra, "strategy": "fedper", "local_epochs": 1,
                              "shared_layers": ["hidden_00", "hidden_01"],
                              "local_layers": ["hidden_02", "output"]})
    output = tmp_path / "F2"
    result = run_federated(dataset, output, cfg, experiment="F2", source_identity={"purpose": "test"})
    branch = _verify_published(output, result)
    assert list(result["branches"]) == ["fedper"] and result["experiment"] == "F2"
    shared = result["shared_parameter_indices"]
    assert shared == [0, 1, 2, 3]
    final = {aid: load_checkpoint(output / "fedper/final" / f"client_{aid}.npz") for aid in (1, 2)}
    for index in range(len(final[1])):
        same = np.array_equal(final[1][index], final[2][index])
        assert same == (index in shared), index
    rows = pd.read_csv(output / branch["rounds"])
    assert len(rows) == 3 and rows.secure_aggregation_max_abs_error.isna().all()
    assert branch["selected_validation_mse"] == pytest.approx(rows.validation_mse.min())
    expected_steps = sum(math.ceil(n / cfg.batch_size) for n in result["fit_counts"].values()) * 2
    assert branch["optimizer_steps"] == expected_steps
    thresholds = pd.read_csv(output / "fedper/thresholds.csv")
    assert len(thresholds) == 2 and result["privacy"] is None


def test_engine_fedavg_matches_f1_and_fedprox_mu0_matches_fedavg(tmp_path):
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    dataset, cfg = _make_export(tmp_path)
    f1 = run_fedavg(dataset, tmp_path / "F1", cfg, run_centralized=False)
    fedavg = run_federated(dataset, tmp_path / "avg", replace(cfg, extra={**cfg.extra, "strategy": "fedavg"}),
                           experiment="F3")
    prox0 = run_federated(dataset, tmp_path / "prox0",
                          replace(cfg, extra={**cfg.extra, "strategy": "fedprox", "mu": 0.0}), experiment="F3")
    run_federated(dataset, tmp_path / "prox", replace(cfg, extra={**cfg.extra, "strategy": "fedprox", "mu": 10.0}),
                  experiment="F3")
    f1_weights = load_checkpoint(tmp_path / "F1" / f1["branches"]["fedavg"]["checkpoint"])
    avg_branch, prox0_branch = fedavg["branches"]["fedavg"], prox0["branches"]["fedprox"]
    assert avg_branch["selected_epoch"] == f1["branches"]["fedavg"]["selected_epoch"]
    assert avg_branch["selected_validation_mse"] == f1["branches"]["fedavg"]["selected_validation_mse"]
    for weights in _selected(tmp_path / "avg", avg_branch, (1, 2)).values():
        for one, two in zip(weights, f1_weights, strict=True):
            np.testing.assert_array_equal(one, two)
    assert prox0_branch["results"] == avg_branch["results"]
    for name in ("final/client_1.npz", "final/client_2.npz"):
        for one, two in zip(load_checkpoint(tmp_path / "avg/fedavg" / name),
                            load_checkpoint(tmp_path / "prox0/fedprox" / name), strict=True):
            np.testing.assert_array_equal(one, two)
    moved = [load_checkpoint(tmp_path / run / "final/client_1.npz") for run in ("avg/fedavg", "prox/fedprox")]
    assert any(not np.array_equal(a, b) for a, b in zip(*moved, strict=True))
    table = summarize_runs([tmp_path / "F1", tmp_path / "avg", tmp_path / "prox0", tmp_path / "prox"])
    assert list(table.mu) == [0.0, 0.0, 0.0, 10.0]
    assert table.CARE.notna().all()


def test_private_run_reports_epsilon_and_secure_sum(tmp_path):
    pytest.importorskip("flwr")
    pytest.importorskip("tensorflow")
    pytest.importorskip("dp_accounting")
    dataset, cfg = _make_export(tmp_path)
    base = {**cfg.extra, "strategy": "fedavg", "secure_aggregation": True, "dp_clip_norm": 1.0,
            "dp_delta": 1.0e-6}
    secure_only = run_federated(dataset, tmp_path / "s0", replace(cfg, extra={**base, "dp_noise_multiplier": 0.0}),
                                experiment="F4")
    noisy = run_federated(dataset, tmp_path / "s1", replace(cfg, extra={**base, "dp_noise_multiplier": 1.0}),
                          experiment="F4")
    for output, result in ((tmp_path / "s0", secure_only), (tmp_path / "s1", noisy)):
        branch = _verify_published(output, result)
        assert list(result["branches"]) == ["fedavg_private"]
        errors = pd.read_csv(output / branch["rounds"]).secure_aggregation_max_abs_error.dropna()
        assert len(errors) == 2 and errors.max() < 1e-6
    assert secure_only["privacy"] is None
    privacy = noisy["privacy"]
    assert privacy["epsilon_max"] == max(privacy["epsilon_by_client"].values()) > 0
    table = pd.read_csv(tmp_path / "s1/fedavg_private/privacy.csv")
    assert set(table.asset_id) == {1, 2} and (table.steps > 0).all()
    sweep = summarize_runs([tmp_path / "s0", tmp_path / "s1"])
    assert math.isinf(sweep.epsilon.iloc[0]) and sweep.epsilon.iloc[1] == privacy["epsilon_max"]
    with pytest.raises(ValueError, match="Incomplete"):
        (tmp_path / "s1/COMPLETE").write_text("tampered", encoding="ascii")
        summarize_runs([tmp_path / "s1"])
