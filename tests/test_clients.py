"""Small raw-source integration fixtures exercise the complete export/reload path."""
import json
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from windturbine.config import Config
from windturbine.data.clients import (
    build_client_dataset,
    canonical_fingerprints,
    load_client_dataset,
    validate_client_dataset,
)


def make_sources(tmp_path):
    farm = tmp_path / "Wind Farm C"
    datasets = farm / "datasets"
    datasets.mkdir(parents=True)
    meta = []
    for eid in range(1, 5):
        aid = (eid - 1) // 2 + 1
        rng = np.random.default_rng(eid)
        frame = pd.DataFrame({
            "time_stamp": pd.date_range("2020-01-01", periods=110, freq="10min"),
            "asset_id": aid, "train_test": ["train"] * 100 + ["prediction"] * 10,
            "status_type_id": 0, "sensor_a_avg": rng.normal(size=110),
            "sensor_b_avg": rng.normal(size=110), "wind_direction_avg": np.linspace(0, 360, 110),
            "power_1_avg": 1.0, "wind_speed_1_avg": 5.0,
        })
        # Different missing patterns must retain a common ordered feature schema.
        if aid == 1:
            frame.loc[:80, "sensor_b_avg"] = np.nan
        # status-only prediction mask must preserve a low-power normal-status row.
        frame.loc[100, "power_1_avg"] = 0
        frame.loc[101, "status_type_id"] = 4
        frame.to_csv(datasets / f"comma_{eid}.csv", index=False)
        if eid == 1:  # duplicate variant: never count/load twice
            frame.to_csv(datasets / f"{eid}.csv", sep=";", index=False)
        meta.append({"event_id": eid, "event_label": "anomaly" if eid % 2 else "normal",
                     "event_start": "2020-01-01 17:00", "event_end": "2020-01-01 18:10"})
    pd.DataFrame(meta).to_csv(farm / "event_info.csv", sep=";", index=False)
    pd.DataFrame({"sensor_name": ["sensor_a", "sensor_b", "wind_direction", "power_1", "wind_speed_1"],
                  "unit": ["m/s", "m/s", "degrees", "kW", "m/s"]}).to_csv(
                      farm / "feature_description.csv", index=False)
    cfg = replace(Config(), val_block_hours=1, val_fraction=0.25)
    return farm, cfg


def build(farm, output, cfg):
    return build_client_dataset(farm, output, cfg, expected_clients=2, expected_events=4)


def test_clean_export_common_schema_and_status_only_evaluation(tmp_path):
    farm, cfg = make_sources(tmp_path)
    datasets = farm / "datasets"
    first = pd.read_csv(datasets / "comma_1.csv")
    second = pd.read_csv(datasets / "comma_2.csv")
    other = pd.read_csv(datasets / "comma_3.csv")
    sensors = [c for c in first if c.endswith("_avg")]
    # Six-row blocks, stride four: row 18 is validation, row 0 training.
    second.loc[0, sensors] = first.loc[18, sensors].to_numpy()
    # Train/pred across physical turbines: global exclusion must catch this.
    other.loc[102, sensors] = first.loc[1, sensors].to_numpy()
    # Same turbine validation/pred collision must also be removed.
    second.loc[103, sensors] = first.loc[19, sensors].to_numpy()
    # Schema angle sanity checks must never inspect validation/prediction ranges.
    first.loc[20, "wind_direction_avg"] = 10000  # validation block
    other.loc[109, "wind_direction_avg"] = 10000  # prediction row
    first.to_csv(datasets / "comma_1.csv", index=False)
    second.to_csv(datasets / "comma_2.csv", index=False)
    other.to_csv(datasets / "comma_3.csv", index=False)
    output = tmp_path / "export"
    manifest = build(farm, output, cfg)
    one, two = [load_client_dataset(output, aid) for aid in (1, 2)]
    assert one["schema"] == two["schema"] == manifest["schema"]
    assert "wind_direction_avg__sin" in manifest["schema"]
    assert one["audit"]["train_removed_validation"] >= 1
    assert one["audit"]["train_removed_prediction"] >= 1
    assert one["audit"]["val_removed_prediction"] >= 1
    assert not any(one["audit"]["intersections"].values())
    assert one["prediction"]["eval_mask"][0]
    assert not one["prediction"]["eval_mask"][1]
    power_idx = one["schema"].index("power_1_avg")
    assert one["scaler"]["active_mask"][power_idx]
    assert one["scaler"]["scale"][power_idx] == 1
    assert one["prediction"]["X"][0, power_idx] == -1  # a deviation from constant healthy power survives
    assert not one["scaler"]["active_mask"][one["schema"].index("sensor_b_avg")]
    assert two["scaler"]["active_mask"][two["schema"].index("sensor_b_avg")]
    for client in (one, two):
        x = client["train"]["X"]
        active = np.asarray(client["scaler"]["active_mask"])
        assert x.dtype == np.float32 and np.isfinite(x).all()
        np.testing.assert_allclose(x[:, active].mean(axis=0), 0, atol=2e-6)
        assert (x[:, ~active] == 0).all()
        for a, b in (("train", "val"), ("train", "prediction"), ("val", "prediction")):
            assert not np.intersect1d(client[a]["fingerprint"], client[b]["fingerprint"]).size
    with pytest.raises(FileExistsError, match="overwrite"):
        build(farm, output, cfg)
    assert manifest == json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert validate_client_dataset(output) == manifest


def test_fingerprints_normalize_signed_zero_and_column_order():
    a = pd.DataFrame({"x": [0.0, np.nan, 1.00000001], "y": [-0.0, 2, 3]})
    b = a.copy()[["y", "x"]]
    b.loc[0, ["x", "y"]] = [-0.0, 0.0]
    b.loc[2, "x"] = 1
    np.testing.assert_array_equal(canonical_fingerprints(a, ["x", "y"]),
                                  canonical_fingerprints(b, ["x", "y"]))


def test_cross_client_fitting_validation_overlap_fails_closed(tmp_path):
    farm, cfg = make_sources(tmp_path)
    first = pd.read_csv(farm / "datasets/comma_1.csv")
    other = pd.read_csv(farm / "datasets/comma_3.csv")
    sensors = [c for c in first if c.endswith("_avg")]
    # Row 18 is calibration at client 1; row 0 is fitting at client 2.
    other.loc[0, sensors] = first.loc[18, sensors].to_numpy()
    other.to_csv(farm / "datasets/comma_3.csv", index=False)
    with pytest.raises(ValueError, match="cross-client fitting-validation overlap"):
        build(farm, tmp_path / "export", cfg)
    assert not (tmp_path / "export").exists()


def test_fail_closed_on_missing_metadata_and_partial_outputs(tmp_path):
    farm, cfg = make_sources(tmp_path)
    meta = pd.read_csv(farm / "event_info.csv", sep=";").iloc[:-1]
    meta.to_csv(farm / "event_info.csv", index=False)
    output = tmp_path / "export"
    with pytest.raises(ValueError, match="match exactly"):
        build(farm, output, cfg)
    assert not output.exists()
    output.mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        load_client_dataset(output, 1)


def test_reject_artifact_tampering_and_manifest_tampering(tmp_path):
    farm, cfg = make_sources(tmp_path)
    output = tmp_path / "export"
    build(farm, output, cfg)
    artifact = output / "clients/1/train.npz"
    original = artifact.read_bytes()
    artifact.write_bytes(original + b"tampered")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_client_dataset(output, 1)
    artifact.write_bytes(original)
    manifest = output / "dataset_manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_client_dataset(output, 1)


def test_reject_unproven_or_mutated_legacy_cache(tmp_path):
    farm, cfg = make_sources(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "care_cache_manifest.json").write_text(json.dumps({"sensor_cols": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        build_client_dataset(farm, tmp_path / "export", cfg, cache_dir=cache,
                             expected_clients=2, expected_events=4)
    assert not (tmp_path / "export").exists()
    (cache / "care_cache_manifest.json").write_text(json.dumps({"raw_sensor_dtype": "float64"}), encoding="utf-8")
    with pytest.raises(ValueError, match="omit cache_dir"):
        build_client_dataset(farm, tmp_path / "export", cfg, cache_dir=cache,
                             expected_clients=2, expected_events=4)


def _refresh_hashes(output, relative):
    manifest_path = output / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative] = sha256((output / relative).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (output / "COMPLETE").write_text(sha256(manifest_path.read_bytes()).hexdigest(), encoding="ascii")


def test_recompute_raw_fingerprints_even_when_recorded_hashes_match(tmp_path):
    farm, cfg = make_sources(tmp_path)
    output = tmp_path / "export"
    build(farm, output, cfg)
    relative = "clients/1/train.npz"
    with np.load(output / relative, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["fingerprint"][0] += np.uint64(1)
    np.savez_compressed(output / relative, **arrays)
    _refresh_hashes(output, relative)
    with pytest.raises(ValueError, match="raw fingerprints"):
        load_client_dataset(output, 1)


def test_recompute_train_only_scaler_even_when_recorded_hashes_match(tmp_path):
    farm, cfg = make_sources(tmp_path)
    output = tmp_path / "export"
    build(farm, output, cfg)
    relative = "clients/1/client.json"
    path = output / relative
    client = json.loads(path.read_text(encoding="utf-8"))
    client["scaler"]["mean"][0] += 1
    path.write_text(json.dumps(client), encoding="utf-8")
    _refresh_hashes(output, relative)
    with pytest.raises(ValueError, match="scaler does not reproduce"):
        load_client_dataset(output, 1)


def test_raw_column_reordering_and_repeated_builds_preserve_schema_and_arrays(tmp_path):
    farm, cfg = make_sources(tmp_path)
    first = tmp_path / "export_first"
    build(farm, first, cfg)
    for path in (farm / "datasets").glob("comma_*.csv"):
        frame = pd.read_csv(path)
        frame[list(reversed(frame.columns))].to_csv(path, index=False)
    second = tmp_path / "export_second"
    build(farm, second, cfg)
    for aid in (1, 2):
        one, two = load_client_dataset(first, aid), load_client_dataset(second, aid)
        assert one["schema"] == two["schema"]
        for split in ("train", "val", "prediction"):
            np.testing.assert_array_equal(one[split]["X"], two[split]["X"])
            np.testing.assert_array_equal(one[split]["fingerprint"], two[split]["fingerprint"])


@pytest.mark.parametrize("change, message", [
    ({"val_fraction": 1}, "validation"), ({"val_block_hours": 0}, "validation"),
    ({"val_split_mode": "oops"}, "split"),
])
def test_invalid_split_settings_fail(tmp_path, change, message):
    farm, cfg = make_sources(tmp_path)
    with pytest.raises(ValueError, match=message):
        build(farm, tmp_path / "export", replace(cfg, **change))


def test_default_cohort_count_is_enforced(tmp_path):
    farm, cfg = make_sources(tmp_path)
    with pytest.raises(ValueError, match="58 events"):
        build_client_dataset(farm, tmp_path / "export", cfg)
