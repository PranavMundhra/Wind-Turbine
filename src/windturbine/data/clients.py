"""Reproducible, CPU-only client partitions for local and federated experiments.

The split reproduces Arm D's retained-row blocks (six retained rows per nominal
hour), with a chronological fallback for short events. It does not impose a
time gap. Fingerprints detect rounded equal measurements, not every possible
statistical dependence. Their collision risk and repeated flat readings remain.
"""
from __future__ import annotations

import json
import platform
import re
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

PREPROCESSING_KEYS = (
    "farm", "use_avg_only", "drop_reactive", "angle_transform", "max_nan_fraction",
    "normal_status_ids", "wind_speed_cutin", "wind_speed_rated", "power_near_zero_thresh",
    "val_fraction", "val_split_mode", "val_block_hours", "seed",
)
DESCRIPTIVE = {"time_stamp", "asset_id", "id", "event_id", "train_test", "status_type_id"}
REQUIRED = {"time_stamp", "asset_id", "train_test", "status_type_id"}


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def _write_json(path, value):
    path.write_text(_json(value), encoding="utf-8")


def _hash(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _object_hash(value):
    return sha256(_json(value).encode()).hexdigest()


def preprocessing_config(cfg):
    """Fields defining the dataset; model/arm/output settings do not affect it."""
    data = {key: getattr(cfg, key) for key in PREPROCESSING_KEYS}
    data["fingerprint_decimals"] = cfg.extra.get("fingerprint_decimals", 6)
    return json.loads(_json(data))


def _csv(path, **kwargs):
    # The source ships semicolon and comma versions. Detect per file, rather
    # than assuming all plain-named exports have one delimiter.
    with Path(path).open(encoding="utf-8-sig") as stream:
        line = stream.readline()
    sep = ";" if line.count(";") > line.count(",") else ","
    return pd.read_csv(path, sep=sep, **kwargs)


def _farm(root, name):
    root = Path(root)
    choices = [root] if (root / "datasets").is_dir() else []
    choices += [p for p in root.rglob(name) if p.is_dir() and (p / "datasets").is_dir()]
    choices = list(dict.fromkeys(p.resolve() for p in choices))
    if len(choices) != 1:
        raise ValueError(f"Expected one {name!r} datasets directory, found {choices}")
    return choices[0]


def _metadata(farm, stem):
    for name in (f"comma_{stem}.csv", f"{stem}.csv"):
        path = farm / name
        if path.exists():
            return path, _csv(path)
    raise FileNotFoundError(f"Missing {stem}.csv in {farm}")


def canonical_fingerprints(frame, columns, decimals=6):
    """Canonical float64, rounded, ordered raw readings; signed zero is equal.

    Metadata, timestamps, and labels are deliberately excluded. Missing values
    have one representation; infinities are rejected by the reader.
    """
    values = frame[list(columns)].to_numpy(dtype=np.float64, copy=True)
    values = np.round(values, decimals)
    values[values == 0] = 0.0
    values[np.isnan(values)] = np.nan
    return pd.util.hash_pandas_object(pd.DataFrame(values, columns=columns), index=False).to_numpy()


def _split(n, cfg, rng):
    cut = int(n * (1 - cfg.val_fraction))
    if cfg.val_split_mode == "random":
        ids = rng.permutation(n)
        return ids[:cut], ids[cut:], False
    if cfg.val_split_mode == "chronological":
        return np.arange(cut), np.arange(cut, n), False
    block = max(1, int(cfg.val_block_hours * 6))
    stride = max(2, round(1 / cfg.val_fraction))
    is_val = (np.arange(n) // block) % stride == stride - 1
    fallback = not is_val.any() or is_val.all()
    if fallback:
        return np.arange(cut), np.arange(cut, n), True
    return np.flatnonzero(~is_val), np.flatnonzero(is_val), False


def _angle_columns(description, sensors, enabled):
    if not enabled:
        return []
    if "sensor_name" not in description:
        raise ValueError("feature_description needs sensor_name")
    pat = re.compile(r"\b(degree|degrees|deg|angle|direction|azimuth)\b", re.IGNORECASE)
    bases = {str(row["sensor_name"]) for _, row in description.iterrows()
             if pat.search(" ".join(str(v) for k, v in row.items() if k != "sensor_name"))}
    if not bases:
        name_pat = re.compile(r"^(wind_direction|nacelle_direction|nacelle_position|"
                              r"pitch_angle|blade_angle|yaw_angle|wind_dir)", re.IGNORECASE)
        bases = {str(v) for v in description.sensor_name if name_pat.match(str(v))}
    return [c for c in sensors if any(c == b or c.startswith(b + "_") for b in bases)]


def _features(raw, sensors, angles):
    linear = [c for c in sensors if c not in angles]
    parts = [raw[:, [sensors.index(c) for c in linear]]]
    if angles:
        rad = np.deg2rad(raw[:, [sensors.index(c) for c in angles]])
        parts.extend([np.sin(rad), np.cos(rad)])
    return np.concatenate(parts, axis=1)


def _concat(parts):
    return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}


def _take(part, mask):
    return {key: value[mask] for key, value in part.items()}


def _overlap(left, right):
    return len(np.intersect1d(left, right))


def _load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def build_client_dataset(data_root, output_root, cfg, *, cache_dir=None,
                         expected_clients=22, expected_events=58, source_identity=None):
    """Build a complete immutable export, atomically, holding one turbine at a time.

    Cache reuse is deliberately disabled in this version: the legacy E00 cache
    rounds readings to float32, so it cannot reproduce canonical raw identities.
    ``cache_dir=None`` parses raw CSV and does not discover caches implicitly.
    """
    if not 0 < cfg.val_fraction < 1 or cfg.val_block_hours <= 0:
        raise ValueError("Invalid validation fraction or block hours")
    if cfg.val_split_mode not in {"blocked", "random", "chronological"}:
        raise ValueError("Invalid validation split mode")
    if not 0 <= cfg.max_nan_fraction < 1:
        raise ValueError("max_nan_fraction must be in [0, 1)")
    decimals = cfg.extra.get("fingerprint_decimals", 6)
    if not isinstance(decimals, int) or not 0 <= decimals <= 12:
        raise ValueError("fingerprint_decimals must be an integer in [0, 12]")
    if cache_dir is not None:
        raise ValueError("Cache reuse is disabled: legacy/unverified cache provenance cannot prove "
                         "canonical raw float64 precision; omit cache_dir to parse raw CSV")
    output = Path(output_root).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    farm = _farm(data_root, cfg.farm)
    info_path, info = _metadata(farm, "event_info")
    description_path, description = _metadata(farm, "feature_description")
    required_info = {"event_id", "event_label", "event_start", "event_end"}
    if not required_info <= set(info):
        raise ValueError(f"event_info missing {required_info - set(info)}")
    event_ids = pd.to_numeric(info.event_id, errors="raise")
    if event_ids.isna().any() or (event_ids % 1 != 0).any():
        raise ValueError("Event IDs must be nonmissing integers")
    info["event_id"] = event_ids.astype(int)
    if info.event_id.duplicated().any():
        raise ValueError("Duplicate event metadata")
    if not info.event_label.isin(["normal", "anomaly"]).all():
        raise ValueError("Unknown event label")
    for col in ("event_start", "event_end"):
        info[col] = pd.to_datetime(info[col], errors="coerce")
    anomalies = info.event_label == "anomaly"
    if (info.loc[anomalies, ["event_start", "event_end"]].isna().any().any()
            or (info.loc[anomalies, "event_start"] > info.loc[anomalies, "event_end"]).any()):
        raise ValueError("Missing or invalid anomaly window")
    files = {}
    for path in sorted((farm / "datasets").glob("*.csv")):
        match = re.fullmatch(r"(?:comma_)?(\d+)\.csv", path.name)
        if match:
            eid = int(match[1])
            if eid not in files or path.name.startswith("comma_"):
                files[eid] = path
    if set(files) != set(info.event_id):
        raise ValueError("Event files and metadata must match exactly")
    if expected_events is not None and len(files) != expected_events:
        raise ValueError(f"Expected {expected_events} events, found {len(files)}")
    first = _csv(files[min(files)], nrows=1)
    sensors = sorted(c for c in first if c not in DESCRIPTIVE
                     and (not cfg.use_avg_only or c.endswith("_avg"))
                     and (not cfg.drop_reactive or not c.startswith("reactive_power_")))
    if not sensors:
        raise ValueError("No sensor features selected")
    power = next((c for c in sensors if c.startswith("power_") and c.endswith("_avg")), None)
    wind = next((c for c in sensors if c.startswith("wind_speed_") and c.endswith("_avg")), None)
    sources = {str(eid): _hash(path) for eid, path in files.items()}
    source_metadata = {info_path.name: _hash(info_path), description_path.name: _hash(description_path)}

    assets, event_meta, pred_sets = {}, {}, []

    def read(eid):
        raw_header = _csv(files[eid], nrows=0)
        if not REQUIRED <= set(raw_header) or not set(sensors) <= set(raw_header):
            raise ValueError(f"Event {eid}: missing required columns")
        candidate = sorted(c for c in raw_header if c not in DESCRIPTIVE
                           and (not cfg.use_avg_only or c.endswith("_avg"))
                           and (not cfg.drop_reactive or not c.startswith("reactive_power_")))
        if candidate != sensors:
            raise ValueError(f"Event {eid}: sensor schema mismatch")
        columns = sorted(REQUIRED) + sensors
        df = _csv(files[eid], usecols=columns)
        df["time_stamp"] = pd.to_datetime(df.time_stamp, errors="raise")
        if df.time_stamp.isna().any() or df.asset_id.isna().any() or len(df.asset_id.unique()) != 1:
            raise ValueError(f"Event {eid}: missing time or mixed/missing assets")
        numeric_asset = pd.to_numeric(df.asset_id, errors="raise")
        if (numeric_asset % 1 != 0).any():
            raise ValueError(f"Event {eid}: asset IDs must be integers")
        df["asset_id"] = numeric_asset.astype(np.int64)
        df["status_type_id"] = pd.to_numeric(df.status_type_id, errors="raise")
        if df.status_type_id.isna().any():
            raise ValueError(f"Event {eid}: missing status")
        if not df.train_test.isin(["train", "prediction"]).all():
            raise ValueError(f"Event {eid}: unknown train_test label")
        for col in sensors:
            df[col] = pd.to_numeric(df[col], errors="raise").astype(np.float64)
        if np.isinf(df[sensors].to_numpy()).any():
            raise ValueError(f"Event {eid}: infinite sensor reading")
        if not (df.train_test == "train").any() or not (df.train_test == "prediction").any():
            raise ValueError(f"Event {eid}: missing training or prediction rows")
        return df.sort_values("time_stamp", kind="stable").reset_index(drop=True)

    # Only fingerprints (not all raw measurements) are retained across turbines.
    for eid in sorted(files):
        df = read(eid)
        aid = int(df.asset_id.iloc[0])
        row = info.loc[info.event_id == eid].iloc[0]
        if "asset_id" in info and int(row.asset_id) != aid:
            raise ValueError(f"Event {eid}: asset metadata mismatch")
        assets.setdefault(aid, []).append(eid)
        event_meta[eid] = {"event_id": eid, "asset_id": aid, "is_anomaly": row.event_label == "anomaly",
                           "event_start": row.event_start.isoformat() if pd.notna(row.event_start) else None,
                           "event_end": row.event_end.isoformat() if pd.notna(row.event_end) else None}
        pred_sets.append(canonical_fingerprints(df.loc[df.train_test == "prediction"], sensors, decimals))
    if expected_clients is not None and len(assets) != expected_clients:
        raise ValueError(f"Expected {expected_clients} clients, found {len(assets)}")
    global_pred = np.unique(np.concatenate(pred_sets))
    del pred_sets, df
    angle_candidates = _angle_columns(description, sensors, cfg.angle_transform)
    lower = np.full(len(sensors), np.inf)
    upper = np.full(len(sensors), -np.inf)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clients-build-", dir=output.parent) as temp:
        stage = Path(temp)
        raw_stage = stage / "raw"
        raw_stage.mkdir()
        audits = {}
        for aid, eids in sorted(assets.items()):
            parts = {key: [] for key in ("train", "val", "prediction")}
            rng = np.random.default_rng(cfg.seed)
            fallback_events = []
            for eid in eids:
                df = read(eid)
                normal = df.status_type_id.isin(cfg.normal_status_ids).to_numpy()
                fit_mask = normal & (df.train_test == "train").to_numpy()
                if power and wind:
                    implausible = (df[wind].between(cfg.wind_speed_cutin, cfg.wind_speed_rated)
                                   & (df[power].abs() <= cfg.power_near_zero_thresh)).to_numpy()
                    fit_mask &= ~implausible
                fp = canonical_fingerprints(df, sensors, decimals)
                base = {"X": df[sensors].to_numpy(), "fingerprint": fp,
                        "event_id": np.full(len(df), eid, dtype=np.int64),
                        "asset_id": np.full(len(df), aid, dtype=np.int64),
                        "status_type_id": df.status_type_id.to_numpy(), "eval_mask": normal,
                        "time_stamp": df.time_stamp.to_numpy(dtype="datetime64[ns]")}
                fit = _take(base, fit_mask)
                if len(fit["X"]) < 2:
                    raise ValueError(f"Event {eid}: fewer than two normal fitting rows")
                tr, va, fallback = _split(len(fit["X"]), cfg, rng)
                if fallback:
                    fallback_events.append(eid)
                parts["train"].append(_take(fit, tr))
                parts["val"].append(_take(fit, va))
                mask = (df.train_test == "prediction").to_numpy()
                pred = _take(base, mask)
                pred["status_type_id"] = df.loc[mask, "status_type_id"].to_numpy()
                pred["eval_mask"] = normal[mask]
                meta = event_meta[eid]
                pred["ground_truth"] = np.zeros(mask.sum(), dtype=np.int8)
                if meta["is_anomaly"]:
                    times = df.loc[mask, "time_stamp"]
                    pred["ground_truth"] = times.between(pd.Timestamp(meta["event_start"]),
                                                          pd.Timestamp(meta["event_end"])).to_numpy(np.int8)
                parts["prediction"].append(pred)
            parts = {key: _concat(values) for key, values in parts.items()}
            train, val, pred = (parts[key] for key in ("train", "val", "prediction"))
            ntr, nva = len(train["X"]), len(val["X"])
            tr_pred = np.isin(train["fingerprint"], global_pred)
            va_pred = np.isin(val["fingerprint"], global_pred)
            val = _take(val, ~va_pred)
            tr_val = ~tr_pred & np.isin(train["fingerprint"], val["fingerprint"])
            train = _take(train, ~(tr_pred | tr_val))
            if not len(train["X"]) or not len(val["X"]):
                raise ValueError(f"Client {aid}: empty cleaned training or validation partition")
            intersections = {"train_val": _overlap(train["fingerprint"], val["fingerprint"]),
                             "train_prediction": _overlap(train["fingerprint"], pred["fingerprint"]),
                             "val_prediction": _overlap(val["fingerprint"], pred["fingerprint"]),
                             "train_global_prediction": _overlap(train["fingerprint"], global_pred),
                             "val_global_prediction": _overlap(val["fingerprint"], global_pred)}
            if any(intersections.values()):
                raise AssertionError(f"Client {aid}: intersections remain {intersections}")
            for j in range(len(sensors)):
                finite = train["X"][:, j][np.isfinite(train["X"][:, j])]
                if finite.size:
                    lower[j] = min(lower[j], float(finite.min()))
                    upper[j] = max(upper[j], float(finite.max()))
            audits[str(aid)] = {
                "train_rows_before": ntr, "val_rows_before": nva,
                "train_removed_prediction": int(tr_pred.sum()), "val_removed_prediction": int(va_pred.sum()),
                "train_removed_validation": int(tr_val.sum()), "intersections": intersections,
                "chronological_fallback_events": fallback_events,
                **{f"{key}_rows": len(value["X"]) for key, value in
                   (("train", train), ("val", val), ("prediction", pred))},
                **{f"{key}_unique_fingerprints": len(np.unique(value["fingerprint"]))
                   for key, value in (("train", train), ("val", val), ("prediction", pred))},
            }
            for key, value in (("train", train), ("val", val), ("prediction", pred)):
                np.savez_compressed(raw_stage / f"{aid}_{key}.npz", **value)
            del parts, train, val, pred, df

        # A shared model must not fit one client's held-out calibration readings
        # through another client's training data. Fail closed on this additional
        # cross-turbine pathology rather than silently changing the local split.
        validation_indexes = []
        for aid in assets:
            with np.load(raw_stage / f"{aid}_val.npz", allow_pickle=False) as saved:
                validation_indexes.append(saved["fingerprint"])
        global_validation = np.unique(np.concatenate(validation_indexes))
        del validation_indexes
        for aid in assets:
            with np.load(raw_stage / f"{aid}_train.npz", allow_pickle=False) as saved:
                if _overlap(saved["fingerprint"], global_validation):
                    raise ValueError(f"Client {aid}: cross-client fitting-validation overlap; "
                                     "shared-model calibration would not be held out")
            audits[str(aid)]["intersections"]["train_global_validation"] = 0

        # Angle range checks see only cleaned fitting rows across clients.
        angles = [c for c in angle_candidates
                  if -190 <= lower[sensors.index(c)] <= 5 and 170 <= upper[sensors.index(c)] <= 370]
        schema = ([c for c in sensors if c not in angles]
                  + [f"{c}__sin" for c in angles] + [f"{c}__cos" for c in angles])
        schema_hash = _object_hash(schema)
        artifact_hashes = {}
        client_dir = stage / "clients"
        client_dir.mkdir()
        for aid, eids in sorted(assets.items()):
            train = _load_npz(raw_stage / f"{aid}_train.npz")
            x = _features(train["X"], sensors, angles)
            finite = np.isfinite(x)
            count = finite.sum(axis=0)
            mean = np.divide(np.where(finite, x, 0).sum(axis=0), count,
                             out=np.zeros(x.shape[1]), where=count > 0)
            variance = np.divide(np.where(finite, (x - mean) ** 2, 0).sum(axis=0), count,
                                 out=np.zeros(x.shape[1]), where=count > 0)
            scale = np.sqrt(variance)
            active = ((1 - count / len(x)) <= cfg.max_nan_fraction)
            scale[(scale <= 1e-12) | ~active] = 1.0
            if not active.any():
                raise ValueError(f"Client {aid}: no active fitting features")
            scaler = {"mean": mean.tolist(), "scale": scale.tolist(), "active_mask": active.tolist(),
                      "finite_count": count.tolist(), "fit_rows": len(x),
                      "inactive_policy": "zero-fill input and target; include all dimensions in MSE and aggregation"}
            destination = client_dir / str(aid)
            destination.mkdir()
            for key in ("train", "val", "prediction"):
                part = train if key == "train" else _load_npz(raw_stage / f"{aid}_{key}.npz")
                values = (_features(part["X"], sensors, angles) - mean) / scale
                values[:, ~active] = 0
                values = np.nan_to_num(values, nan=0, posinf=0, neginf=0)
                if (np.abs(values) > np.finfo(np.float32).max).any():
                    raise ValueError(f"Client {aid}: transformed values overflow float32")
                part["raw_sensors"] = part["X"]
                part["X"] = values.astype(np.float32)
                target = destination / f"{key}.npz"
                np.savez_compressed(target, **part)
                artifact_hashes[target.relative_to(stage).as_posix()] = _hash(target)
                if key in {"prediction", "val"}:
                    fp_path = destination / f"{key}_fingerprints.npy"
                    np.save(fp_path, part["fingerprint"], allow_pickle=False)
                    artifact_hashes[fp_path.relative_to(stage).as_posix()] = _hash(fp_path)
                (raw_stage / f"{aid}_{key}.npz").unlink()
            sidecar = destination / "client.json"
            _write_json(sidecar, {"asset_id": aid, "schema": schema, "schema_hash": schema_hash,
                                  "scaler": scaler, "events": [event_meta[eid] for eid in eids],
                                  "audit": audits[str(aid)]})
            artifact_hashes[sidecar.relative_to(stage).as_posix()] = _hash(sidecar)
        # Do not publish if inputs changed while the multi-pass build was running.
        if any(_hash(path) != sources[str(eid)] for eid, path in files.items()):
            raise ValueError("Raw source changed during partition build")
        if any(_hash(farm / name) != digest for name, digest in source_metadata.items()):
            raise ValueError("Metadata changed during partition build")
        # raw staging files are outside the completed dataset and removed by the
        # TemporaryDirectory owner. Only the publish directory is renamed.
        publish = stage / "publish"
        publish.mkdir()
        client_dir.rename(publish / "clients")
        prep = preprocessing_config(cfg)
        manifest = {"format_version": 1, "clients": sorted(assets), "n_events": len(files),
                    "schema": schema, "schema_hash": schema_hash, "raw_sensor_cols": sensors,
                    "angle_cols": angles, "angle_candidates_rejected": sorted(set(angle_candidates) - set(angles)),
                    "fingerprint_decimals": decimals, "preprocessing_config": prep,
                    "fingerprint_policy": "ordered raw float64 readings, rounded decimals, signed zero normalized",
                    "preprocessing_hash": _object_hash(prep), "config": json.loads(_json(asdict(cfg))),
                    "raw_sources": sources, "metadata_sources": source_metadata,
                    "cache_manifest_sha256": None,
                    "artifacts": artifact_hashes, "audit": audits,
                    "global_prediction_unique_fingerprints": len(global_pred),
                    "global_validation_unique_fingerprints": len(global_validation),
                    "power_filter": {"power": power, "wind": wind, "enabled": bool(power and wind)},
                    "split_policy": "per-event retained-row blocks; chronological fallback for short events; no embargo",
                    "duplicate_policy": "retain duplicates within each split; remove train matches to validation",
                    "scaler_policy": "local cleaned-train means/std; missing and inactive dimensions become zero",
                    "inactive_policy": "zero-fill input and target; include all common dimensions in MSE and aggregation",
                    "schema_policy": "common ordered schema; angle ranges use cleaned fitting rows only",
                    "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                    "pandas": pd.__version__},
                    "implementation_sha256": _hash(Path(__file__))}
        manifest["source_identity"] = source_identity
        _write_json(publish / "dataset_manifest.json", manifest)
        (publish / "COMPLETE").write_text(_hash(publish / "dataset_manifest.json"), encoding="ascii")
        publish.rename(output)
    return manifest


def load_client_dataset(output_root, asset_id, verify=True):
    """Reload arrays without pickle and independently recheck identity/schema/audit.

    Hashes detect accidental changes, not malicious replacement of both manifest
    and completion marker. ``verify=True`` checks this client's artifacts and all
    clients' small prediction fingerprint indexes to reconstruct the global
    prediction universe. Loading every client checks the entire artifact set.
    """
    root = Path(output_root)
    manifest_path = root / "dataset_manifest.json"
    if not (root / "COMPLETE").exists():
        raise ValueError("Dataset is incomplete (missing COMPLETE marker)")
    if (root / "COMPLETE").read_text(encoding="ascii") != _hash(manifest_path):
        raise ValueError("Dataset manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aid = int(asset_id)
    if aid not in manifest["clients"]:
        raise KeyError(f"Unknown client {aid}")
    if verify:
        expected_names = {f"clients/{other}/{split}_fingerprints.npy"
                          for other in manifest["clients"] for split in ("prediction", "val")}
        expected_names |= {f"clients/{aid}/{name}" for name in
                           ("train.npz", "val.npz", "prediction.npz", "client.json")}
        if not expected_names <= set(manifest["artifacts"]):
            raise ValueError("Missing required artifact hashes")
        for relative, digest in manifest["artifacts"].items():
            if relative not in expected_names:
                continue
            target = (root / relative).resolve()
            if not target.is_relative_to(root.resolve()) or _hash(target) != digest:
                raise ValueError(f"Artifact hash mismatch: {relative}")
    directory = root / "clients" / str(aid)
    client = json.loads((directory / "client.json").read_text(encoding="utf-8"))
    if (client["schema"] != manifest["schema"] or client["schema_hash"] != _object_hash(manifest["schema"])
            or client["schema_hash"] != manifest["schema_hash"]):
        raise ValueError("Client/global schema mismatch")
    result = {key: _load_npz(directory / f"{key}.npz") for key in ("train", "val", "prediction")}
    if verify:
        x = _features(result["train"]["raw_sensors"], manifest["raw_sensor_cols"], manifest["angle_cols"])
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        mean = np.divide(np.where(finite, x, 0).sum(axis=0), counts,
                         out=np.zeros(x.shape[1]), where=counts > 0)
        variance = np.divide(np.where(finite, (x - mean) ** 2, 0).sum(axis=0), counts,
                             out=np.zeros(x.shape[1]), where=counts > 0)
        scale = np.sqrt(variance)
        active = ((1 - counts / len(x)) <= manifest["preprocessing_config"]["max_nan_fraction"])
        scale[(scale <= 1e-12) | ~active] = 1
        if (not np.array_equal(mean, client["scaler"]["mean"])
                or not np.array_equal(scale, client["scaler"]["scale"])
                or not np.array_equal(active, client["scaler"]["active_mask"])
                or not np.array_equal(counts, client["scaler"]["finite_count"])
                or client["scaler"]["fit_rows"] != len(x)):
            raise ValueError(f"Client {aid}: scaler does not reproduce from cleaned fitting rows")
    for key, part in result.items():
        if (part["X"].dtype != np.float32 or part["X"].ndim != 2
                or part["X"].shape[1] != len(client["schema"]) or not np.isfinite(part["X"]).all()
                or not len(part["X"]) or not (part["asset_id"] == aid).all()
                or any(len(v) != len(part["X"]) for v in part.values())
                or len(part["X"]) != client["audit"][f"{key}_rows"]):
            raise ValueError(f"Client {aid}: invalid {key} arrays")
        if verify:
            raw = pd.DataFrame(part["raw_sensors"], columns=manifest["raw_sensor_cols"])
            expected_fp = canonical_fingerprints(raw, manifest["raw_sensor_cols"], manifest["fingerprint_decimals"])
            if not np.array_equal(expected_fp, part["fingerprint"]):
                raise ValueError(f"Client {aid}: raw fingerprints do not reproduce")
            expected_x = (_features(part["raw_sensors"], manifest["raw_sensor_cols"], manifest["angle_cols"])
                          - np.asarray(client["scaler"]["mean"])) / np.asarray(client["scaler"]["scale"])
            expected_x[:, ~np.asarray(client["scaler"]["active_mask"])] = 0
            expected_x = np.nan_to_num(expected_x, nan=0, posinf=0, neginf=0).astype(np.float32)
            if not np.array_equal(expected_x, part["X"]):
                raise ValueError(f"Client {aid}: transformed arrays do not reproduce")
    if verify:
        global_pred, global_val = [], []
        for other in manifest["clients"]:
            global_pred.append(np.load(root / "clients" / str(other) / "prediction_fingerprints.npy",
                                       allow_pickle=False))
            global_val.append(np.load(root / "clients" / str(other) / "val_fingerprints.npy", allow_pickle=False))
            if other == aid and not np.array_equal(global_pred[-1], result["prediction"]["fingerprint"]):
                raise ValueError("Prediction fingerprint index does not match prediction rows")
            if other == aid and not np.array_equal(global_val[-1], result["val"]["fingerprint"]):
                raise ValueError("Validation fingerprint index does not match validation rows")
        global_pred = np.unique(np.concatenate(global_pred))
        global_val = np.unique(np.concatenate(global_val))
        tr, va, pr = [result[key]["fingerprint"] for key in ("train", "val", "prediction")]
        intersections = [_overlap(tr, va), _overlap(tr, pr), _overlap(va, pr),
                         _overlap(tr, global_pred), _overlap(va, global_pred), _overlap(tr, global_val)]
        if any(intersections):
            raise ValueError(f"Client {aid}: reloaded partition intersections {intersections}")
        if not np.array_equal(result["prediction"]["eval_mask"],
                              np.isin(result["prediction"]["status_type_id"],
                                      manifest["preprocessing_config"]["normal_status_ids"])):
            raise ValueError("Evaluation mask is not status-only")
    return {**result, "schema": client["schema"], "scaler": client["scaler"],
            "events": client["events"], "audit": client["audit"]}


def validate_client_dataset(output_root):
    """Independently verify every exported client, holding only one at a time.

    Returns the verified manifest. Hashes, raw fingerprints, preprocessing,
    fitting-only scaler statistics, status masks and intersections are checked.
    """
    root = Path(output_root)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for aid in manifest["clients"]:
        load_client_dataset(root, aid, verify=True)
    return manifest
