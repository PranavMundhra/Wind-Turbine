"""One CARE definition for local baselines and federated client reports.

Clients report unrounded per-event components. The server concatenates those
reports and calls ``aggregate_event_metrics`` once: training row weights never
enter metric aggregation, and client CARE scores must not be averaged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .care import accuracy_care, earliness_ws, event_detected, fbeta_from_counts


def _options(beta, criticality_threshold, care_weights):
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    if int(criticality_threshold) != criticality_threshold or criticality_threshold <= 0:
        raise ValueError("criticality_threshold must be a positive integer")
    weights = np.asarray(care_weights, dtype=float)
    if weights.shape != (4,) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("care_weights must contain four finite nonnegative values with positive total")
    return weights


def aggregate_event_metrics(per_event, *, beta=0.5, care_weights=(1, 1, 1, 2),
                            expected_event_ids=None, verbose=False):
    """Compute CARE from client event reports, including clients with partial support.

    Required columns are ``event_id``, ``is_anomaly`` and ``detected``. Components
    ``coverage``, ``earliness`` and ``accuracy`` are unrounded values or NaN for
    absent support. Return the same result mapping as ``evaluate_care``.
    Event IDs must be unique across reports. Reliability uses all event decisions;
    an event with no eligible timestamps has no supported pointwise component.
    """
    weights = _options(beta, 72, care_weights)
    frame = pd.DataFrame(per_event).copy()
    required = {"event_id", "is_anomaly", "detected"}
    if not required.issubset(frame):
        raise ValueError(f"event reports missing columns: {sorted(required - set(frame))}")
    if frame.event_id.isna().any() or frame.event_id.duplicated().any():
        raise ValueError("event reports must have unique, nonmissing event_id values")
    if expected_event_ids is not None and set(frame.event_id) != set(expected_event_ids):
        raise ValueError("received event IDs do not match the expected evaluation cohort")
    for name in ("is_anomaly", "detected"):
        if not frame[name].isin([0, 1]).all():
            raise ValueError(f"{name} must contain binary event decisions")
    for name in ("coverage", "earliness", "accuracy"):
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        supported = frame[name].notna()
        if not frame.loc[supported, name].between(0, 1).all():
            raise ValueError(f"{name} must be in [0, 1] or NaN for absent support")
        wrong_type = frame.is_anomaly.eq(1 if name == "accuracy" else 0)
        if (supported & wrong_type).any():
            raise ValueError(f"{name} is reported on the wrong event type")
    yt, yp = frame.is_anomaly.to_numpy(), frame.detected.to_numpy()
    confusion = {
        "tp": int(((yt == 1) & (yp == 1)).sum()),
        "fp": int(((yt == 0) & (yp == 1)).sum()),
        "fn": int(((yt == 1) & (yp == 0)).sum()),
        "tn": int(((yt == 0) & (yp == 0)).sum()),
    }
    support = {name: int(frame[name].notna().sum()) for name in ("coverage", "earliness", "accuracy")}
    c, e, a = (float(frame[name].mean()) for name in ("coverage", "earliness", "accuracy"))
    r = fbeta_from_counts(confusion["tp"], confusion["fp"], confusion["fn"], beta=beta)
    if not np.isfinite([c, e, a]).all():
        care = float("nan")
    elif not yp.any():
        care = 0.0  # No event reached the criticality threshold.
    elif a < 0.5:
        care = a
    else:
        care = float(np.dot(weights, [c, e, r, a]) / weights.sum())
    out = {
        "Coverage": c, "Earliness": e, "Reliability": r, "Accuracy": a, "CARE": care,
        "support": support, "event_confusion": confusion,
        "n_anomaly_events": int((yt == 1).sum()), "n_normal_events": int((yt == 0).sum()),
    }
    if verbose:
        for name in ("Coverage", "Earliness", "Reliability", "Accuracy", "CARE"):
            print(f"{name:<12} {out[name]:.6f}")
        print("Event confusion:", confusion)
        print("Supporting events:", support)
    return out


def evaluate_care(meta, records, *, beta=0.5, criticality_threshold=72,
                  care_weights=(1, 1, 1, 2), verbose=True):
    """Evaluate event predictions using a status-only ``normal_mask``.

    ``meta`` maps IDs to ``is_anomaly``, ``event_start``, ``event_end`` and
    ``asset_id``. ``records`` maps the same IDs to timestamp, binary prediction,
    and boolean ``normal_mask`` arrays. Power-curve filters belong to training,
    not this evaluation mask. Empty eligible events do not earn perfect Accuracy.
    """
    _options(beta, criticality_threshold, care_weights)
    if set(meta) != set(records):
        raise ValueError("metadata and prediction records must contain exactly the same events")
    rows = []
    for eid, info in meta.items():
        raw = records[eid]
        prediction = np.asarray(raw["prediction"])
        mask = np.asarray(raw["normal_mask"])
        times = pd.DatetimeIndex(pd.to_datetime(raw["time_stamp"]))
        if prediction.ndim != 1 or mask.shape != prediction.shape or len(times) != len(prediction):
            raise ValueError(f"event {eid}: timestamp, prediction and mask lengths must agree")
        if mask.dtype != np.dtype(bool) or not np.isin(prediction, [0, 1]).all():
            raise ValueError(f"event {eid}: require a boolean mask and binary predictions")
        if times.isna().any() or not times.is_monotonic_increasing:
            raise ValueError(f"event {eid}: timestamps must be valid and time ordered")
        rec = {"prediction": prediction, "normal_mask": mask, "time_stamp": times}
        detected, peak = event_detected(rec, thr=criticality_threshold)
        label = info["is_anomaly"]
        if label not in (0, 1, False, True):
            raise ValueError(f"event {eid}: is_anomaly must be binary")
        row = {
            "event_id": eid, "asset_id": info.get("asset_id"), "is_anomaly": int(label),
            "detected": detected, "peak_criticality": peak, "n_pred_rows": len(prediction),
            "n_eligible_rows": int(mask.sum()),
            "flagged_pct": float(prediction[mask].mean() * 100) if mask.any() else np.nan,
            "coverage": np.nan, "earliness": np.nan, "accuracy": np.nan,
        }
        if label:
            start, end = info.get("event_start"), info.get("event_end")
            if pd.isna(start) or pd.isna(end) or start is None or end is None:
                raise ValueError(f"event {eid}: anomaly window is required")
            start, end = pd.Timestamp(start), pd.Timestamp(end)
            if end < start:
                raise ValueError(f"event {eid}: anomaly window ends before it starts")
            gt = np.asarray((times >= start) & (times <= end))
            if (gt & mask).any():
                p, g = prediction[mask], gt[mask]
                row["coverage"] = fbeta_from_counts(
                    int(((p == 1) & g).sum()), int(((p == 1) & ~g).sum()),
                    int(((p == 0) & g).sum()), beta=beta,
                )
                row["earliness"] = earliness_ws(rec, start, end)
        else:
            row["accuracy"] = accuracy_care(rec)
        rows.append(row)
    columns = ["event_id", "asset_id", "is_anomaly", "detected", "peak_criticality", "n_pred_rows",
               "n_eligible_rows", "flagged_pct", "coverage", "earliness", "accuracy"]
    per_event = pd.DataFrame(rows, columns=columns)
    return aggregate_event_metrics(per_event, beta=beta, care_weights=care_weights, verbose=verbose), per_event
