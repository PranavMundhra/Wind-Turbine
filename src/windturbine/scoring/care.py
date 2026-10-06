"""CARE-score building blocks.

Copied from ``care_farmC_isolated`` / ``care_farmC_pooled`` (the two copies were
identical). Module-level constants became keyword arguments with the same defaults;
no logic was changed. ``fbeta`` was renamed ``fbeta_from_counts`` to say what it takes.

A ``rec`` is the per-event record the notebooks build: a mapping with
``prediction`` (0/1 array), ``normal_mask`` (bool array) and ``time_stamp``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fbeta_from_counts(tp, fp, fn, beta: float = 0.5) -> float:
    num = (1 + beta**2) * tp
    den = (1 + beta**2) * tp + (beta**2) * fn + fp
    return float(num / den) if den > 0 else 0.0


def criticality(status_normal, predictions) -> np.ndarray:
    """FAQ: +1 when an anomaly is flagged on a normal-status timestamp, -1 when
    it is not, unchanged on abnormal-status timestamps. Floored at 0."""
    c = np.zeros(len(predictions) + 1, dtype=np.int32)
    for i in range(1, len(predictions) + 1):
        if not status_normal[i - 1]:
            c[i] = c[i - 1]
        elif predictions[i - 1] == 1:
            c[i] = c[i - 1] + 1
        else:
            c[i] = max(c[i - 1] - 1, 0)
    return c[1:]


def event_detected(rec, thr: int = 72):
    """Return (detected, max_criticality). Detected when the counter reaches ``thr``."""
    c = criticality(rec["normal_mask"], rec["prediction"])
    return (int(c.max() >= thr), int(c.max()) if len(c) else 0) if len(c) else (0, 0)


def accuracy_care(rec) -> float:
    """True-negative rate over normal-status timestamps of a normal event."""
    p = rec["prediction"][rec["normal_mask"]]
    tn, fp = np.sum(p == 0), np.sum(p == 1)
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0


def earliness_ws(rec, start, end):
    """Weighted coverage inside the event window: weight 1.0 through the first
    half, decaying linearly to 0 at the end. NOTE: not verified against the
    paper's Figure 1 (see docs/MODEL_DOCUMENTATION.md, section 8)."""
    if start is None or end is None or pd.isna(start) or pd.isna(end):
        return None
    t = pd.to_datetime(rec["time_stamp"])
    inside = np.asarray((t >= start) & (t <= end)) & rec["normal_mask"]
    if inside.sum() == 0:
        return None
    span = (end - start).total_seconds()
    rel = (np.asarray(pd.to_datetime(rec["time_stamp"])[inside]) - np.datetime64(start))
    rel = rel.astype("timedelta64[s]").astype(float) / span if span > 0 else np.zeros(inside.sum())
    w = np.where(rel <= 0.5, 1.0, np.clip(1.0 - (rel - 0.5) / 0.5, 0.0, 1.0))
    return float((w * rec["prediction"][inside]).sum() / w.sum()) if w.sum() > 0 else 0.0
