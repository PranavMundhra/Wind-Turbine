"""Row-fingerprint duplicate detection.

``row_fingerprints`` is copied from the arm notebooks (``SENSOR_COLS`` became a
required argument). ``leaked_fraction`` is new and small: it states in code the
measurement MODEL_DOCUMENTATION.md section 3 reports (share of prediction rows whose
fingerprint also appears in the training corpus).

Caveat from the docs: identical fingerprints at two timestamps can also occur on
flatlined or all-NaN rows (0.008% in Arm A), so this is a floor to compare against,
not proof of leakage for any single row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def row_fingerprints(df: pd.DataFrame, cols, decimals: int = 6) -> np.ndarray:
    sub = df[list(cols)].round(decimals)
    return pd.util.hash_pandas_object(sub, index=False).to_numpy()


def leaked_fraction(train_fps, pred_fps) -> float:
    """Fraction of ``pred_fps`` present in ``train_fps``."""
    pred_fps = np.asarray(pred_fps)
    if pred_fps.size == 0:
        return 0.0
    return float(np.isin(pred_fps, np.asarray(train_fps)).mean())
