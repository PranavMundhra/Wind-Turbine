"""Synthetic stand-in for Farm C with the duplication pathology planted.

Two "event files" for the same turbine share a block of byte-identical rows, as in
the real dataset (MODEL_DOCUMENTATION.md section 3). Used by tests/test_audit.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = [f"sensor_{i}_avg" for i in range(12)]


def make_event(n: int, seed: int, start: str = "2020-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(SENSORS))), columns=SENSORS)
    df.insert(0, "time_stamp", pd.date_range(start, periods=n, freq="10min"))
    return df


def make_farm(n_rows: int = 2000, overlap: int = 700, seed: int = 0):
    """Return (event_a, event_b, planted_fraction_of_b).

    ``event_b`` reuses the last ``overlap`` rows of ``event_a`` verbatim (values and
    timestamps), so the true leaked fraction of ``event_b`` is overlap / n_rows.
    """
    a = make_event(n_rows, seed)
    b = make_event(n_rows, seed + 1)
    if overlap > 0:  # a[-0:] would select the whole frame
        b.iloc[:overlap, 1:] = a.iloc[-overlap:, 1:].to_numpy()
        b.iloc[:overlap, 0] = a.iloc[-overlap:, 0].to_numpy()
    return a, b, overlap / n_rows


if __name__ == "__main__":
    a, b, frac = make_farm()
    print(f"event_a {a.shape}, event_b {b.shape}, planted overlap {frac:.1%}")
