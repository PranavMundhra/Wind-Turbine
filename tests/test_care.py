import numpy as np
import pandas as pd
import pytest

from windturbine.scoring import (
    accuracy_care,
    criticality,
    earliness_ws,
    event_detected,
    fbeta_from_counts,
    pick_threshold,
)


def test_criticality_counts_up_down_and_floors_at_zero():
    normal = np.array([True] * 6)
    pred = np.array([1, 1, 1, 0, 0, 0])
    assert criticality(normal, pred).tolist() == [1, 2, 3, 2, 1, 0]
    assert criticality(normal, np.zeros(4, dtype=int)).tolist() == [0, 0, 0, 0]


def test_criticality_unchanged_on_abnormal_status():
    normal = np.array([True, True, False, False, True])
    pred = np.array([1, 1, 1, 0, 1])
    assert criticality(normal, pred).tolist() == [1, 2, 2, 2, 3]


def test_event_detected_uses_threshold_72_by_default():
    n = 80
    rec = {"normal_mask": np.ones(n, bool), "prediction": np.ones(n, int)}
    assert event_detected(rec) == (1, 80)
    rec["prediction"][:10] = 0  # counter reaches 70 only
    assert event_detected(rec) == (0, 70)
    assert event_detected(rec, thr=70) == (1, 70)


def test_event_detected_empty_record():
    rec = {"normal_mask": np.array([], bool), "prediction": np.array([], int)}
    assert event_detected(rec) == (0, 0)


def test_accuracy_is_true_negative_rate_on_normal_rows_only():
    rec = {
        "normal_mask": np.array([True, True, True, True, False]),
        "prediction": np.array([0, 0, 0, 1, 1]),  # last row is abnormal status, ignored
    }
    assert accuracy_care(rec) == pytest.approx(0.75)


def test_fbeta_matches_hand_value_and_zero_guard():
    # beta=0.5, tp=8, fp=2, fn=4 -> 1.25*8 / (1.25*8 + 0.25*4 + 2) = 10/13
    assert fbeta_from_counts(8, 2, 4) == pytest.approx(10 / 13)
    assert fbeta_from_counts(0, 0, 0) == 0.0


def test_earliness_weights_flat_then_decay():
    t = pd.date_range("2020-01-01", periods=11, freq="1h")
    start, end = t[0], t[-1]
    flagged_early = {"time_stamp": t, "normal_mask": np.ones(11, bool),
                     "prediction": np.array([1] * 6 + [0] * 5)}
    flagged_late = {"time_stamp": t, "normal_mask": np.ones(11, bool),
                    "prediction": np.array([0] * 5 + [1] * 6)}
    assert earliness_ws(flagged_early, start, end) > earliness_ws(flagged_late, start, end)
    assert earliness_ws(flagged_early, None, end) is None


def test_pick_threshold_quantile_and_warning(capsys):
    scores = np.concatenate([np.ones(990), np.full(10, 1000.0)])
    thr, info = pick_threshold(scores, q=0.995, tag="evt")
    assert thr == pytest.approx(np.quantile(scores, 0.995))
    assert info["thr_over_median"] > 10
    assert "WARNING evt" in capsys.readouterr().out
