import numpy as np
import pandas as pd
import pytest

from windturbine.scoring import accuracy_care, aggregate_event_metrics, evaluate_care


def fixture_events():
    times = pd.date_range("2020-01-01", periods=100, freq="10min", tz="UTC")
    meta = {
        1: {"asset_id": 10, "is_anomaly": True, "event_start": times[10], "event_end": times[-1]},
        2: {"asset_id": 20, "is_anomaly": False, "event_start": None, "event_end": None},
        3: {"asset_id": 30, "is_anomaly": False, "event_start": None, "event_end": None},
    }
    records = {eid: {"time_stamp": times, "normal_mask": np.ones(100, bool),
                         "prediction": np.zeros(100, int)} for eid in meta}
    records[1]["prediction"][10:] = 1
    records[2]["prediction"][:30] = 1
    records[3]["prediction"][:10] = 1
    return meta, records


def test_client_reports_exactly_reproduce_direct_care_with_partial_support():
    meta, records = fixture_events()
    direct, _ = evaluate_care(meta, records, verbose=False)
    reports = []
    for eid in meta:
        local, report = evaluate_care({eid: meta[eid]}, {eid: records[eid]}, verbose=False)
        assert np.isnan(local["CARE"])  # Single-type clients cannot form complete CARE.
        reports.append(report)
    combined = aggregate_event_metrics(pd.concat(reports, ignore_index=True))
    assert combined == direct
    assert direct["Accuracy"] == pytest.approx(0.8)
    assert direct["Reliability"] == 1
    assert direct["support"] == {"coverage": 1, "earliness": 1, "accuracy": 2}


def test_no_eligible_normal_event_does_not_fabricate_perfect_accuracy():
    meta, records = fixture_events()
    records[2]["normal_mask"][:] = False
    records[3]["normal_mask"][:] = False
    assert np.isnan(accuracy_care(records[2]))
    result, report = evaluate_care(meta, records, verbose=False)
    assert result["support"]["accuracy"] == 0
    assert np.isnan(result["CARE"])
    assert report.accuracy.isna().all()


def test_no_detected_events_guard_and_low_accuracy_guard_apply_on_server():
    meta, records = fixture_events()
    records[1]["prediction"][10:] = 0
    records[1]["prediction"][10:20] = 1
    result, _ = evaluate_care(meta, records, verbose=False)
    assert result["Coverage"] > 0 and result["Earliness"] > 0
    assert result["CARE"] == 0  # Point flags alone are not a detected event.
    records[1]["prediction"][10:] = 1
    records[2]["prediction"][:] = 1
    records[3]["prediction"][:] = 1
    result, _ = evaluate_care(meta, records, verbose=False)
    assert result["CARE"] == result["Accuracy"] == 0


def test_empty_anomaly_window_support_is_explicit():
    meta, records = fixture_events()
    records[1]["normal_mask"][10:] = False
    result, report = evaluate_care(meta, records, verbose=False)
    assert result["support"]["coverage"] == result["support"]["earliness"] == 0
    assert np.isnan(result["CARE"])
    assert np.isnan(report.iloc[0].coverage)


def test_duplicate_event_reports_and_invalid_records_are_rejected():
    meta, records = fixture_events()
    _, report = evaluate_care(meta, records, verbose=False)
    with pytest.raises(ValueError, match="unique"):
        aggregate_event_metrics(pd.concat([report, report]))
    with pytest.raises(ValueError, match="expected evaluation cohort"):
        aggregate_event_metrics(report.iloc[:-1], expected_event_ids=meta)
    with pytest.raises(ValueError, match="same events"):
        evaluate_care(meta, {1: records[1]})
    records[1]["prediction"][0] = 2
    with pytest.raises(ValueError, match="binary"):
        evaluate_care(meta, records)
