import numpy as np
import pytest

from make_fake_data import SENSORS, make_farm
from windturbine.audit import leaked_fraction, row_fingerprints


def test_fingerprints_are_deterministic_and_row_wise():
    a, _, _ = make_farm()
    f1, f2 = row_fingerprints(a, SENSORS), row_fingerprints(a, SENSORS)
    assert (f1 == f2).all() and len(f1) == len(a)


def test_planted_overlap_is_recovered_exactly():
    a, b, planted = make_farm(n_rows=2000, overlap=700)
    got = leaked_fraction(row_fingerprints(a, SENSORS), row_fingerprints(b, SENSORS))
    assert got == pytest.approx(planted)


def test_independent_events_have_no_collisions():
    a, b, _ = make_farm(overlap=0)
    assert leaked_fraction(row_fingerprints(a, SENSORS), row_fingerprints(b, SENSORS)) == 0.0


def test_rounding_to_six_decimals_matches_the_notebooks():
    a, _, _ = make_farm()
    a[SENSORS] = a[SENSORS].round(6)  # values already on the 6-decimal grid
    jittered = a.copy()
    jittered[SENSORS] = jittered[SENSORS] + 1e-9  # below the grid spacing, so rounds back
    got = leaked_fraction(row_fingerprints(a, SENSORS), row_fingerprints(jittered, SENSORS))
    assert got == 1.0


def test_empty_prediction_set():
    assert leaked_fraction(np.array([1, 2]), np.array([])) == 0.0
