"""Tests for the refit-aware jump classifier."""

import numpy as np

from naviguard.anomaly.refit import EDGE, REFIT, SPIKE, SUSTAINED, classify_jumps, shape_counts


def _z(n=60, seed=0):
    return np.random.default_rng(seed).normal(0, 0.5, n)


def _contig(n):
    return np.r_[np.ones(n - 1, dtype=bool), False]


def test_isolated_level_shift_is_a_refit_and_not_flagged():
    z = _z()
    z[20] = 9.0                                   # one jump, series continues normally
    alarms, events = classify_jumps(z, _contig(len(z)))
    assert events == [(20, REFIT, -1)]
    assert not alarms.any()


def test_full_reversion_is_a_spike_flagged_on_the_next_record():
    z = _z()
    z[20], z[21] = 9.0, -9.0                      # +A then -A: a single bad sample
    alarms, events = classify_jumps(z, _contig(len(z)))
    assert events[0][:2] == (20, SPIKE)
    assert np.flatnonzero(alarms).tolist() == [21]


def test_partial_reversion_counts_as_refit():
    z = _z()
    z[20], z[21] = 9.0, -4.5                      # reverts only half: typical of natural refits
    alarms, events = classify_jumps(z, _contig(len(z)))
    assert events[0][1] == REFIT and not alarms.any()


def test_sustained_same_sign_excursion_is_flagged_when_confirmed():
    z = _z()
    z[20:24] = [6.0, 5.0, 5.0, 5.0]               # ramp-like drift
    alarms, events = classify_jumps(z, _contig(len(z)), look=3)
    labels = {onset: label for onset, label, _ in events}
    assert labels[20] == SUSTAINED
    assert alarms[22]                             # confirmed at the second same-sign excursion


def test_gap_inside_lookahead_is_an_unflagged_edge():
    z = _z()
    z[20] = 9.0
    contig = _contig(len(z))
    contig[21] = False                            # a data gap right after the jump
    alarms, events = classify_jumps(z, contig)
    assert events == [(20, EDGE, -1)] and not alarms.any()


def test_shape_counts():
    z = _z()
    z[10] = 8.0
    z[30], z[31] = 8.0, -8.0
    _, events = classify_jumps(z, _contig(len(z)))
    assert shape_counts(events) == {SPIKE: 1, SUSTAINED: 0, REFIT: 1, EDGE: 0}
