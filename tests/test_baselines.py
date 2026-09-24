"""Tests for the classical baseline forecasters."""

import numpy as np

from naviguard import baselines
from naviguard.config import N_FEATURES, TARGET_IDX


def _windows(fn, n=4, seq_len=10):
    X = np.zeros((n, seq_len, N_FEATURES))
    t = np.arange(seq_len)
    X[:, :, TARGET_IDX] = fn(t)[None, :]
    return X


def test_persistence_repeats_last_value():
    X = _windows(lambda t: t * 1.0)
    p = baselines.persistence(X, 3)
    assert p.shape == (4, 3) and np.allclose(p, 9.0)


def test_linear_extrapolation_is_exact_on_a_line():
    X = _windows(lambda t: 2.0 * t + 1.0)
    p = baselines.linear_extrapolation(X, 4)
    expected = 2.0 * (9 + np.arange(1, 5)) + 1.0
    assert np.allclose(p, expected[None, :])


def test_ridge_learns_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6, N_FEATURES))
    y = np.stack([X[:, -1, TARGET_IDX], 2 * X[:, -2, TARGET_IDX]], axis=1)
    model = baselines.fit_ridge(X, y, alpha=1e-8)
    assert np.allclose(baselines.predict_ridge(model, X), y, atol=1e-4)
