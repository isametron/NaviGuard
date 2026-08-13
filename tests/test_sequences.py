"""Tests for naviguard.preprocessing.sequences."""

import numpy as np

from naviguard.preprocessing.sequences import build_sequences, fit_and_scale
from naviguard.config import FEATURES, TARGET_IDX


def test_fit_and_scale_uses_only_first_80_percent(synthetic_df, tmp_path):
    scaler_path = str(tmp_path / "scaler.pkl")
    df_scaled, scaler = fit_and_scale(synthetic_df, scaler_path=scaler_path)

    n_train = int(len(synthetic_df) * 0.8)
    # The scaler's observed data_min_/data_max_ must match only the train slice,
    # not the full dataframe (this is the data-leakage guard).
    train_min = synthetic_df[FEATURES].iloc[:n_train].min().values
    train_max = synthetic_df[FEATURES].iloc[:n_train].max().values
    assert np.allclose(scaler.data_min_, train_min)
    assert np.allclose(scaler.data_max_, train_max)
    assert df_scaled.shape == synthetic_df.shape


def test_build_sequences_shapes(synthetic_df):
    # build_sequences only needs the FEATURE columns; using the raw df directly
    # is fine since exact scaling doesn't matter for a shape test.
    seq_len, horizon = 10, 4
    X, y = build_sequences(synthetic_df, seq_len=seq_len, horizon=horizon)

    expected_n = len(synthetic_df) - seq_len - horizon + 1
    assert X.shape == (expected_n, seq_len, len(FEATURES))
    assert y.shape == (expected_n, horizon)


def test_build_sequences_target_matches_source_column(synthetic_df):
    seq_len, horizon = 5, 3
    X, y = build_sequences(synthetic_df, seq_len=seq_len, horizon=horizon)

    vals = synthetic_df[FEATURES].values
    # First window's target should equal the next `horizon` target-column values.
    assert np.allclose(y[0], vals[seq_len:seq_len + horizon, TARGET_IDX])
    assert np.allclose(X[0], vals[0:seq_len])


def test_build_sequences_horizon_one_matches_original_single_step_shape(synthetic_df):
    X, y = build_sequences(synthetic_df, seq_len=20, horizon=1)
    assert y.shape == (len(synthetic_df) - 20, 1)
