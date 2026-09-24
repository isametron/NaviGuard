"""Tests for scaling, chronological splitting, and windowing."""

import numpy as np
import pandas as pd
import pytest

from naviguard.config import FEATURES, TARGET_IDX
from naviguard.data.generate import generate_dataset
from naviguard.errors import TelemetryNotFoundError
from naviguard.preprocessing.sequences import (
    TEST, TRAIN, VAL, build_sequences, fit_scaler, load_sequences, load_telemetry,
    save_sequences, split_bounds,
)


def test_shapes_and_split_counts(synthetic_df):
    scaler = fit_scaler(synthetic_df, scaler_path=None)
    seqs = build_sequences(synthetic_df, scaler, seq_len=5, horizon=3)
    assert seqs.X.shape[1:] == (5, 3)
    assert seqs.y.shape[1:] == (3,)
    assert len(seqs.X) == len(seqs.y) == len(seqs.split)
    assert all(seqs.count(s) > 0 for s in ("train", "val", "test"))


def test_no_target_leakage_across_splits():
    """Every window's target rows must lie entirely inside its own split's rows."""
    n, seq_len, horizon = 200, 5, 3
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in FEATURES})
    df["timestamp_s"] = np.arange(n) * 900
    scaler = fit_scaler(df, scaler_path=None)
    seqs = build_sequences(df, scaler, seq_len, horizon)
    n_train, n_val = split_bounds(n)

    # Bias column equals the row index -> recover target row indices from scaled values.
    raw_min, raw_max = scaler.data_min_[TARGET_IDX], scaler.data_max_[TARGET_IDX]
    rows = np.rint(seqs.y * (raw_max - raw_min) + raw_min).astype(int)
    # scaler was fit on train rows only, so val/test targets scale beyond [0, 1] — still invertible.
    for split_id, lo, hi in ((TRAIN, 0, n_train), (VAL, n_train, n_val), (TEST, n_val, n)):
        r = rows[seqs.split == split_id]
        assert r.min() >= lo and r.max() < hi


def test_scaler_fit_on_training_rows_only():
    n = 100
    df = pd.DataFrame({c: np.zeros(n) for c in FEATURES})
    df["clock_bias_s"] = np.r_[np.linspace(0, 1, 70), np.full(30, 1000.0)]  # huge values after train
    scaler = fit_scaler(df, scaler_path=None)
    assert scaler.data_max_[TARGET_IDX] <= 1.0


def test_windows_do_not_cross_satellites(tmp_path):
    df = generate_dataset(n_samples=120, n_satellites=3, out_path=str(tmp_path / "t.csv"))
    scaler = fit_scaler(df, scaler_path=None)
    seqs = build_sequences(df, scaler, seq_len=10, horizon=4)
    assert set(np.unique(seqs.satellite)) == {0, 1, 2}
    per_sat = [(seqs.satellite == s).sum() for s in range(3)]
    assert len(set(per_sat)) == 1            # identical lengths -> identical window counts


def test_too_few_samples_raises():
    df = pd.DataFrame({c: np.arange(30, dtype=float) for c in FEATURES})
    scaler = fit_scaler(df, scaler_path=None)
    with pytest.raises(ValueError, match="Not enough samples"):
        build_sequences(df, scaler, seq_len=20, horizon=6)


def test_save_load_roundtrip(tmp_path, synthetic_df):
    scaler = fit_scaler(synthetic_df, scaler_path=None)
    seqs = build_sequences(synthetic_df, scaler, 5, 2)
    path = str(tmp_path / "seq.npz")
    save_sequences(seqs, path)
    loaded = load_sequences(path)
    assert np.array_equal(loaded.X, seqs.X) and np.array_equal(loaded.split, seqs.split)


def test_load_telemetry_missing_file(tmp_path):
    with pytest.raises(TelemetryNotFoundError):
        load_telemetry(str(tmp_path / "nope.csv"))


def test_load_telemetry_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"clock_bias_s": [1, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_telemetry(str(p))
