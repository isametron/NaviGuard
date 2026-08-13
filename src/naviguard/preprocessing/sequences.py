"""naviguard.preprocessing.sequences — scaling + sliding-window sequence construction.

Assumes a single chronological telemetry series (the default single-satellite
generate_dataset() output). Multi-satellite sequence construction is not
handled here — out of scope for today's rebuild.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from naviguard.config import FEATURES, SCALER_PATH, SEQ_LEN, HORIZON, TARGET_IDX, TELEMETRY_CSV


def load_telemetry(path: str = TELEMETRY_CSV) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[preprocess] ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_csv(path)
    assert all(c in df.columns for c in FEATURES), \
        f"[preprocess] Missing expected columns in {path}"
    print(f"[preprocess] Telemetry loaded: {len(df)} rows  |  {len(FEATURES)} features")
    return df


def fit_and_scale(df: pd.DataFrame, scaler_path: str = SCALER_PATH):
    """Fit scaler on the training portion only — prevents data leakage."""
    n_train = int(len(df) * 0.8)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[FEATURES].iloc[:n_train])
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"[preprocess] MinMaxScaler fitted on first {n_train} rows  →  saved: {scaler_path}")
    return df_scaled, scaler


def build_sequences(df_scaled: pd.DataFrame, seq_len: int = SEQ_LEN, horizon: int = HORIZON):
    """Build sliding-window sequences with a multi-step target.

    X[i] = scaled_values[i-seq_len : i, :]                 shape (seq_len, n_features)
    y[i] = scaled_clock_bias[i : i+horizon]                 shape (horizon,)

    horizon=1 reproduces the original single-step-ahead shape (n, 1).
    """
    vals = df_scaled[FEATURES].values
    X, y = [], []
    for i in range(seq_len, len(vals) - horizon + 1):
        X.append(vals[i - seq_len:i])
        y.append(vals[i:i + horizon, TARGET_IDX])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
