"""
preprocess.py  ·  NaviGuard — Data & Preprocessing Layer
─────────────────────────────────────────────────────────
NavIC/GNSS Source : data/satellite_telemetry.csv
Outputs           : data/X_seq.npy  |  data/y_seq.npy  |  models/scaler.pkl
SEQ_LEN           : 20 steps  (5 hours @ 15-min intervals)
Split             : 80 / 20  (chronological, no shuffle)
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = "data/satellite_telemetry.csv"
SEQ_LEN     = 20
FEATURES    = ["clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m"]
TARGET_COL  = "clock_bias_s"
TARGET_IDX  = FEATURES.index(TARGET_COL)
SCALER_PATH = "models/scaler.pkl"
RANDOM_SEED = 42

def load_telemetry(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[preprocess] ERROR: {path} not found.")
        sys.exit(1)
    df = pd.read_csv(path)
    assert all(c in df.columns for c in FEATURES),         f"[preprocess] Missing expected columns in {path}"
    print(f"[preprocess] NavIC/GNSS source loaded: {len(df)} rows  |  "
          f"{len(FEATURES)} features")
    return df

def fit_and_scale(df: pd.DataFrame):
    """Fit scaler on training portion only — prevents data leakage."""
    n_train = int(len(df) * 0.8)
    scaler  = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[FEATURES].iloc[:n_train])
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[preprocess] MinMaxScaler fitted on first {n_train} rows  "
          f"→  saved: {SCALER_PATH}")
    return df_scaled, scaler

def build_sequences(df_scaled: pd.DataFrame):
    vals = df_scaled[FEATURES].values
    X, y = [], []
    for i in range(SEQ_LEN, len(vals)):
        X.append(vals[i - SEQ_LEN:i])          # shape (SEQ_LEN, 3)
        y.append(vals[i, TARGET_IDX])           # clock_bias_s at step i
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

if __name__ == "__main__":
    df              = load_telemetry(DATA_PATH)
    df_scaled, _    = fit_and_scale(df)
    X, y            = build_sequences(df_scaled)
    os.makedirs("data", exist_ok=True)
    np.save("data/X_seq.npy", X)
    np.save("data/y_seq.npy", y)
    print(f"[preprocess] Sequences  →  X:{X.shape}  y:{y.shape}")
    print(f"[preprocess] Saved: data/X_seq.npy  |  data/y_seq.npy")
    print("[preprocess] ✓ Complete")
