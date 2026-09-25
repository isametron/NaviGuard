"""naviguard.preprocessing.sequences — scaling, splitting, and windowing.

Handles one or many satellites. Every satellite series is split
chronologically (TRAIN_FRAC / VAL_FRAC / remainder) and windows never cross
satellite boundaries. A window is assigned to a split by where its *target*
lies; windows whose target straddles a split boundary are dropped, so no
training target ever overlaps validation/test rows. Look-back inputs of
val/test windows may reach into earlier rows, which is normal (past only).

The scaler is fit on the training rows of every satellite only.
"""

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from naviguard.config import (
    FEATURES, HORIZON, SCALER_PATH, SEQ_LEN, SEQUENCES_PATH, TARGET_IDX,
    TELEMETRY_CSV, TRAIN_FRAC, VAL_FRAC,
)
from naviguard.errors import TelemetryNotFoundError

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_IDS = {"train": TRAIN, "val": VAL, "test": TEST}


@dataclass
class SequenceSet:
    X: np.ndarray           # (n, seq_len, n_features), scaled
    y: np.ndarray           # (n, horizon), scaled clock bias
    split: np.ndarray       # (n,) int8 in {TRAIN, VAL, TEST}
    satellite: np.ndarray   # (n,) int32 satellite id (0 for single-satellite data)

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        mask = self.split == SPLIT_IDS[name]
        return self.X[mask], self.y[mask]

    def count(self, name: str) -> int:
        return int((self.split == SPLIT_IDS[name]).sum())


def load_telemetry(path: str = TELEMETRY_CSV, required: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise TelemetryNotFoundError(
            f"Telemetry CSV not found: {path}. Run `naviguard generate` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in (FEATURES if required is None else required) if c not in df.columns]
    if missing:
        raise ValueError(f"Telemetry CSV {path} is missing columns: {missing}")
    return df


def split_bounds(n: int) -> tuple[int, int]:
    """Row indices (train_end, val_end) for a series of n rows."""
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def satellite_frames(df: pd.DataFrame) -> list[tuple[int, pd.DataFrame]]:
    """Split telemetry into per-satellite, time-ordered frames."""
    order = "timestamp_s" if "timestamp_s" in df.columns else None
    if "satellite_id" not in df.columns:
        frames = [(0, df)]
    else:
        frames = [(int(k), g) for k, g in df.groupby("satellite_id", sort=True)]
    if order:
        frames = [(k, g.sort_values(order, kind="stable")) for k, g in frames]
    return [(k, g.reset_index(drop=True)) for k, g in frames]


def fit_scaler(df: pd.DataFrame, scaler_path: str | None = SCALER_PATH) -> MinMaxScaler:
    """Fit the scaler on training rows only (all satellites pooled) — no leakage."""
    train_rows = []
    for _, g in satellite_frames(df):
        n_train, _ = split_bounds(len(g))
        train_rows.append(g[FEATURES].to_numpy(dtype=np.float64)[:n_train])
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(np.vstack(train_rows))
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    return scaler


def build_sequences(
    df: pd.DataFrame, scaler, seq_len: int = SEQ_LEN, horizon: int = HORIZON
) -> SequenceSet:
    """Window every satellite's scaled series and label each window's split.

    X[i] = scaled[i-seq_len : i, :]              (seq_len, n_features)
    y[i] = scaled_clock_bias[i : i+horizon]      (horizon,)
    """
    Xs, ys, splits, sats = [], [], [], []
    for sat_id, g in satellite_frames(df):
        vals = scaler.transform(g[FEATURES].to_numpy(dtype=np.float64))
        n = len(vals)
        n_train, n_val = split_bounds(n)
        for i in range(seq_len, n - horizon + 1):
            end = i + horizon
            if end <= n_train:
                s = TRAIN
            elif i >= n_train and end <= n_val:
                s = VAL
            elif i >= n_val:
                s = TEST
            else:
                continue  # target straddles a split boundary
            Xs.append(vals[i - seq_len:i])
            ys.append(vals[i:end, TARGET_IDX])
            splits.append(s)
            sats.append(sat_id)

    seqs = SequenceSet(
        X=np.asarray(Xs, dtype=np.float32).reshape(-1, seq_len, len(FEATURES)),
        y=np.asarray(ys, dtype=np.float32).reshape(-1, horizon),
        split=np.asarray(splits, dtype=np.int8),
        satellite=np.asarray(sats, dtype=np.int32),
    )
    empty = [name for name in SPLIT_IDS if seqs.count(name) == 0]
    if empty:
        raise ValueError(
            f"Not enough samples per satellite for seq_len={seq_len}, horizon={horizon}: "
            f"no windows in split(s) {empty}. Generate more samples or shrink seq_len/horizon."
        )
    return seqs


def save_sequences(seqs: SequenceSet, path: str = SEQUENCES_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, X=seqs.X, y=seqs.y, split=seqs.split, satellite=seqs.satellite)


def load_sequences(path: str = SEQUENCES_PATH) -> SequenceSet:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sequences not found: {path}. Run `naviguard preprocess` first.")
    with np.load(path) as z:
        return SequenceSet(X=z["X"], y=z["y"], split=z["split"], satellite=z["satellite"])
