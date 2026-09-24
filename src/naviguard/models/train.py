"""naviguard.models.train — train the attention-LSTM and persist checkpoint + metadata.

Trains on the train split, uses the validation split for early stopping /
LR decay / checkpoint selection, and only *after* training scores the
untouched test split once (recorded in model_meta.json alongside the
persistence baseline for context).
"""

import hashlib
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from naviguard import baselines
from naviguard.config import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, FEATURES, HORIZON,
    MODEL_PATH, META_PATH, N_FEATURES, RANDOM_SEED, SCALER_PATH, SEQ_LEN,
    SEQUENCES_PATH, TARGET_COL, TELEMETRY_CSV, ModelHParams,
)
from naviguard.inference.artifacts import Artifacts
from naviguard.inference.predict import inverse_transform_target, score_split
from naviguard.models.lstm_attention import build_model
from naviguard.preprocessing.sequences import load_sequences


def _seed_everything(seed: int = RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train(
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    hparams: ModelHParams | None = None,
    sequences_path: str = SEQUENCES_PATH,
    scaler_path: str = SCALER_PATH,
    model_path: str = MODEL_PATH,
    meta_path: str = META_PATH,
    telemetry_path: str = TELEMETRY_CSV,
    verbose: int = 1,
) -> dict:
    hparams = hparams or ModelHParams()
    _seed_everything()

    seqs = load_sequences(sequences_path)
    if seqs.X.shape[1] != seq_len or seqs.y.shape[1] != horizon:
        raise ValueError(
            f"Sequences at {sequences_path} have seq_len={seqs.X.shape[1]}, horizon={seqs.y.shape[1]} "
            f"but training was asked for seq_len={seq_len}, horizon={horizon}. "
            "Re-run `naviguard preprocess` with matching values."
        )
    X_train, y_train = seqs.part("train")
    X_val, y_val = seqs.part("val")
    print(f"[train] Loaded sequences  →  X:{seqs.X.shape}  y:{seqs.y.shape}")
    print(f"[train] Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test (held out): {seqs.count('test')}")

    model = build_model(seq_len, N_FEATURES, horizon, hparams)
    if verbose:
        model.summary()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=verbose),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=verbose),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience // 2),
                          min_lr=1e-6, verbose=verbose),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    best_val_loss = min(history.history["val_loss"])
    epochs_run = len(history.history["loss"])
    print(f"\n[train] Epochs run    : {epochs_run}")
    print(f"[train] Best val_loss  : {best_val_loss:.6f}")
    print(f"[train] Model saved    : {model_path}")

    # One-shot score of the held-out test split with the restored best weights.
    scaler = joblib.load(scaler_path)
    test = score_split(Artifacts(model=model, scaler=scaler, meta={}), seqs, "test")
    X_test, y_test = seqs.part("test")
    persist_res = (inverse_transform_target(scaler, y_test)
                   - inverse_transform_target(scaler, baselines.persistence(X_test, horizon))) * 1e9
    persist_mae = np.mean(np.abs(persist_res), axis=0)
    print(f"[train] Test MAE step 1: {test['mae_ns'][0]:.3f} ns  (persistence: {persist_mae[0]:.3f} ns)")

    meta = {
        "seq_len": seq_len,
        "horizon": horizon,
        "features": FEATURES,
        "target": TARGET_COL,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": float(best_val_loss),
        "epochs_run": epochs_run,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_mae_ns": test["mae_ns"],
        "persistence_mae_ns": persist_mae.tolist(),
        "hparams": {**asdict(hparams), "lstm_units": list(hparams.lstm_units)},
        "telemetry_sha256": _sha256(telemetry_path),
        "seed": RANDOM_SEED,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved : {meta_path}")
    print("[train] ✓ Complete")
    return meta


if __name__ == "__main__":
    train()
