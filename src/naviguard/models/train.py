"""naviguard.models.train — train the attention-LSTM and persist the checkpoint + metadata."""

import json
import os
import random
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from naviguard.config import (
    FEATURES, HORIZON, MODEL_PATH, META_PATH, N_FEATURES, RANDOM_SEED, SEQ_LEN,
    TARGET_COL, X_SEQ_PATH, Y_SEQ_PATH,
)
from naviguard.models.lstm_attention import build_model


def _seed_everything(seed: int = RANDOM_SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train(
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    epochs: int = 50,
    batch_size: int = 16,
    model_path: str = MODEL_PATH,
    meta_path: str = META_PATH,
) -> dict:
    _seed_everything()

    X = np.load(X_SEQ_PATH)
    y = np.load(Y_SEQ_PATH)
    print(f"[train] Loaded sequences  →  X:{X.shape}  y:{y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    print(f"[train] Train: {X_train.shape[0]} sequences  |  Val: {X_val.shape[0]} sequences")

    model = build_model(seq_len, N_FEATURES, horizon)
    model.summary()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    best_val_loss = min(history.history["val_loss"])
    epochs_run = len(history.history["loss"])
    print(f"\n[train] Epochs run    : {epochs_run}")
    print(f"[train] Best val_loss  : {best_val_loss:.6f}")
    print(f"[train] Model saved    : {model_path}")

    meta = {
        "seq_len": seq_len,
        "horizon": horizon,
        "features": FEATURES,
        "target": TARGET_COL,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": float(best_val_loss),
        "epochs_run": epochs_run,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadata saved : {meta_path}")
    print("[train] ✓ Complete")
    return meta


if __name__ == "__main__":
    train()
