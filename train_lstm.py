import os, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ReduceLROnPlateau)
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN    = 20
N_FEATURES = 3
EPOCHS     = 50
BATCH_SIZE = 16
MODEL_PATH = "models/lstm_satellite.keras"

def build_model(seq_len: int, n_features: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ], name="NaviGuard_LSTM")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="mse")
    return model

if __name__ == "__main__":
    X = np.load("data/X_seq.npy")
    y = np.load("data/y_seq.npy")
    print(f"[train] Loaded sequences  →  X:{X.shape}  y:{y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False)
    print(f"[train] Train: {X_train.shape[0]} sequences  "
          f"|  Val: {X_val.shape[0]} sequences")

    model = build_model(SEQ_LEN, N_FEATURES)
    model.summary()

    os.makedirs("models", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_loss",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    best_val_loss = min(history.history["val_loss"])
    epochs_run    = len(history.history["loss"])
    print(f"\n[train] Epochs run    : {epochs_run}")
    print(f"[train] Best val_loss  : {best_val_loss:.6f}")
    print(f"[train] Model saved    : {MODEL_PATH}")
    print("[train] ✓ Complete")
