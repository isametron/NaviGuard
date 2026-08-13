import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
from tensorflow.keras.models import load_model

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/lstm_satellite.keras"
SCALER_PATH = "models/scaler.pkl"
OUT_PATH    = "outputs/prediction_plot.png"
N_FEATURES  = 3

def check_files(*paths):
    for p in paths:
        if not os.path.exists(p):
            print(f"[predict] ERROR: {p} not found. Run preprocess.py and train_lstm.py first.")
            sys.exit(1)

def inverse_clock_bias(scaler, scaled_vals: np.ndarray) -> np.ndarray:
    """Reconstruct clock_bias_s (feature 0) from scaled 1-D array."""
    dummy        = np.zeros((len(scaled_vals), N_FEATURES))
    dummy[:, 0]  = scaled_vals
    return scaler.inverse_transform(dummy)[:, 0]

def run():
    check_files(MODEL_PATH, SCALER_PATH, "data/X_seq.npy", "data/y_seq.npy")

    X      = np.load("data/X_seq.npy")
    y      = np.load("data/y_seq.npy")
    scaler = joblib.load(SCALER_PATH)
    model  = load_model(MODEL_PATH)
    print(f"[predict] Model loaded  →  {MODEL_PATH}")

    split        = int(len(X) * 0.8)
    X_test       = X[split:]
    y_test       = y[split:]
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    actual    = inverse_clock_bias(scaler, y_test)
    predicted = inverse_clock_bias(scaler, y_pred_scaled)
    residuals = actual - predicted

    mae_s  = float(np.mean(np.abs(residuals)))
    mae_ns = mae_s * 1e9
    mae_us = mae_s * 1e6
    rmse_s = float(np.sqrt(np.mean(residuals ** 2)))

    print(f"[predict] Test samples  : {len(y_test)}")
    print(f"[predict] MAE           : {mae_s:.4e} s  |  {mae_ns:.4f} ns  |  {mae_us:.6f} μs")
    print(f"[predict] RMSE          : {rmse_s:.4e} s")
    status = "✓ PASS" if mae_ns <= 50 else "✗ EXCEEDS TARGET"
    print(f"[predict] Target ≤ 50 ns: {status}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    fig = plt.figure(figsize=(14, 8), facecolor="#ffffff")
    gs  = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50,
                             height_ratios=[2, 1])

    # Panel 1 — Actual vs Predicted
    ax1 = fig.add_subplot(gs[0])
    t   = np.arange(len(actual))
    ax1.plot(t, actual * 1e6,    color="#1565c0", lw=1.8, label="Actual Clock Bias")
    ax1.plot(t, predicted * 1e6, color="#f44336", lw=1.8, ls="--",
             label="LSTM Prediction", alpha=0.88)
    ax1.fill_between(t, actual * 1e6, predicted * 1e6,
                     alpha=0.08, color="#9c27b0")
    ax1.set_xlabel("Test Time Steps  (×15 min)", fontsize=10)
    ax1.set_ylabel("Clock Bias  (μs)", fontsize=10)
    ax1.set_title("NaviGuard — Satellite Clock Bias: Actual vs LSTM Prediction",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.annotate(
        f"MAE = {mae_ns:.4f} ns  |  RMSE = {rmse_s*1e9:.4f} ns",
        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9.5,
        color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5",
                  edgecolor="#f44336", alpha=0.90)
    )

    # Panel 2 — Residuals
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(t, residuals * 1e9, color="#9c27b0", alpha=0.6, width=0.9)
    ax2.axhline(0, color="#333333", lw=0.9, linestyle="--")
    ax2.set_xlabel("Test Time Steps  (×15 min)", fontsize=10)
    ax2.set_ylabel("Residual  (ns)", fontsize=10)
    ax2.set_title("Prediction Residuals  (Actual − Predicted)", fontsize=10)
    ax2.grid(alpha=0.2, linestyle="--")

    fig.text(0.5, 0.01,
             "NaviGuard  ·  NavIC/GNSS Clock Bias Prediction  ·  "
             "LSTM(64→32) + Dropout  ·  Dept. of AI & DS, BMSCE",
             ha="center", fontsize=7.5, color="#888888")

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"[predict] Plot saved    : {OUT_PATH}")
    print("[predict] ✓ Complete")

if __name__ == "__main__":
    run()
