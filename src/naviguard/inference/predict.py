"""naviguard.inference.predict — evaluation, forecasting, and plotting.

evaluate_on_test() and forecast() are pure functions returning plain dicts/
arrays (no matplotlib import needed) so they can be shared by both the CLI
and the FastAPI service without duplicating logic or paying for matplotlib
on the API's hot path. save_prediction_plot() is the opt-in convenience
wrapper that reproduces the original PNG output.
"""

import os

import numpy as np
import pandas as pd

from naviguard.config import (
    FEATURES, MAE_TARGET_NS, N_FEATURES, PLOT_PATH, SEQ_LEN, TARGET_IDX,
    TELEMETRY_CSV, X_SEQ_PATH, Y_SEQ_PATH,
)
from naviguard.inference.artifacts import Artifacts, get_artifacts


def inverse_transform_target(scaler, scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1-D or 2-D array of scaled clock_bias_s values
    back to physical units, using the dummy-column trick (the scaler was
    fit on all N_FEATURES columns jointly)."""
    shape = scaled.shape
    flat = scaled.reshape(-1)
    dummy = np.zeros((len(flat), N_FEATURES))
    dummy[:, TARGET_IDX] = flat
    return scaler.inverse_transform(dummy)[:, TARGET_IDX].reshape(shape)


def evaluate_on_test(artifacts: Artifacts = None) -> dict:
    """Run inference on the held-out (chronological, last 20%) test split
    and return MAE/RMSE per horizon step plus the raw actual/predicted/
    residual series, all in nanoseconds."""
    artifacts = artifacts or get_artifacts()
    model, scaler = artifacts.model, artifacts.scaler

    X = np.load(X_SEQ_PATH)
    y = np.load(Y_SEQ_PATH)
    split = int(len(X) * 0.8)
    X_test, y_test = X[split:], y[split:]

    y_pred_scaled = model.predict(X_test, verbose=0)
    horizon = y_test.shape[1]

    actual_ns = inverse_transform_target(scaler, y_test) * 1e9
    predicted_ns = inverse_transform_target(scaler, y_pred_scaled) * 1e9
    residual_ns = actual_ns - predicted_ns

    mae_per_step = np.mean(np.abs(residual_ns), axis=0).tolist()
    rmse_per_step = np.sqrt(np.mean(residual_ns ** 2, axis=0)).tolist()

    return {
        "n_test_samples": int(len(X_test)),
        "horizon": int(horizon),
        "target_ns": MAE_TARGET_NS,
        "mae_ns": mae_per_step,
        "rmse_ns": rmse_per_step,
        "pass_step1": bool(mae_per_step[0] <= MAE_TARGET_NS),
        "actual_ns": actual_ns.tolist(),
        "predicted_ns": predicted_ns.tolist(),
        "residual_ns": residual_ns.tolist(),
    }


def forecast(window: list[list[float]] | None = None, artifacts: Artifacts = None) -> dict:
    """Forecast `horizon` steps ahead from a seq_len-length window of raw
    [clock_bias_s, clock_drift_s_per_s, ephemeris_error_m] rows. If no
    window is given, uses the latest seq_len rows of the telemetry CSV."""
    artifacts = artifacts or get_artifacts()
    model, scaler = artifacts.model, artifacts.scaler
    seq_len = artifacts.meta.get("seq_len", SEQ_LEN)

    if window is None:
        df = pd.read_csv(TELEMETRY_CSV)
        window = df[FEATURES].values[-seq_len:]
    else:
        window = np.asarray(window, dtype=np.float32)
        if window.shape != (seq_len, N_FEATURES):
            raise ValueError(f"window must have shape ({seq_len}, {N_FEATURES}), got {window.shape}")

    scaled_window = scaler.transform(window)
    X = scaled_window.reshape(1, seq_len, N_FEATURES)
    y_pred_scaled = model.predict(X, verbose=0)
    predicted_ns = inverse_transform_target(scaler, y_pred_scaled)[0] * 1e9

    return {
        "horizon": int(y_pred_scaled.shape[1]),
        "predicted_clock_bias_ns": predicted_ns.tolist(),
    }


def save_prediction_plot(eval_result: dict, out_path: str = PLOT_PATH) -> str:
    """Render the step-1-ahead actual-vs-predicted + residuals chart (the
    original two-panel PNG), using an already-computed evaluate_on_test()
    result so plotting never duplicates the inference logic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    actual = np.array(eval_result["actual_ns"])[:, 0] / 1e3     # ns -> us, step 1
    predicted = np.array(eval_result["predicted_ns"])[:, 0] / 1e3
    residual_ns = np.array(eval_result["residual_ns"])[:, 0]
    mae_ns = eval_result["mae_ns"][0]
    rmse_ns = eval_result["rmse_ns"][0]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(14, 8), facecolor="#ffffff")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.50, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0])
    t = np.arange(len(actual))
    ax1.plot(t, actual, color="#1565c0", lw=1.8, label="Actual Clock Bias")
    ax1.plot(t, predicted, color="#f44336", lw=1.8, ls="--", label="Attention-LSTM Prediction (step 1)", alpha=0.88)
    ax1.fill_between(t, actual, predicted, alpha=0.08, color="#9c27b0")
    ax1.set_xlabel("Test Time Steps  (×15 min)", fontsize=10)
    ax1.set_ylabel("Clock Bias  (μs)", fontsize=10)
    ax1.set_title("NaviGuard — Satellite Clock Bias: Actual vs Attention-LSTM Prediction",
                  fontsize=12, fontweight="bold", pad=12)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.annotate(
        f"MAE = {mae_ns:.4f} ns  |  RMSE = {rmse_ns:.4f} ns  (step 1 / {eval_result['horizon']})",
        xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9.5, color="#b71c1c",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff5f5", edgecolor="#f44336", alpha=0.90),
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.bar(t, residual_ns, color="#9c27b0", alpha=0.6, width=0.9)
    ax2.axhline(0, color="#333333", lw=0.9, linestyle="--")
    ax2.set_xlabel("Test Time Steps  (×15 min)", fontsize=10)
    ax2.set_ylabel("Residual  (ns)", fontsize=10)
    ax2.set_title("Prediction Residuals (step 1) — Actual − Predicted", fontsize=10)
    ax2.grid(alpha=0.2, linestyle="--")

    fig.text(0.5, 0.01,
              "NaviGuard  ·  NavIC/GNSS Clock Bias Prediction  ·  "
              "Attention-LSTM(64→32)  ·  Dept. of AI & DS, BMSCE",
              ha="center", fontsize=7.5, color="#888888")

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    return out_path
