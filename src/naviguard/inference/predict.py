"""naviguard.inference.predict — evaluation, forecasting, anomaly scoring, plotting.

evaluate_split()/forecast()/detect_anomalies() are pure functions returning
plain dicts (no matplotlib import) so the CLI and the FastAPI service share
one code path. Sequences are rebuilt in memory from the telemetry CSV with
the *persisted* scaler, so inference never depends on stale .npy files and
always matches the scaler the model was trained with. Results are cached per
(artifacts, telemetry file version), so repeated API calls are cheap.
"""

import os
import threading

import numpy as np

from naviguard import baselines
from naviguard.anomaly import detect as anomaly
from naviguard.config import (
    ANOMALY_Z_THRESHOLD, FEATURES, HORIZON, MAE_TARGET_NS, N_FEATURES, PLOT_PATH,
    SEQ_LEN, TARGET_IDX, TELEMETRY_CSV,
)
from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.preprocessing.sequences import (
    SequenceSet, build_sequences, load_telemetry, satellite_frames,
)

_NS = 1e9


def inverse_transform_target(scaler, scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform a 1-D or 2-D array of scaled clock_bias_s values
    back to physical units, using the dummy-column trick (the scaler was
    fit on all N_FEATURES columns jointly)."""
    shape = scaled.shape
    flat = scaled.reshape(-1)
    dummy = np.zeros((len(flat), N_FEATURES))
    dummy[:, TARGET_IDX] = flat
    return scaler.inverse_transform(dummy)[:, TARGET_IDX].reshape(shape)


# ── Per-(artifacts, telemetry) memoisation ────────────────────────────────────
_lock = threading.Lock()
_seq_cache: dict = {}
_eval_cache: dict = {}   # (id(artifacts), split, id(seqs)) -> (result, artifacts, seqs)
_MAX_CACHED = 8


def _file_version(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _get_sequences(artifacts: Artifacts) -> SequenceSet:
    seq_len = artifacts.meta.get("seq_len", SEQ_LEN)
    horizon = artifacts.meta.get("horizon", HORIZON)
    key = (id(artifacts), TELEMETRY_CSV, _file_version(TELEMETRY_CSV), seq_len, horizon)
    with _lock:
        hit = _seq_cache.get("k") == key
        if hit:
            return _seq_cache["v"]
    seqs = build_sequences(load_telemetry(TELEMETRY_CSV), artifacts.scaler, seq_len, horizon)
    with _lock:
        # Hold the artifacts object so its id cannot be recycled while cached.
        _seq_cache.update(k=key, v=seqs, ref=artifacts)
    return seqs


def score_split(artifacts: Artifacts, seqs: SequenceSet, split: str) -> dict:
    """Model metrics + raw series (nanoseconds) for one split of a SequenceSet."""
    X, y = seqs.part(split)
    y_pred = artifacts.model.predict(X, verbose=0)
    actual_ns = inverse_transform_target(artifacts.scaler, y) * _NS
    predicted_ns = inverse_transform_target(artifacts.scaler, np.asarray(y_pred)) * _NS
    residual_ns = actual_ns - predicted_ns
    return {
        "split": split,
        "n_test_samples": int(len(X)),
        "horizon": int(y.shape[1]),
        "target_ns": MAE_TARGET_NS,
        "mae_ns": np.mean(np.abs(residual_ns), axis=0).tolist(),
        "rmse_ns": np.sqrt(np.mean(residual_ns ** 2, axis=0)).tolist(),
        "actual_ns": actual_ns.tolist(),
        "predicted_ns": predicted_ns.tolist(),
        "residual_ns": residual_ns.tolist(),
    }


def _baseline_metrics(artifacts: Artifacts, seqs: SequenceSet, split: str) -> dict:
    X, y = seqs.part(split)
    Xtr, ytr = seqs.part("train")
    horizon = y.shape[1]
    ridge = baselines.fit_ridge(Xtr, ytr)
    preds = {
        "persistence": baselines.persistence(X, horizon),
        "linear_extrapolation": baselines.linear_extrapolation(X, horizon),
        "ridge": baselines.predict_ridge(ridge, X),
    }
    actual_ns = inverse_transform_target(artifacts.scaler, y) * _NS
    out = {}
    for name, p in preds.items():
        res = actual_ns - inverse_transform_target(artifacts.scaler, p) * _NS
        out[name] = {
            "mae_ns": np.mean(np.abs(res), axis=0).tolist(),
            "rmse_ns": np.sqrt(np.mean(res ** 2, axis=0)).tolist(),
        }
    return out


def evaluate_split(artifacts: Artifacts | None = None, split: str = "test") -> dict:
    """Evaluate on a held-out split (default: the untouched test split) and
    compare against classical baselines (persistence, linear extrapolation,
    ridge). `skill_vs_persistence` is 1 - MAE_model/MAE_persistence at step 1
    (>0 means the model beats naive persistence)."""
    artifacts = artifacts or get_artifacts()
    seqs = _get_sequences(artifacts)
    key = (id(artifacts), split, id(seqs))
    with _lock:
        hit = _eval_cache.get(key)
        if hit is not None:
            return hit[0]

    result = score_split(artifacts, seqs, split)
    result["baselines"] = _baseline_metrics(artifacts, seqs, split)
    persist1 = result["baselines"]["persistence"]["mae_ns"][0]
    result["skill_vs_persistence"] = (
        float(1.0 - result["mae_ns"][0] / persist1) if persist1 > 0 else None
    )
    result["pass_step1"] = bool(result["mae_ns"][0] <= MAE_TARGET_NS)
    with _lock:
        if len(_eval_cache) >= _MAX_CACHED:
            _eval_cache.clear()
        # Hold artifacts/seqs so their ids cannot be recycled while cached.
        _eval_cache[key] = (result, artifacts, seqs)
    return result


def evaluate_on_test(artifacts: Artifacts | None = None) -> dict:
    return evaluate_split(artifacts, "test")


def detect_anomalies(artifacts: Artifacts | None = None, z_threshold: float | None = None) -> dict:
    """Flag test-split windows whose step-1 residual is an outlier relative to
    the (nominal) validation-split residuals. Returns the detector summary."""
    artifacts = artifacts or get_artifacts()
    seqs = _get_sequences(artifacts)
    key = (id(artifacts), "val_calibration", id(seqs))
    with _lock:
        hit = _eval_cache.get(key)
    if hit is not None:
        center, scale = hit[0]
    else:
        val = score_split(artifacts, seqs, "val")
        center, scale = anomaly.calibrate(np.array(val["residual_ns"])[:, 0])
        with _lock:
            _eval_cache[key] = ((center, scale), artifacts, seqs)
    test = evaluate_split(artifacts, "test")
    return anomaly.detect(
        np.array(test["residual_ns"])[:, 0], center, scale,
        z_threshold if z_threshold is not None else ANOMALY_Z_THRESHOLD,
    )


def _latest_window(seq_len: int, satellite_id: int | None) -> np.ndarray:
    df = load_telemetry(TELEMETRY_CSV)
    frames = dict(satellite_frames(df))
    if satellite_id is None:
        satellite_id = next(iter(frames))
    if satellite_id not in frames:
        raise ValueError(f"unknown satellite_id {satellite_id}; available: {sorted(frames)}")
    g = frames[satellite_id]
    if len(g) < seq_len:
        raise ValueError(f"satellite {satellite_id} has only {len(g)} rows; need {seq_len}")
    return g[FEATURES].to_numpy(dtype=np.float64)[-seq_len:]


def forecast(
    window: list[list[float]] | None = None,
    artifacts: Artifacts | None = None,
    satellite_id: int | None = None,
) -> dict:
    """Forecast `horizon` steps ahead from a seq_len-length window of raw
    [clock_bias_s, clock_drift_s_per_s, ephemeris_error_m] rows. If no
    window is given, uses the latest seq_len rows of the telemetry CSV
    (for `satellite_id`, default the first satellite). Raises ValueError on
    malformed input."""
    artifacts = artifacts or get_artifacts()
    model, scaler = artifacts.model, artifacts.scaler
    seq_len = artifacts.meta.get("seq_len", SEQ_LEN)

    if window is None:
        window = _latest_window(seq_len, satellite_id)
    else:
        window = np.asarray(window, dtype=np.float64)
        if window.shape != (seq_len, N_FEATURES):
            raise ValueError(f"window must have shape ({seq_len}, {N_FEATURES}), got {window.shape}")
        if not np.isfinite(window).all():
            raise ValueError("window contains NaN or infinite values")

    X = scaler.transform(window).reshape(1, seq_len, N_FEATURES)
    y_pred_scaled = np.asarray(model.predict(X, verbose=0))
    predicted_ns = inverse_transform_target(scaler, y_pred_scaled)[0] * _NS

    return {
        "horizon": int(y_pred_scaled.shape[1]),
        "predicted_clock_bias_ns": predicted_ns.tolist(),
    }


def save_prediction_plot(eval_result: dict, out_path: str = PLOT_PATH) -> str:
    """Render the step-1-ahead actual-vs-predicted + residuals chart, using an
    already-computed evaluate_split() result so plotting never duplicates the
    inference logic. Persistence is overlaid for context."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    actual = np.array(eval_result["actual_ns"])[:, 0] / 1e3     # ns -> us, step 1
    predicted = np.array(eval_result["predicted_ns"])[:, 0] / 1e3
    residual_ns = np.array(eval_result["residual_ns"])[:, 0]
    mae_ns = eval_result["mae_ns"][0]
    rmse_ns = eval_result["rmse_ns"][0]
    persist = eval_result.get("baselines", {}).get("persistence", {}).get("mae_ns", [None])[0]

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
    note = f"MAE = {mae_ns:.4f} ns  |  RMSE = {rmse_ns:.4f} ns  (step 1 / {eval_result['horizon']})"
    if persist is not None:
        note += f"  |  persistence MAE = {persist:.4f} ns"
    ax1.annotate(
        note, xy=(0.02, 0.92), xycoords="axes fraction", fontsize=9.5, color="#b71c1c",
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
