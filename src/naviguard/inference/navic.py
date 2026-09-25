"""naviguard.inference.navic — the real-data ("navic") profile: train, persist, evaluate, forecast.

Uses level-free windows over real NavIC broadcast clock series (see benchmark/windows.py), pooled
across satellites with a chronological, purged 70/15/15 split per satellite. Three candidates are
trained (broadcast-drift extrapolation, ridge, attention-LSTM) and the one with the lowest
*validation* MAE is served; test metrics for all candidates are recorded but never used to choose.

Everything returns the same plain-dict shapes as inference/predict.py so the API and CLI work
unchanged (`NAVIGUARD_PROFILE=navic`).
"""

import hashlib
import json
import os
import threading
from dataclasses import asdict
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from naviguard.anomaly import detect as anomaly
from naviguard.benchmark.models import ARIMA, BroadcastDrift, LinearExtrap, Neural, RidgeDirect, make_model
from naviguard.benchmark.windows import Windows, load_series, make_windows, relative_window, split_windows
from naviguard.config import (
    ANOMALY_Z_THRESHOLD, HORIZON, MAE_TARGET_NS, NAVIC_DIR, NAVIC_TELEMETRY_CSV, SEQ_LEN, ModelHParams,
)
from naviguard.errors import TelemetryNotFoundError
from naviguard.inference.artifacts import Artifacts, ArtifactsNotFoundError

NAVIC_FEATURES = ["clock_bias_s", "clock_drift_s_per_s"]
CANDIDATES = ("broadcast_drift", "ridge", "attn_lstm")
META_NAME, MODEL_NAME, FORECASTER_NAME = "model_meta.json", "model.keras", "forecaster.pkl"
_EMPIRICAL_QUANTILE = 0.995


# ── windows ───────────────────────────────────────────────────────────────────
def _build(series: dict, seq_len: int, horizon: int) -> dict[int, tuple[Windows, dict]]:
    out = {}
    for sat, s in sorted(series.items()):
        try:
            w = make_windows(s, seq_len, horizon)
            out[sat] = (w, split_windows(w, horizon))
        except ValueError:
            continue                                   # too short / too gappy for this window size
    if not out:
        raise ValueError("no satellite has enough gap-free data for the requested seq_len/horizon")
    return out


def _stack(per_sat: dict, split: str, satellite_id: int | None = None):
    if satellite_id is not None:
        if satellite_id not in per_sat:
            raise ValueError(f"unknown satellite_id {satellite_id}; available: {sorted(int(s) for s in per_sat)}")
        per_sat = {satellite_id: per_sat[satellite_id]}
    X = np.concatenate([w.X[ix[split]] for w, ix in per_sat.values()])
    y = np.concatenate([w.y[ix[split]] for w, ix in per_sat.values()])
    last = np.concatenate([w.last_bias_ns[ix[split]] for w, ix in per_sat.values()])
    sat = np.concatenate([np.full(len(ix[split]), s) for s, (_, ix) in per_sat.items()])
    return X, y, last, sat


def _sha256(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── training / persistence ────────────────────────────────────────────────────
def train_navic(csv_path: str = NAVIC_TELEMETRY_CSV, out_dir: str = NAVIC_DIR, seq_len: int = SEQ_LEN,
                horizon: int = HORIZON, epochs: int = 40, seed: int = 0, verbose: bool = True) -> dict:
    if not os.path.exists(csv_path):
        raise TelemetryNotFoundError(f"NavIC telemetry not found: {csv_path}. Run `naviguard fetch` first.")
    series = load_series(csv_path)
    per_sat = _build(series, seq_len, horizon)
    Xtr, ytr, _, _ = _stack(per_sat, "train")
    Xva, yva, _, _ = _stack(per_sat, "val")
    Xte, yte, _, _ = _stack(per_sat, "test")
    if verbose:
        print(f"[train-navic] satellites {sorted(per_sat)}  train/val/test windows: "
              f"{len(Xtr)}/{len(Xva)}/{len(Xte)}")

    fitted, table = {}, {}
    for kind in CANDIDATES:
        kw = {"epochs": epochs} if kind == "attn_lstm" else {}
        m = make_model(kind, horizon, seed=seed, **kw).fit(Xtr, ytr, Xva, yva)
        val_mae = float(np.abs(yva - m.predict(Xva)).mean())
        test_mae = np.abs(yte - m.predict(Xte)).mean(0).tolist()
        fitted[kind] = m
        table[kind] = {"val_mae_ns": val_mae, "test_mae_ns": test_mae}
        if verbose:
            print(f"[train-navic] {kind:<16} val MAE {val_mae:8.3f} ns   test step-1 MAE {test_mae[0]:8.3f} ns")

    chosen = min(table, key=lambda k: table[k]["val_mae_ns"])         # selected on validation only
    persist_mae = np.abs(yte).mean(0).tolist()
    step_s = float(np.median([series[s].step_s for s in per_sat]))
    meta = {
        "profile": "navic", "seq_len": seq_len, "horizon": horizon, "features": NAVIC_FEATURES,
        "target": "clock_bias_s", "trained_at": datetime.now(timezone.utc).isoformat(),
        "best_val_loss": table[chosen]["val_mae_ns"],                  # validation MAE (ns) of the served model
        "epochs_run": epochs if chosen == "attn_lstm" else 0,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)), "n_test": int(len(Xte)),
        "test_mae_ns": table[chosen]["test_mae_ns"], "persistence_mae_ns": persist_mae,
        "hparams": {"kind": chosen, **asdict(ModelHParams())}, "candidates": table,
        "satellites": sorted(per_sat), "step_s": step_s, "telemetry_sha256": _sha256(csv_path), "seed": seed,
    }

    os.makedirs(out_dir, exist_ok=True)
    for name in (MODEL_NAME, FORECASTER_NAME):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            os.remove(p)
    best = fitted[chosen]
    if chosen == "attn_lstm":
        best.model.save(os.path.join(out_dir, MODEL_NAME))
        meta["norm"] = {"mu": best.mu.tolist(), "sd": best.sd.tolist(), "ysd": best.ysd.tolist()}
    elif chosen == "ridge":
        joblib.dump(best, os.path.join(out_dir, FORECASTER_NAME))
    with open(os.path.join(out_dir, META_NAME), "w") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"[train-navic] serving '{chosen}' (lowest validation MAE)  →  {out_dir}")
    return meta


def load_navic(out_dir: str = NAVIC_DIR) -> Artifacts:
    meta_path = os.path.join(out_dir, META_NAME)
    if not os.path.exists(meta_path):
        raise ArtifactsNotFoundError(
            f"No NavIC model at {out_dir}. Run `naviguard fetch` then `naviguard train-navic` first.")
    with open(meta_path) as f:
        meta = json.load(f)
    kind, horizon = meta["hparams"]["kind"], meta["horizon"]
    if kind == "broadcast_drift":
        model = BroadcastDrift(horizon)
    elif kind == "ridge":
        model = joblib.load(os.path.join(out_dir, FORECASTER_NAME))
    else:
        import tensorflow as tf

        from naviguard.models.attention import AttentionPooling
        model = Neural(kind, horizon)
        model.model = tf.keras.models.load_model(
            os.path.join(out_dir, MODEL_NAME), custom_objects={"AttentionPooling": AttentionPooling})
        model.mu, model.sd, model.ysd = (np.array(meta["norm"][k]) for k in ("mu", "sd", "ysd"))
    return Artifacts(model=model, scaler=None, meta=meta)


def is_navic(artifacts: Artifacts) -> bool:
    return artifacts.meta.get("profile") == "navic"


# ── inference ─────────────────────────────────────────────────────────────────
_lock = threading.Lock()
_cache: dict = {}


def _file_version(path: str):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _windows(artifacts: Artifacts) -> dict:
    seq_len, horizon = artifacts.meta["seq_len"], artifacts.meta["horizon"]
    key = ("windows", id(artifacts), NAVIC_TELEMETRY_CSV, _file_version(NAVIC_TELEMETRY_CSV))
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit[0]
    if not os.path.exists(NAVIC_TELEMETRY_CSV):
        raise TelemetryNotFoundError(f"NavIC telemetry not found: {NAVIC_TELEMETRY_CSV}. Run `naviguard fetch`.")
    per_sat = _build(load_series(NAVIC_TELEMETRY_CSV), seq_len, horizon)
    with _lock:
        if len(_cache) > 12:
            _cache.clear()
        _cache[key] = (per_sat, artifacts)                   # hold artifacts so its id is not recycled
    return per_sat


def _metrics(err: np.ndarray) -> dict:
    return {"mae_ns": np.abs(err).mean(0).tolist(), "rmse_ns": np.sqrt((err ** 2).mean(0)).tolist()}


def evaluate_navic(artifacts: Artifacts, split: str = "test", satellite_id: int | None = None) -> dict:
    key = ("eval", id(artifacts), split, satellite_id, id(_windows(artifacts)))
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit[0]
    per_sat = _windows(artifacts)
    horizon = artifacts.meta["horizon"]
    X, y, last, _ = _stack(per_sat, split, satellite_id)
    pred = artifacts.model.predict(X)
    resid = y - pred

    Xtr, ytr, _, _ = _stack(per_sat, "train")
    base = {
        "persistence": _metrics(y),
        "broadcast_drift": _metrics(y - BroadcastDrift(horizon).predict(X)),
        "linear_extrap": _metrics(y - LinearExtrap(horizon).predict(X)),
        "arima_p10": _metrics(y - ARIMA(horizon).fit(Xtr, ytr, None, None).predict(X)),
        "ridge": _metrics(y - RidgeDirect(horizon).fit(Xtr, ytr, None, None).predict(X)),
    }
    m = _metrics(resid)
    persist1 = base["persistence"]["mae_ns"][0]
    result = {
        "split": split, "n_test_samples": int(len(X)), "horizon": horizon, "target_ns": MAE_TARGET_NS,
        **m, "pass_step1": bool(m["mae_ns"][0] <= MAE_TARGET_NS),
        "actual_ns": (last[:, None] + y).tolist(), "predicted_ns": (last[:, None] + pred).tolist(),
        "residual_ns": resid.tolist(), "baselines": base,
        "skill_vs_persistence": float(1 - m["mae_ns"][0] / persist1) if persist1 > 0 else None,
    }
    with _lock:
        _cache[key] = (result, artifacts, per_sat)
    return result


def detect_navic(artifacts: Artifacts, z_threshold: float | None = None, satellite_id: int | None = None) -> dict:
    """Per-satellite robust-z of step-1 residuals, calibrated on each satellite's validation split.

    Broadcast-clock residuals are heavy-tailed (see README), so unless a threshold is given the
    cut-off is the larger of the configured z and the 99.5th percentile of validation |z|.
    """
    per_sat = _windows(artifacts)
    if satellite_id is not None:
        if satellite_id not in per_sat:
            raise ValueError(f"unknown satellite_id {satellite_id}; available: {sorted(int(s) for s in per_sat)}")
        per_sat = {satellite_id: per_sat[satellite_id]}
    z_val, z_test, centers, scales = [], [], [], []
    for w, ix in per_sat.values():
        r_val = w.y[ix["val"], 0] - artifacts.model.predict(w.X[ix["val"]])[:, 0]
        r_test = w.y[ix["test"], 0] - artifacts.model.predict(w.X[ix["test"]])[:, 0]
        c, s = anomaly.calibrate(r_val)
        centers.append(c)
        scales.append(s)
        z_val.append((r_val - c) / s)
        z_test.append((r_test - c) / s)
    z_val, z_test = np.concatenate(z_val), np.concatenate(z_test)
    thr = z_threshold if z_threshold is not None else max(
        ANOMALY_Z_THRESHOLD, float(np.quantile(np.abs(z_val), _EMPIRICAL_QUANTILE)))
    det = anomaly.detect(z_test, 0.0, 1.0, thr)
    det["center"], det["scale"] = float(np.mean(centers)), float(np.mean(scales))     # ns, averaged over satellites
    return det


def forecast_navic(window=None, artifacts: Artifacts | None = None, satellite_id: int | None = None) -> dict:
    """Forecast from raw rows [clock_bias_s, clock_drift_s_per_s] (seq_len of them) or, if omitted,
    from the latest rows of `satellite_id` in the NavIC telemetry CSV."""
    seq_len, step = artifacts.meta["seq_len"], artifacts.meta["step_s"]
    if window is None:
        if not os.path.exists(NAVIC_TELEMETRY_CSV):
            raise TelemetryNotFoundError(f"NavIC telemetry not found: {NAVIC_TELEMETRY_CSV}. Run `naviguard fetch`.")
        df = pd.read_csv(NAVIC_TELEMETRY_CSV)
        sats = sorted(df["satellite_id"].unique())
        sat = sats[0] if satellite_id is None else satellite_id
        if sat not in sats:
            raise ValueError(f"unknown satellite_id {sat}; available: {[int(s) for s in sats]}")
        g = df[df["satellite_id"] == sat].sort_values("timestamp_s").tail(seq_len)
        if len(g) < seq_len:
            raise ValueError(f"satellite {sat} has only {len(g)} rows; need {seq_len}")
        if np.diff(g["timestamp_s"].to_numpy()).max() > 1.5 * step:
            raise ValueError(f"the latest {seq_len} records of satellite {sat} contain a data gap; "
                             "supply an explicit window or wait for fresh data")
        bias_ns, drift = g["clock_bias_s"].to_numpy() * 1e9, g["clock_drift_s_per_s"].to_numpy()
    else:
        arr = np.asarray(window, dtype=np.float64)
        if arr.shape != (seq_len, 2):
            raise ValueError(f"window must have shape ({seq_len}, 2) = [clock_bias_s, clock_drift_s_per_s], "
                             f"got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("window contains NaN or infinite values")
        bias_ns, drift = arr[:, 0] * 1e9, arr[:, 1]
    rel = np.asarray(artifacts.model.predict(relative_window(bias_ns, drift, step)))[0]
    return {"horizon": int(rel.shape[0]), "predicted_clock_bias_ns": (bias_ns[-1] + rel).tolist()}
