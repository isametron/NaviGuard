"""naviguard.anomaly.detect — residual-based anomaly detection.

The forecaster's step-1 residuals on a healthy reference period (the
validation split) define a robust baseline (median / MAD). Residuals on new
data are scored as robust z-scores against it; large scores mean the clock
behaved in a way the model, trained on nominal telemetry, could not predict.
Pure numpy — no model or TF import.
"""

import numpy as np

from naviguard.config import (
    ANOMALY_ANOMALOUS_FLAG_RATE, ANOMALY_ANOMALOUS_Z_MULT, ANOMALY_Z_THRESHOLD,
)

_MAD_TO_SIGMA = 1.4826
_MIN_SCALE = 1e-12


def calibrate(reference_residuals: np.ndarray) -> tuple[float, float]:
    """Return (center, scale): median and MAD-derived sigma of healthy residuals."""
    ref = np.asarray(reference_residuals, dtype=np.float64).ravel()
    if ref.size == 0:
        raise ValueError("reference residuals are empty")
    center = float(np.median(ref))
    scale = float(_MAD_TO_SIGMA * np.median(np.abs(ref - center)))
    return center, max(scale, _MIN_SCALE)


def classify_severity(n_flagged: int, n_scored: int, max_abs_z: float,
                      z_threshold: float = ANOMALY_Z_THRESHOLD) -> str:
    if n_flagged == 0:
        return "nominal"
    flag_rate = n_flagged / max(n_scored, 1)
    if flag_rate >= ANOMALY_ANOMALOUS_FLAG_RATE or max_abs_z >= ANOMALY_ANOMALOUS_Z_MULT * z_threshold:
        return "anomalous"
    return "watch"


def detect(residuals: np.ndarray, center: float, scale: float,
           z_threshold: float = ANOMALY_Z_THRESHOLD) -> dict:
    """Score residuals against a calibrated baseline and flag outliers."""
    r = np.asarray(residuals, dtype=np.float64).ravel()
    z = (r - center) / scale
    flagged = np.flatnonzero(np.abs(z) > z_threshold)
    max_abs_z = float(np.max(np.abs(z))) if z.size else 0.0
    return {
        "z_threshold": float(z_threshold),
        "center": float(center),
        "scale": float(scale),
        "n_scored": int(z.size),
        "n_flagged": int(flagged.size),
        "flag_rate": float(flagged.size / z.size) if z.size else 0.0,
        "max_abs_z": max_abs_z,
        "severity": classify_severity(int(flagged.size), int(z.size), max_abs_z, z_threshold),
        "flagged_indices": flagged.tolist(),
        "z_scores": z.tolist(),
    }
