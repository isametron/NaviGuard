"""naviguard.data.generate — synthetic NavIC/GNSS telemetry simulation.

Physics-motivated synthetic telemetry, one row per sample per satellite:
  clock_bias      -> 24h sinusoid + Gaussian noise   (NavIC Rubidium spec)
  clock_drift     -> 24h cosine derivative + noise
  ephemeris_error -> 12h sinusoid + noise             (NavIC L5-band orbit accuracy)

These are the same formulas as the original single-satellite generator;
this module makes sample count, satellite count, cadence, seed, and noise
levels configurable instead of hardcoded constants.
"""

import math
import os

import numpy as np
import pandas as pd

from naviguard.config import TELEMETRY_CSV

DEFAULT_N_SAMPLES = 200
DEFAULT_INTERVAL_S = 900
DEFAULT_SEED = 42

DEFAULT_CB_NOISE_STD = 2e-8
DEFAULT_CD_NOISE_STD = 5e-12
DEFAULT_EE_NOISE_STD = 0.05


def generate_satellite_series(
    satellite_id: int,
    n_samples: int = DEFAULT_N_SAMPLES,
    interval_s: int = DEFAULT_INTERVAL_S,
    seed: int = DEFAULT_SEED,
    cb_noise_std: float = DEFAULT_CB_NOISE_STD,
    cd_noise_std: float = DEFAULT_CD_NOISE_STD,
    ee_noise_std: float = DEFAULT_EE_NOISE_STD,
) -> pd.DataFrame:
    """Generate one satellite's telemetry series.

    A small per-satellite phase/frequency jitter (deterministic from
    satellite_id) differentiates multiple satellites; satellite_id=0
    reproduces the original single-satellite formulas exactly (phase=0,
    freq_jitter=1.0).
    """
    rng = np.random.default_rng(seed + satellite_id)
    phase = 0.15 * satellite_id
    freq_jitter = 1.0 + 0.01 * satellite_id

    rows = []
    for i in range(n_samples):
        t = i * interval_s
        cb = (1e-6 * math.sin(freq_jitter * 2 * math.pi * t / 86400 + phase)
              + rng.normal(0, cb_noise_std))
        cd = (1e-10 * math.cos(freq_jitter * 2 * math.pi * t / 86400 + phase)
              + rng.normal(0, cd_noise_std))
        ee = (0.30 * math.sin(freq_jitter * 2 * math.pi * t / 43200 + phase)
              + rng.normal(0, ee_noise_std))
        rows.append((satellite_id, i, t, round(cb, 12), round(cd, 14), round(ee, 6)))

    return pd.DataFrame(rows, columns=[
        "satellite_id", "sample_id", "timestamp_s",
        "clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m",
    ])


def generate_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_satellites: int = 1,
    interval_s: int = DEFAULT_INTERVAL_S,
    seed: int = DEFAULT_SEED,
    out_path: str = TELEMETRY_CSV,
    cb_noise_std: float = DEFAULT_CB_NOISE_STD,
    cd_noise_std: float = DEFAULT_CD_NOISE_STD,
    ee_noise_std: float = DEFAULT_EE_NOISE_STD,
) -> pd.DataFrame:
    """Generate telemetry for n_satellites and write it to out_path as CSV.

    n_satellites=1 (the default) preserves the original single-satellite
    column schema (no satellite_id column) for backward compatibility with
    the rest of the pipeline.
    """
    frames = [
        generate_satellite_series(
            sat_id, n_samples, interval_s, seed, cb_noise_std, cd_noise_std, ee_noise_std
        )
        for sat_id in range(n_satellites)
    ]
    df = pd.concat(frames, ignore_index=True)

    if n_satellites == 1:
        df = df.drop(columns=["satellite_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
