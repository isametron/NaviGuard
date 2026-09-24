"""naviguard.data.generate — synthetic NavIC/GNSS telemetry simulation.

Physics-motivated synthetic telemetry, one row per sample per satellite:
  clock_bias      -> 24h sinusoid + random-walk wander + Gaussian noise
  clock_drift     -> 24h cosine derivative + noise
  ephemeris_error -> 12h sinusoid + noise             (NavIC L5-band orbit accuracy)

The random-walk term (oscillator wander) keeps the series from being
perfectly periodic, so naive extrapolation does not trivially solve it.
Optional labelled anomalies (bias spike, drift excursion, ephemeris jump)
can be injected into the held-out test region only, so training data stays
nominal.
"""

import math
import os

import numpy as np
import pandas as pd

from naviguard.config import TELEMETRY_CSV, TRAIN_FRAC, VAL_FRAC

DEFAULT_N_SAMPLES = 2000
DEFAULT_INTERVAL_S = 900
DEFAULT_SEED = 42

DEFAULT_CB_NOISE_STD = 2e-8
DEFAULT_CD_NOISE_STD = 5e-12
DEFAULT_EE_NOISE_STD = 0.05
DEFAULT_CB_RW_STD = 5e-9      # per-step random-walk std of clock bias (s)

ANOMALY_KINDS = ("spike", "drift_excursion", "ephemeris_jump")


def generate_satellite_series(
    satellite_id: int,
    n_samples: int = DEFAULT_N_SAMPLES,
    interval_s: int = DEFAULT_INTERVAL_S,
    seed: int = DEFAULT_SEED,
    cb_noise_std: float = DEFAULT_CB_NOISE_STD,
    cd_noise_std: float = DEFAULT_CD_NOISE_STD,
    ee_noise_std: float = DEFAULT_EE_NOISE_STD,
    cb_rw_std: float = DEFAULT_CB_RW_STD,
) -> pd.DataFrame:
    """Generate one satellite's telemetry series.

    A small per-satellite phase/frequency jitter (deterministic from
    satellite_id) differentiates multiple satellites.
    """
    rng = np.random.default_rng(seed + satellite_id)
    phase = 0.15 * satellite_id
    freq_jitter = 1.0 + 0.01 * satellite_id

    walk = np.cumsum(rng.normal(0, cb_rw_std, n_samples))
    rows = []
    for i in range(n_samples):
        t = i * interval_s
        cb = (1e-6 * math.sin(freq_jitter * 2 * math.pi * t / 86400 + phase)
              + walk[i] + rng.normal(0, cb_noise_std))
        cd = (1e-10 * math.cos(freq_jitter * 2 * math.pi * t / 86400 + phase)
              + rng.normal(0, cd_noise_std))
        ee = (0.30 * math.sin(freq_jitter * 2 * math.pi * t / 43200 + phase)
              + rng.normal(0, ee_noise_std))
        rows.append((satellite_id, i, t, round(cb, 12), round(cd, 14), round(ee, 6)))

    return pd.DataFrame(rows, columns=[
        "satellite_id", "sample_id", "timestamp_s",
        "clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m",
    ])


def inject_anomalies(df: pd.DataFrame, count: int, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Inject `count` labelled anomalies into the test region of one satellite's
    series (adds an `is_anomaly` 0/1 column). Kinds cycle through ANOMALY_KINDS:
      spike            single-sample clock-bias jump
      drift_excursion  10-sample bias ramp (frequency excursion)
      ephemeris_jump   3-sample ephemeris-error offset
    """
    df = df.copy()
    df["is_anomaly"] = 0
    if count <= 0:
        return df
    rng = np.random.default_rng(seed + 10_000)
    n = len(df)
    start = int(n * (TRAIN_FRAC + VAL_FRAC)) + 5
    stop = n - 15
    if stop - start < count * 2:
        raise ValueError("Not enough samples to inject that many anomalies into the test region.")
    slots = np.linspace(start, stop, count + 1, dtype=int)[:-1]
    jitter_max = max(1, (stop - start) // (count * 2))
    for k, base in enumerate(slots):
        kind = ANOMALY_KINDS[k % len(ANOMALY_KINDS)]
        i = int(base + rng.integers(0, jitter_max))
        sign = rng.choice([-1.0, 1.0])
        if kind == "spike":
            df.loc[i, "clock_bias_s"] += sign * rng.uniform(3e-7, 8e-7)
            df.loc[i, "is_anomaly"] = 1
        elif kind == "drift_excursion":
            ramp = np.linspace(0, sign * rng.uniform(2e-7, 5e-7), 10)
            df.loc[i:i + 9, "clock_bias_s"] += ramp
            df.loc[i:i + 9, "is_anomaly"] = 1
        else:
            df.loc[i:i + 2, "ephemeris_error_m"] += sign * rng.uniform(1.0, 2.0)
            df.loc[i:i + 2, "is_anomaly"] = 1
    return df


def generate_dataset(
    n_samples: int = DEFAULT_N_SAMPLES,
    n_satellites: int = 1,
    interval_s: int = DEFAULT_INTERVAL_S,
    seed: int = DEFAULT_SEED,
    out_path: str = TELEMETRY_CSV,
    cb_noise_std: float = DEFAULT_CB_NOISE_STD,
    cd_noise_std: float = DEFAULT_CD_NOISE_STD,
    ee_noise_std: float = DEFAULT_EE_NOISE_STD,
    cb_rw_std: float = DEFAULT_CB_RW_STD,
    anomaly_count: int = 0,
) -> pd.DataFrame:
    """Generate telemetry for n_satellites and write it to out_path as CSV.

    n_satellites=1 (the default) omits the satellite_id column. With
    anomaly_count > 0, that many labelled anomalies are injected into each
    satellite's test region (adds an `is_anomaly` column).
    """
    frames = []
    for sat_id in range(n_satellites):
        frame = generate_satellite_series(
            sat_id, n_samples, interval_s, seed,
            cb_noise_std, cd_noise_std, ee_noise_std, cb_rw_std,
        )
        if anomaly_count > 0:
            frame = inject_anomalies(frame, anomaly_count, seed + sat_id)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)

    if n_satellites == 1:
        df = df.drop(columns=["satellite_id"])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
