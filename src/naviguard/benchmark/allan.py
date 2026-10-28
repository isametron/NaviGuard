"""Overlapping Allan deviation for gap-aware satellite clock-bias series."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from naviguard.benchmark.windows import load_series

SECONDS_PER_DAY = 24 * 60 * 60
DEFAULT_SATELLITES = (2, 9, 10)


def overlapping_allan_deviation(
    t: np.ndarray,
    clock_bias_s: np.ndarray,
    max_tau_s: float = SECONDS_PER_DAY,
    max_gap_factor: float = 1.5,
) -> pd.DataFrame:
    """Compute overlapping Allan deviation using only gap-free sample stretches."""
    t = np.asarray(t, dtype=np.float64)
    clock_bias_s = np.asarray(clock_bias_s, dtype=np.float64)
    if t.ndim != 1 or clock_bias_s.ndim != 1 or len(t) != len(clock_bias_s):
        raise ValueError("timestamps and clock bias must be one-dimensional arrays of equal length")
    if len(t) < 3 or not np.isfinite(t).all() or not np.isfinite(clock_bias_s).all():
        raise ValueError("at least three finite timestamp and clock-bias samples are required")
    if not np.isfinite(max_tau_s) or max_tau_s <= 0:
        raise ValueError("max_tau_s must be positive and finite")
    if not np.isfinite(max_gap_factor) or max_gap_factor <= 0:
        raise ValueError("max_gap_factor must be positive and finite")

    deltas = np.diff(t)
    if np.any(deltas <= 0):
        raise ValueError("timestamps must be strictly increasing")
    cadence_s = float(np.median(deltas))
    gap_indices = np.flatnonzero(deltas > max_gap_factor * cadence_s) + 1
    segment_bounds = np.r_[0, gap_indices, len(t)]
    max_m = int(np.ceil(max_tau_s / cadence_s))
    sums_squared = np.zeros(max_m, dtype=np.float64)
    observation_counts = np.zeros(max_m, dtype=np.int64)
    segment_counts = np.zeros(max_m, dtype=np.int64)

    for start, end in zip(segment_bounds[:-1], segment_bounds[1:]):
        seg_t = t[start:end]
        grid = np.arange(seg_t[0], seg_t[-1] + 1e-9, cadence_s)    # exact cadence, so m samples = m * cadence_s
        bias = np.interp(grid, seg_t, clock_bias_s[start:end])
        for m in range(1, min(max_m, (len(bias) - 1) // 2) + 1):
            second_differences = bias[2 * m:] - 2 * bias[m:-m] + bias[:-2 * m]
            sums_squared[m - 1] += np.dot(second_differences, second_differences)
            observation_counts[m - 1] += len(second_differences)
            segment_counts[m - 1] += 1

    valid = observation_counts > 0
    if not valid.any():
        raise ValueError("no gap-free stretches contain enough samples for an Allan deviation")
    taus = cadence_s * np.arange(1, max_m + 1, dtype=np.float64)
    rows = pd.DataFrame({
        "m": np.arange(1, max_m + 1, dtype=np.int64)[valid],
        "tau_s": taus[valid],
        "tau_days": taus[valid] / SECONDS_PER_DAY,
        "allan_deviation": np.sqrt(
            sums_squared[valid] / (2 * taus[valid] ** 2 * observation_counts[valid])
        ),
        "overlapping_terms": observation_counts[valid],
        "gap_free_stretches": segment_counts[valid],
    })
    if rows["tau_s"].max() < max_tau_s:
        raise ValueError(
            f"gap-free stretches support averaging times only up to {rows['tau_s'].max():.0f} s; "
            f"at least {max_tau_s:.0f} s is required"
        )
    return rows


def build_allan_deviation_reports(
    csv_path: str,
    out_dir: str,
    satellites: tuple[int, ...] = DEFAULT_SATELLITES,
    max_tau_s: float = SECONDS_PER_DAY,
    max_gap_factor: float = 1.5,
) -> list[tuple[str, str]]:
    """Write a CSV and log-log plot for each requested satellite."""
    series = load_series(csv_path)
    missing = sorted(set(satellites) - set(series))
    if missing:
        raise ValueError(f"CSV is missing requested satellites: {', '.join(f'I{s:02d}' for s in missing)}")

    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for sat in satellites:
        data = series[sat]
        results = overlapping_allan_deviation(
            data.t, data.bias_ns / 1e9, max_tau_s=max_tau_s, max_gap_factor=max_gap_factor,
        )
        csv_out = os.path.join(out_dir, f"allan_I{sat:02d}.csv")
        plot_out = os.path.join(out_dir, f"allan_I{sat:02d}.png")
        results.to_csv(csv_out, index=False)

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.loglog(results["tau_s"], results["allan_deviation"], "o-", ms=3, lw=1.2)
        ax.set_xlabel("Averaging time τ (s)")
        ax.set_ylabel("Overlapping Allan deviation")
        ax.set_title(f"NavIC I{sat:02d} clock-bias stability")
        ax.grid(True, which="both", alpha=0.3)
        fig.savefig(plot_out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        outputs.append((csv_out, plot_out))
    return outputs
