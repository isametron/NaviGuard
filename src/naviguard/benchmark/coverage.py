"""Per-satellite telemetry coverage and gap reports for real NavIC data."""

import os

import numpy as np
import pandas as pd

from naviguard.benchmark.windows import load_series, make_windows
from naviguard.config import SEQ_LEN


def build_coverage_report(
    csv_path: str,
    out_dir: str,
    horizons: tuple[int, ...] = (1, 6, 12, 24),
    seq_len: int = SEQ_LEN,
    min_windows: int = 200,
    max_gap_factor: float = 1.5,
) -> tuple[str, str]:
    """Write CSV and Markdown coverage reports; return their paths.

    A satellite qualifies for a horizon when it has at least ``min_windows``
    complete, gap-free windows of ``seq_len`` input samples followed by that
    many forecast samples. A gap is any interval exceeding ``max_gap_factor``
    times that satellite's median sample cadence.
    """
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if seq_len <= 0 or min_windows <= 0:
        raise ValueError("seq_len and min_windows must be positive")
    if max_gap_factor <= 0:
        raise ValueError("max_gap_factor must be positive")

    series = load_series(csv_path)
    satellite_stats = {}
    rows = []
    for sat, data in sorted(series.items()):
        deltas = np.diff(data.t)
        cadence = float(np.median(deltas)) if len(deltas) else float("nan")
        if len(deltas) and cadence <= 0:
            raise ValueError(f"satellite {sat} timestamps must be strictly increasing")
        gaps = deltas[deltas > max_gap_factor * cadence] if len(deltas) else np.array([])
        satellite_stats[sat] = {
            "samples": int(len(data.t)),
            "cadence_s": cadence,
            "gap_count": int(len(gaps)),
            "largest_gap_s": float(np.max(gaps)) if len(gaps) else 0.0,
            "start_s": float(data.t[0]),
            "end_s": float(data.t[-1]),
        }

        for horizon in horizons:
            try:
                windows = len(make_windows(data, seq_len, horizon).i)
                reason = "qualified" if windows >= min_windows else "insufficient_windows"
            except ValueError:
                windows = 0
                reason = "no_gap_free_windows"
            rows.append({
                "satellite_id": sat,
                "samples": satellite_stats[sat]["samples"],
                "gap_count": satellite_stats[sat]["gap_count"],
                "median_cadence_s": cadence,
                "largest_gap_s": satellite_stats[sat]["largest_gap_s"],
                "horizon": horizon,
                "seq_len": seq_len,
                "gap_free_windows": windows,
                "min_windows": min_windows,
                "qualified": windows >= min_windows,
                "qualification": reason,
            })

    os.makedirs(out_dir, exist_ok=True)
    csv_out = os.path.join(out_dir, "coverage.csv")
    md_out = os.path.join(out_dir, "coverage.md")
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    lines = [
        "# NavIC telemetry coverage and gaps",
        "",
        f"Source: `{os.path.abspath(csv_path)}`  ",
        f"Qualification: at least {min_windows} gap-free windows with {seq_len} input samples, "
        "then the requested forecast horizon. Gaps exceed 1.5× median satellite cadence.",
        "",
        "## Per-satellite coverage",
        "",
        "| Satellite | Samples | Gap count | Median cadence (s) | Largest gap (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for sat, stats in satellite_stats.items():
        cadence = f"{stats['cadence_s']:.1f}" if np.isfinite(stats["cadence_s"]) else "n/a"
        lines.append(f"| I{sat:02d} | {stats['samples']} | {stats['gap_count']} | {cadence} | "
                     f"{stats['largest_gap_s']:.1f} |")

    lines.extend(["", "## Satellites qualifying by horizon", ""])
    for horizon in horizons:
        eligible = [f"I{r['satellite_id']:02d}" for r in rows
                    if r["horizon"] == horizon and r["qualified"]]
        lines.append(f"- Horizon {horizon}: {', '.join(eligible) if eligible else 'none'}")
    lines.extend([
        "",
        "Detailed per-satellite/per-horizon counts are in `coverage.csv`.",
        "",
    ])
    with open(md_out, "w", encoding="utf-8") as report:
        report.write("\n".join(lines))
    return csv_out, md_out