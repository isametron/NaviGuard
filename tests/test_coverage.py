"""Tests for gap-aware per-satellite coverage reporting."""

import numpy as np
import pandas as pd

from naviguard.benchmark.coverage import build_coverage_report
from naviguard.cli import main


def _telemetry(path):
    frames = []
    for sat, n in ((2, 300), (9, 10)):
        timestamps = np.arange(n, dtype=float) * 900
        if sat == 2:
            timestamps[150:] += 9000
        frames.append(pd.DataFrame({
            "satellite_id": sat,
            "timestamp_s": timestamps,
            "clock_bias_s": np.arange(n, dtype=float) * 1e-9,
            "clock_drift_s_per_s": np.full(n, 1e-12),
        }))
    pd.concat(frames).to_csv(path, index=False)
    return path


def test_coverage_report_counts_samples_gaps_and_horizon_eligibility(tmp_path):
    source = _telemetry(tmp_path / "telemetry.csv")
    csv_path, md_path = build_coverage_report(
        str(source), str(tmp_path / "coverage"), horizons=(1, 24), seq_len=10, min_windows=250,
    )

    report = pd.read_csv(csv_path)
    sat2 = report[report["satellite_id"] == 2].set_index("horizon")
    sat9 = report[report["satellite_id"] == 9].set_index("horizon")
    assert sat2.loc[1, "samples"] == 300
    assert sat2.loc[1, "gap_count"] == 1
    assert sat2.loc[1, "qualified"]
    assert not sat2.loc[24, "qualified"]
    assert sat9.loc[1, "qualification"] == "no_gap_free_windows"
    assert "Horizon 1: I02" in open(md_path, encoding="utf-8").read()
    assert "Horizon 24:" in open(md_path, encoding="utf-8").read()


def test_coverage_cli_writes_reports(tmp_path):
    source = _telemetry(tmp_path / "telemetry.csv")
    out_dir = tmp_path / "coverage"
    rc = main(["coverage", "--csv", str(source), "--out-dir", str(out_dir),
               "--horizons", "1,6", "--seq-len", "10", "--min-windows", "50"])
    assert rc == 0
    assert (out_dir / "coverage.csv").exists()
    assert (out_dir / "coverage.md").exists()


def test_coverage_rejects_invalid_horizon(tmp_path, capsys):
    source = _telemetry(tmp_path / "telemetry.csv")
    rc = main(["coverage", "--csv", str(source), "--horizons", "1,0"])
    assert rc == 1
    assert "positive integers" in capsys.readouterr().err