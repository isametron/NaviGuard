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


def test_overlapping_allan_deviation_matches_quadratic_time_error():
    from naviguard.benchmark.allan import overlapping_allan_deviation

    cadence_s = 912.0
    t = np.arange(10) * cadence_s
    result = overlapping_allan_deviation(t, t**2, max_tau_s=3 * cadence_s)

    np.testing.assert_allclose(result["tau_s"], cadence_s * np.arange(1, 4))
    np.testing.assert_allclose(result["allan_deviation"], np.sqrt(2) * result["tau_s"])


def test_overlapping_allan_deviation_does_not_use_cross_gap_samples():
    from naviguard.benchmark.allan import overlapping_allan_deviation

    cadence_s = 10.0
    first_t = np.arange(8) * cadence_s
    second_t = 10_000 + np.arange(8) * cadence_s
    t = np.r_[first_t, second_t]
    bias = np.r_[first_t, second_t + 100]
    result = overlapping_allan_deviation(t, bias, max_tau_s=3 * cadence_s)

    assert (result["gap_free_stretches"] == 2).all()
    np.testing.assert_allclose(result["allan_deviation"], 0, atol=1e-12)


def test_allan_cli_writes_reports_for_requested_satellites(tmp_path):
    frames = []
    for sat in (2, 9, 10):
        timestamps = np.arange(200, dtype=float) * 912
        frames.append(pd.DataFrame({
            "satellite_id": sat,
            "timestamp_s": timestamps,
            "clock_bias_s": timestamps**2 * 1e-16,
            "clock_drift_s_per_s": np.zeros(len(timestamps)),
        }))
    source = tmp_path / "telemetry.csv"
    pd.concat(frames).to_csv(source, index=False)

    out_dir = tmp_path / "allan"
    assert main(["allan", "--csv", str(source), "--out-dir", str(out_dir)]) == 0

    for sat in (2, 9, 10):
        csv_path = out_dir / f"allan_I{sat:02d}.csv"
        plot_path = out_dir / f"allan_I{sat:02d}.png"
        result = pd.read_csv(csv_path)
        assert result["tau_s"].iloc[0] == 912
        assert result["tau_s"].max() >= 86_400
        assert plot_path.stat().st_size > 0