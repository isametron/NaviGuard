"""Tests for split-conformal forecast ranges (hermetic, no network)."""

import numpy as np
import pandas as pd
import pytest

from naviguard.benchmark.intervals import build_interval_report, conformal_halfwidth, interval_score
from naviguard.benchmark.run import run_benchmark
from naviguard.cli import main


def test_halfwidth_uses_the_finite_sample_rank():
    err = np.arange(1, 20, dtype=float)[:, None]          # 19 calibration errors: rank ceil(20 * 0.9) = 18
    assert conformal_halfwidth(err, 0.1)[0] == 18.0
    assert np.isinf(conformal_halfwidth(err[:5], 0.1)[0])  # too few windows for a 90% range


def test_interval_score_adds_the_miss_penalty():
    y, lo, hi = np.array([0.0, 5.0, -3.0]), np.full(3, -1.0), np.full(3, 1.0)
    np.testing.assert_allclose(interval_score(y, lo, hi, 0.1), [2.0, 2.0 + 20 * 4.0, 2.0 + 20 * 2.0])


def test_ranges_reach_nominal_coverage_when_errors_are_exchangeable(tmp_path):
    rng = np.random.default_rng(0)
    n_val, n_test, steps = 2000, 6000, 3
    z = {"val_y": rng.normal(0, 1, (n_val, steps)), "y": rng.normal(0, 1, (n_test, steps)),
         "val_fold": np.repeat([0, 1], n_val // 2), "fold": np.repeat([0, 1], n_test // 2)}
    z["val_zero"], z["zero"] = np.zeros((n_val, steps)), np.zeros((n_test, steps))
    np.savez(tmp_path / "predictions_I02.npz", **z)
    res = build_interval_report(str(tmp_path), alpha=0.1)
    assert set(res["model"]) == {"zero"} and set(res["step"]) == {1, 2, 3}
    assert np.allclose(res["coverage"], 0.9, atol=0.02)
    assert np.allclose(res["width"], 2 * 1.645, atol=0.15)  # Gaussian 90% range is ±1.645σ
    assert "90% split-conformal" in (tmp_path / "intervals.md").read_text(encoding="utf-8")


def test_cli_scores_a_real_benchmark_and_rejects_bad_input(tmp_path, capsys):
    t = np.arange(400) * 912.0
    bias = 5.6e5 + 30 * np.arange(400) + np.cumsum(np.random.default_rng(1).normal(0, 0.3, 400))
    pd.DataFrame({"satellite_id": 2, "timestamp_s": t, "clock_bias_s": bias * 1e-9,
                  "clock_drift_s_per_s": 30 / 912 * 1e-9}).to_csv(tmp_path / "t.csv", index=False)
    bench = tmp_path / "bench"
    run_benchmark(str(tmp_path / "t.csv"), str(bench), models=("persistence", "broadcast_drift"), seq_len=10,
                  horizon=4, n_folds=2, seeds=1, min_windows=100, verbose=False)
    assert main(["intervals", "--bench-dir", str(bench)]) == 0
    res = pd.read_csv(bench / "intervals.csv")
    w = res[res.step == 4].set_index("model")["width"]
    assert w["broadcast_drift"] < w["persistence"]          # the better forecaster gets the narrower range
    assert main(["intervals", "--bench-dir", str(bench), "--alpha", "1.5"]) == 1
    assert "between 0 and 1" in capsys.readouterr().err
    with pytest.raises(FileNotFoundError):
        build_interval_report(str(tmp_path / "missing"))
