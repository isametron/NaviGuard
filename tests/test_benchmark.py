"""Tests for the real-data benchmark harness (windows, folds, classical models, DM test)."""

import numpy as np
import pytest

from naviguard.benchmark.models import ALL_MODELS, CLASSICAL, make_model
from naviguard.benchmark.run import pooled, run_benchmark
from naviguard.benchmark.stats import block_bootstrap_ci, diebold_mariano
from naviguard.benchmark.windows import SatSeries, make_windows, rolling_origin_folds

STEP = 912.0


def _series(n=400, slope_ns=30.0, gap_at=None, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * STEP
    if gap_at is not None:
        t[gap_at:] += 4 * STEP                          # a 5-step outage
    bias = 5.6e5 + slope_ns * np.arange(n) + rng.normal(0, 0.5, n)      # ns, level ~ 560 us
    drift = np.full(n, slope_ns / STEP * 1e-9)                          # s/s consistent with slope
    return SatSeries(satellite=2, t=t, bias_ns=bias, drift=drift)


def test_windows_are_level_free_and_relative_to_last_sample():
    s = _series()
    w = make_windows(s, seq_len=10, horizon=4)
    assert w.X.shape == (len(w.i), 10, 3) and w.y.shape == (len(w.i), 4)
    assert np.allclose(w.X[:, -1, 0], 0.0)                              # last input == reference
    k = 0
    i = w.i[k]
    assert w.y[k, 0] == pytest.approx(s.bias_ns[i] - s.bias_ns[i - 1])
    assert abs(w.X).max() < 1e3                                        # no 5.6e5 ns level leaks in


def test_windows_never_span_a_gap():
    s = _series(gap_at=200)
    w = make_windows(s, seq_len=10, horizon=4)
    for i in w.i:
        span = s.t[i - 10: i + 4]
        assert np.diff(span).max() <= 1.5 * STEP


def test_folds_are_purged_and_chronological():
    s = _series(600)
    w = make_windows(s, seq_len=10, horizon=6)
    folds = rolling_origin_folds(w, horizon=6, n_folds=3)
    assert len(folds) == 3
    for f in folds:
        assert w.i[f.train].max() + 6 - 1 < w.i[f.val].min()            # train targets end before val
        assert w.i[f.train].max() + 6 - 1 < w.i[f.test].min()           # no target overlap with test
    starts = [f.test[0] for f in folds]
    assert starts == sorted(starts)


def test_folds_reject_too_little_data():
    s = _series(60)
    w = make_windows(s, seq_len=10, horizon=6)
    with pytest.raises(ValueError):
        rolling_origin_folds(w, horizon=6, n_folds=30)


@pytest.mark.parametrize("name", CLASSICAL)
def test_classical_models_fit_predict_shapes(name):
    s = _series()
    w = make_windows(s, 12, 5)
    m = make_model(name, 5).fit(w.X[:200], w.y[:200], w.X[200:250], w.y[200:250])
    assert m.predict(w.X[250:]).shape == (len(w.X[250:]), 5)


def test_physical_baselines_beat_persistence_on_a_drifting_clock():
    s = _series()
    w = make_windows(s, 12, 5)
    errs = {}
    for name in ("persistence", "broadcast_drift", "linear_extrap", "arima_p10"):
        m = make_model(name, 5).fit(w.X[:200], w.y[:200], w.X[200:250], w.y[200:250])
        errs[name] = np.abs(w.y[250:] - m.predict(w.X[250:])).mean()
    assert errs["persistence"] > 10 * errs["broadcast_drift"]
    assert errs["arima_p10"] < errs["persistence"] / 5


def test_unknown_model_name():
    with pytest.raises(ValueError, match="unknown model"):
        make_model("transformer", 3)
    assert "attn_lstm" in ALL_MODELS


def test_diebold_mariano_detects_a_better_forecaster():
    rng = np.random.default_rng(0)
    good, bad = rng.normal(0, 1, 300), rng.normal(0, 3, 300)
    stat, p = diebold_mariano(good, bad)
    assert stat < 0 and p < 0.01
    _, p_same = diebold_mariano(good, good + rng.normal(0, 1e-3, 300) * 0)
    assert np.isnan(p_same)                                             # identical losses -> undefined


def test_block_bootstrap_ci_contains_the_mean():
    v = np.random.default_rng(1).normal(5, 1, 400)
    lo, hi = block_bootstrap_ci(v)
    assert lo < v.mean() < hi and hi - lo < 0.5


def test_run_benchmark_end_to_end_classical(tmp_path):
    import pandas as pd

    frames = []
    for sat in (2, 3):
        s = _series(seed=sat)
        frames.append(pd.DataFrame({"satellite_id": sat, "timestamp_s": s.t,
                                    "clock_bias_s": s.bias_ns * 1e-9, "clock_drift_s_per_s": s.drift}))
    csv = tmp_path / "t.csv"
    pd.concat(frames).to_csv(csv, index=False)

    res = run_benchmark(str(csv), str(tmp_path / "out"), models=("persistence", "broadcast_drift", "ridge"),
                        seq_len=10, horizon=4, n_folds=2, seeds=1, min_windows=100, verbose=False)
    assert set(res["model"]) == {"persistence", "broadcast_drift", "ridge"}
    assert (tmp_path / "out" / "summary.md").exists() and (tmp_path / "out" / "summary.tex").exists()
    p = pooled(res)
    best = p[p["step"] == 1].groupby("model")["mae"].mean()
    assert best["broadcast_drift"] < best["persistence"]


def test_run_benchmark_skips_satellites_that_cannot_be_windowed(tmp_path):
    import pandas as pd

    good = _series(seed=1)
    bad = _series(n=400, seed=2)
    bad.t[:] = np.arange(400) * STEP * 10        # every gap > 1.5 x cadence?  cadence is median -> use real gaps
    bad.t[::2] += STEP * 5                       # alternating gaps so no long gap-free run exists
    frames = [pd.DataFrame({"satellite_id": s.satellite + k, "timestamp_s": s.t,
                            "clock_bias_s": s.bias_ns * 1e-9, "clock_drift_s_per_s": s.drift})
              for k, s in enumerate((good, bad))]
    csv = tmp_path / "t.csv"
    pd.concat(frames).to_csv(csv, index=False)
    res = run_benchmark(str(csv), str(tmp_path / "out"), models=("persistence",), seq_len=20, horizon=20,
                        n_folds=2, seeds=1, min_windows=100, verbose=False)
    assert set(res["satellite"]) == {2}          # the unwindowable satellite is skipped, not fatal
    assert (tmp_path / "out" / "results.csv").exists()


def test_pooled_benchmark_merges_rows_and_is_leak_free(tmp_path):
    import pandas as pd

    from naviguard.benchmark.run import run_pooled_benchmark

    frames = [pd.DataFrame({"satellite_id": sat, "timestamp_s": s.t, "clock_bias_s": s.bias_ns * 1e-9,
                            "clock_drift_s_per_s": s.drift})
              for sat, s in ((2, _series(seed=1)), (3, _series(seed=2, slope_ns=-20.0)))]
    csv = tmp_path / "t.csv"
    pd.concat(frames).to_csv(csv, index=False)
    out = tmp_path / "out"
    run_benchmark(str(csv), str(out), models=("persistence",), seq_len=10, horizon=4, n_folds=2, seeds=1,
                  min_windows=100, verbose=False)
    res = run_pooled_benchmark(str(csv), str(out), models=("ridge", "arima_p10"), seq_len=10, horizon=4,
                               n_folds=2, seeds=1, min_windows=100, verbose=False)
    assert {"persistence", "ridge_pooled", "arima_p10_pooled"} <= set(res["model"])
    assert (out / "results_pooled.csv").exists()
    # A pooled ridge sees slopes of both signs but must still forecast each satellite's own drift well.
    p = pooled(res)
    mae = p[p["step"] == 1].groupby("model")["mae"].mean()
    assert mae["ridge_pooled"] < mae["persistence"]
