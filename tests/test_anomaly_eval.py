"""Tests for the real-series anomaly-detector evaluation (hermetic, synthetic gaussian-noise series)."""

import numpy as np
import pandas as pd

from naviguard.benchmark.anomaly_eval import (
    DETECTORS, KINDS, _cusum_alarms, evaluate_injection, inject, real_event_agreement, run_anomaly_eval,
)
from naviguard.benchmark.windows import SatSeries

STEP = 912.0


def _series(n=1500, slope_ns=30.0, noise=0.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * STEP
    return SatSeries(satellite=2, t=t, bias_ns=5.6e5 + slope_ns * np.arange(n) + rng.normal(0, noise, n),
                     drift=np.full(n, slope_ns / STEP * 1e-9))


def test_inject_kinds_change_the_expected_samples():
    s = _series()
    rng = np.random.default_rng(0)
    starts = np.array([600])
    spike = inject(s, "spike", 10.0, starts, rng)
    assert np.count_nonzero(spike.bias_ns != s.bias_ns) == 1
    step = inject(s, "step", 10.0, starts, np.random.default_rng(0))
    assert np.count_nonzero(step.bias_ns != s.bias_ns) == len(s.t) - 600
    ramp = inject(s, "ramp", 10.0, starts, np.random.default_rng(0))
    assert np.count_nonzero(ramp.bias_ns != s.bias_ns) == len(s.t) - 600      # offset persists after the ramp
    assert np.array_equal(s.bias_ns, _series().bias_ns)                        # original untouched


def test_cusum_alarms_on_a_sustained_shift_but_not_noise():
    rng = np.random.default_rng(1)
    quiet = _cusum_alarms(rng.normal(0, 1, 500))
    assert quiet.sum() <= 2
    shifted = np.r_[rng.normal(0, 1, 100), rng.normal(4, 1, 50)]
    assert _cusum_alarms(shifted)[100:].any()


def test_large_faults_are_detected_with_low_false_alarms_on_gaussian_data():
    events, fa = evaluate_injection(_series(), seq_len=20, n_events=6, trials=3)
    big = events[(events["magnitude_sigma"] == 20) & (events["kind"].isin(["spike", "step"]))]
    assert big[big["detector"] == "physics_z"]["detected"].mean() > 0.95
    assert set(events["detector"]) == set(DETECTORS) and set(events["kind"]) == set(KINDS)
    assert fa[fa["detector"] == "physics_z"]["fp_per_1000"].iloc[0] < 20        # gaussian noise -> few alarms
    assert (events["fp_in_trial"] >= 0).all()


def test_real_event_agreement_reports_lift(tmp_path):
    s = _series(n=1200)
    df = pd.DataFrame({"satellite_id": 2, "timestamp_s": s.t, "clock_bias_s": s.bias_ns * 1e-9,
                       "clock_drift_s_per_s": s.drift, "ura_index": 2.0, "health": 0.0})
    df.loc[1000:, "ura_index"] = 2.8
    csv = tmp_path / "t.csv"
    df.to_csv(csv, index=False)
    out = real_event_agreement(str(csv))
    assert list(out["satellite"]) == [2] and {"lift", "chance_share_near_meta"} <= set(out.columns)


def test_run_anomaly_eval_writes_summary(tmp_path):
    s = _series()
    pd.DataFrame({"satellite_id": 2, "timestamp_s": s.t, "clock_bias_s": s.bias_ns * 1e-9,
                  "clock_drift_s_per_s": s.drift, "ura_index": 2.0, "health": 0.0}).to_csv(tmp_path / "t.csv", index=False)
    run_anomaly_eval(str(tmp_path / "t.csv"), str(tmp_path / "out"), n_events=4, trials=2, verbose=False)
    text = (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")
    assert "Precision and F1" in text and "physics_adaptive" in text
