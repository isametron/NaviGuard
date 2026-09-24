"""Tests for residual anomaly detection and synthetic anomaly injection."""

import numpy as np
import pytest

from naviguard.anomaly.detect import calibrate, classify_severity, detect
from naviguard.data.generate import generate_dataset, generate_satellite_series, inject_anomalies


def test_calibrate_recovers_sigma():
    rng = np.random.default_rng(0)
    center, scale = calibrate(rng.normal(3.0, 2.0, 20_000))
    assert center == pytest.approx(3.0, abs=0.1)
    assert scale == pytest.approx(2.0, rel=0.05)


def test_calibrate_rejects_empty():
    with pytest.raises(ValueError):
        calibrate(np.array([]))


def test_detect_flags_injected_outliers():
    rng = np.random.default_rng(1)
    center, scale = calibrate(rng.normal(0, 1, 5000))
    r = rng.normal(0, 1, 300)
    r[[10, 200]] = [15.0, -20.0]
    out = detect(r, center, scale, z_threshold=4.0)
    assert {10, 200} <= set(out["flagged_indices"])
    assert out["severity"] == "anomalous"          # max |z| >= 2 * threshold


def test_detect_nominal_when_clean():
    rng = np.random.default_rng(2)
    center, scale = calibrate(rng.normal(0, 1, 5000))
    out = detect(rng.normal(0, 1, 300), center, scale, z_threshold=6.0)
    assert out["n_flagged"] == 0 and out["severity"] == "nominal"


def test_severity_levels():
    assert classify_severity(0, 100, 0.0) == "nominal"
    assert classify_severity(1, 100, 5.0, z_threshold=4.0) == "watch"
    assert classify_severity(10, 100, 5.0, z_threshold=4.0) == "anomalous"   # flag rate
    assert classify_severity(1, 100, 9.0, z_threshold=4.0) == "anomalous"    # extreme z


def test_injection_labels_only_test_region_and_perturbs_data():
    base = generate_satellite_series(0, n_samples=1000)
    out = inject_anomalies(base, count=6)
    labelled = np.flatnonzero(out["is_anomaly"].to_numpy())
    assert labelled.size >= 6
    assert labelled.min() >= int(1000 * 0.85)          # never in train/val rows
    assert (not np.allclose(out["clock_bias_s"], base["clock_bias_s"])
            or not np.allclose(out["ephemeris_error_m"], base["ephemeris_error_m"]))


def test_generate_dataset_label_column_only_when_requested(tmp_path):
    df = generate_dataset(n_samples=50, out_path=str(tmp_path / "t.csv"))
    assert "is_anomaly" not in df.columns and "satellite_id" not in df.columns
    df2 = generate_dataset(n_samples=400, anomaly_count=3, out_path=str(tmp_path / "t2.csv"))
    assert df2["is_anomaly"].sum() > 0
