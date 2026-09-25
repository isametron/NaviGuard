"""Tests for the real-data (navic) profile: training/selection, persistence, and the API on top of it.

Hermetic: a small synthetic broadcast-clock-like CSV (drifting bias at ~560 us, 912 s cadence, per-record
noise, a big jump in the held-out tail) stands in for data/navic_telemetry.csv.
"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from naviguard.api.main import create_app
from naviguard.inference import navic
from naviguard.inference.artifacts import get_artifacts

STEP = 912
SEQ, HOR = 10, 3
N = 700


def _csv(path, jump_ns=0.0, sats=(2, 10)):
    frames = []
    for k, sat in enumerate(sats):
        rng = np.random.default_rng(k)
        slope = 30.0 * (1 + 0.3 * k)                              # ns per record
        bias_ns = 5.6e5 + slope * np.arange(N) + np.cumsum(rng.normal(0, 0.3, N)) + rng.normal(0, 0.5, N)
        if jump_ns:
            bias_ns[int(N * 0.93):] += jump_ns                     # a discontinuity late in the test region
        frames.append(pd.DataFrame({
            "satellite_id": sat, "sample_id": np.arange(N), "timestamp_s": np.arange(N) * STEP,
            "clock_bias_s": bias_ns * 1e-9, "clock_drift_s_per_s": slope / STEP * 1e-9,
            "ura_index": 2.0, "health": 0.0, "tgd_s": -1.8e-9}))
    pd.concat(frames).to_csv(path, index=False)
    return path


@pytest.fixture
def trained(tmp_path, monkeypatch):
    csv = _csv(tmp_path / "navic.csv", jump_ns=400.0)
    monkeypatch.setattr(navic, "CANDIDATES", ("broadcast_drift", "ridge"))
    out = tmp_path / "model"
    meta = navic.train_navic(str(csv), str(out), seq_len=SEQ, horizon=HOR, verbose=False)
    monkeypatch.setattr(navic, "NAVIC_TELEMETRY_CSV", str(csv))
    return csv, out, meta


def test_train_selects_on_validation_and_records_test_metrics(trained):
    _, out, meta = trained
    assert meta["profile"] == "navic" and meta["satellites"] == [2, 10]
    assert set(meta["candidates"]) == {"broadcast_drift", "ridge"}
    chosen = meta["hparams"]["kind"]
    assert meta["best_val_loss"] == min(c["val_mae_ns"] for c in meta["candidates"].values())
    assert meta["candidates"][chosen]["val_mae_ns"] == meta["best_val_loss"]
    assert len(meta["test_mae_ns"]) == HOR and meta["n_test"] > 0
    assert meta["persistence_mae_ns"][0] > 2 * meta["test_mae_ns"][0]         # drifting clock: persistence is poor
    assert json.load(open(out / "model_meta.json"))["step_s"] == STEP


def test_neural_candidate_roundtrips_through_disk(tmp_path, monkeypatch):
    csv = _csv(tmp_path / "navic.csv")
    monkeypatch.setattr(navic, "CANDIDATES", ("attn_lstm",))
    out = tmp_path / "m"
    meta = navic.train_navic(str(csv), str(out), seq_len=SEQ, horizon=HOR, epochs=2, verbose=False)
    assert meta["hparams"]["kind"] == "attn_lstm" and "norm" in meta
    arts = navic.load_navic(str(out))
    X = np.random.default_rng(0).normal(size=(4, SEQ, 3))
    p1, p2 = arts.model.predict(X), navic.load_navic(str(out)).model.predict(X)
    assert p1.shape == (4, HOR) and np.allclose(p1, p2, atol=1e-5)


def test_load_navic_missing_model(tmp_path):
    from naviguard.inference.artifacts import ArtifactsNotFoundError
    with pytest.raises(ArtifactsNotFoundError, match="train-navic"):
        navic.load_navic(str(tmp_path / "nothing"))


def test_train_navic_missing_csv(tmp_path):
    from naviguard.errors import TelemetryNotFoundError
    with pytest.raises(TelemetryNotFoundError):
        navic.train_navic(str(tmp_path / "nope.csv"), str(tmp_path / "o"), verbose=False)


def test_evaluate_forecast_and_detection(trained):
    _, out, _ = trained
    arts = navic.load_navic(str(out))
    res = navic.evaluate_navic(arts)
    assert res["horizon"] == HOR and len(res["mae_ns"]) == HOR
    assert set(res["baselines"]) == {"persistence", "broadcast_drift", "linear_extrap", "arima_p10", "ridge"}
    assert res["skill_vs_persistence"] > 0.5
    assert np.array(res["actual_ns"]).min() > 5e5                       # absolute bias, not relative
    assert navic.evaluate_navic(arts) is res                            # cached

    f = navic.forecast_navic(None, arts, satellite_id=2)
    assert f["horizon"] == HOR and f["predicted_clock_bias_ns"][0] > 5e5
    with pytest.raises(ValueError, match="unknown satellite_id"):
        navic.forecast_navic(None, arts, satellite_id=99)
    with pytest.raises(ValueError, match="shape"):
        navic.forecast_navic([[0.0, 0.0]] * (SEQ + 1), arts)

    det = navic.detect_navic(arts)
    assert det["n_scored"] == res["n_test_samples"] and det["n_flagged"] >= 1     # the injected jump
    assert navic.detect_navic(arts, z_threshold=1e9)["n_flagged"] == 0


def test_forecast_rejects_a_latest_window_with_a_gap(tmp_path, monkeypatch):
    csv = _csv(tmp_path / "navic.csv")
    df = pd.read_csv(csv)
    df.loc[(df["satellite_id"] == 2) & (df["sample_id"] >= N - 5), "timestamp_s"] += 5 * STEP
    df.to_csv(csv, index=False)
    monkeypatch.setattr(navic, "CANDIDATES", ("broadcast_drift",))
    navic.train_navic(str(csv), str(tmp_path / "m"), seq_len=SEQ, horizon=HOR, verbose=False)
    monkeypatch.setattr(navic, "NAVIC_TELEMETRY_CSV", str(csv))
    with pytest.raises(ValueError, match="gap"):
        navic.forecast_navic(None, navic.load_navic(str(tmp_path / "m")), satellite_id=2)


# ── API under NAVIGUARD_PROFILE=navic ─────────────────────────────────────────
@pytest.fixture
def client(trained, monkeypatch):
    csv, out, _ = trained
    monkeypatch.setenv("NAVIGUARD_PROFILE", "navic")
    monkeypatch.setattr("naviguard.api.routes.telemetry.NAVIC_TELEMETRY_CSV", str(csv))
    monkeypatch.setattr("naviguard.api.routes.health.NAVIC_TELEMETRY_CSV", str(csv))
    monkeypatch.setattr("naviguard.api.routes.health.NAVIC_META_PATH", str(out / "model_meta.json"))
    arts = navic.load_navic(str(out))
    app = create_app()
    app.dependency_overrides[get_artifacts] = lambda: arts
    with TestClient(app) as c:
        yield c


def test_api_health_reports_navic_profile(client):
    body = client.get("/health").json()
    assert body["profile"] == "navic" and body["model_loaded"] and body["telemetry_available"]


def test_api_model_info_and_telemetry_without_ephemeris_column(client):
    info = client.get("/model/info").json()
    assert info["features"] == ["clock_bias_s", "clock_drift_s_per_s"] and info["n_test"] > 0
    tel = client.get("/telemetry?satellite_id=10&limit=5").json()
    assert tel["n_rows"] == N and "ephemeris_error_m" not in tel["columns"]


def test_api_evaluate_predict_and_anomaly_report(client):
    ev = client.get("/predict/evaluate").json()
    assert ev["horizon"] == HOR and "broadcast_drift" in ev["baselines"]

    p = client.post("/predict", json={"satellite_id": 10}).json()
    assert len(p["predicted_clock_bias_ns"]) == HOR
    bad = client.post("/predict", json={"window": [[0.0, 0.0]] * (SEQ + 1)})
    assert bad.status_code == 422
    assert client.post("/predict", json={"satellite_id": 77}).status_code == 422

    rep = client.post("/anomaly-report", json={"include_llm": False}).json()
    assert rep["detection"]["n_flagged"] >= 1 and rep["detection"]["severity"] in {"watch", "anomalous"}
    assert rep["llm_status"] == "not requested"
