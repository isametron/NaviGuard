"""Tests for the FastAPI service.

Hermetic by design: get_artifacts is overridden via FastAPI's
dependency_overrides with fake model/scaler stubs, telemetry paths are
monkeypatched to a tiny generated CSV, and the LLM functions are
monkeypatched at their call site — no trained model, real telemetry files,
or running LM Studio are required.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from naviguard.api.main import create_app
from naviguard.data.generate import generate_dataset
from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.llm.client import LMStudioUnavailableError

HORIZON = 3
SEQ_LEN = 5
N_SAMPLES = 200
FEATURES = ["clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m"]
# 200 rows: train_end=140, val_end=170 -> test windows i in [170, 200-3] => 28
EXPECTED_TEST_WINDOWS = 28


class FakeModel:
    """Deterministic stand-in for a trained Keras model."""

    def predict(self, X, verbose=0):
        n = X.shape[0]
        return np.tile(np.arange(1, HORIZON + 1, dtype="float32") * 1e-7, (n, 1))


@pytest.fixture
def telemetry(monkeypatch, tmp_path):
    csv = tmp_path / "telemetry.csv"
    generate_dataset(n_samples=N_SAMPLES, out_path=str(csv))
    for target in ("naviguard.inference.predict.TELEMETRY_CSV",
                   "naviguard.api.routes.telemetry.TELEMETRY_CSV",
                   "naviguard.api.routes.health.TELEMETRY_CSV"):
        monkeypatch.setattr(target, str(csv))
    return csv


@pytest.fixture
def fake_artifacts(telemetry, fake_scaler):
    return Artifacts(
        model=FakeModel(),
        scaler=fake_scaler,
        meta={"seq_len": SEQ_LEN, "horizon": HORIZON, "features": FEATURES, "target": "clock_bias_s"},
    )


@pytest.fixture
def client(fake_artifacts):
    app = create_app()
    app.dependency_overrides[get_artifacts] = lambda: fake_artifacts
    with TestClient(app) as c:
        yield c


# ── health / model info / telemetry ──────────────────────────────────────────
def test_health_always_200():
    with TestClient(create_app()) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)
    assert body["llm_reachable"] is None            # not probed by default


def test_health_check_llm_probes_server(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1/v1")   # nothing listens here
    with TestClient(create_app()) as c:
        body = c.get("/health?check_llm=true").json()
    assert body["llm_reachable"] is False


def test_telemetry_returns_latest_rows(telemetry):
    with TestClient(create_app()) as c:
        resp = c.get("/telemetry?limit=2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_rows"] == N_SAMPLES
    assert len(body["rows"]) == 2
    assert body["rows"][-1]["sample_id"] == N_SAMPLES - 1


def test_telemetry_503_when_csv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr("naviguard.api.routes.telemetry.TELEMETRY_CSV", str(tmp_path / "nope.csv"))
    with TestClient(create_app()) as c:
        resp = c.get("/telemetry")
    assert resp.status_code == 503
    assert resp.json()["error"] == "telemetry_missing"


def test_telemetry_satellite_filter(monkeypatch, tmp_path):
    csv = tmp_path / "multi.csv"
    generate_dataset(n_samples=30, n_satellites=2, out_path=str(csv))
    monkeypatch.setattr("naviguard.api.routes.telemetry.TELEMETRY_CSV", str(csv))
    with TestClient(create_app()) as c:
        body = c.get("/telemetry?satellite_id=1&limit=1000").json()
        assert body["n_rows"] == 30
        assert {r["satellite_id"] for r in body["rows"]} == {1}
        assert c.get("/telemetry?satellite_id=9").status_code == 404


def test_model_info_503_when_untrained(monkeypatch, tmp_path):
    monkeypatch.setattr("naviguard.api.routes.health.META_PATH", str(tmp_path / "does_not_exist.json"))
    with TestClient(create_app()) as c:
        resp = c.get("/model/info")
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_not_trained"


# ── predict ───────────────────────────────────────────────────────────────────
def test_predict_evaluate_returns_expected_shape(client):
    resp = client.get("/predict/evaluate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == HORIZON
    assert body["n_test_samples"] == EXPECTED_TEST_WINDOWS
    assert len(body["mae_ns"]) == HORIZON
    assert set(body["baselines"]) == {"persistence", "linear_extrapolation", "ridge"}
    assert "skill_vs_persistence" in body


def test_predict_forecast_with_explicit_window(client):
    window = [[0.0, 0.0, 0.0]] * SEQ_LEN
    resp = client.post("/predict", json={"window": window})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == HORIZON
    assert len(body["predicted_clock_bias_ns"]) == HORIZON


def test_predict_forecast_from_latest_telemetry(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 200
    assert len(resp.json()["predicted_clock_bias_ns"]) == HORIZON


def test_predict_forecast_rejects_wrong_window_shape(client):
    resp = client.post("/predict", json={"window": [[0.0, 0.0, 0.0]] * (SEQ_LEN + 1)})
    assert resp.status_code == 422  # malformed client input, not a server error


def test_predict_forecast_rejects_nan_window(client):
    # JSON has no NaN literal; a very large float overflows to inf after scaling checks only
    # if non-finite, so exercise the guard through the function directly.
    from naviguard.inference.predict import forecast
    bad = [[float("nan"), 0.0, 0.0]] * SEQ_LEN
    with pytest.raises(ValueError, match="NaN"):
        forecast(window=bad, artifacts=Artifacts(FakeModel(), None, {"seq_len": SEQ_LEN, "horizon": HORIZON}))


def test_predict_unknown_satellite_is_422(client):
    resp = client.post("/predict", json={"satellite_id": 42})
    assert resp.status_code == 422


# ── anomaly report ────────────────────────────────────────────────────────────
def test_anomaly_report_without_llm(client):
    resp = client.post("/anomaly-report", json={"include_llm": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_report"] is None
    assert body["llm_status"] == "not requested"
    det = body["detection"]
    assert det["severity"] in {"nominal", "watch", "anomalous"}
    assert det["n_scored"] == EXPECTED_TEST_WINDOWS


def test_anomaly_report_z_threshold_override(client):
    strict = client.post("/anomaly-report", json={"include_llm": False, "z_threshold": 0.1}).json()
    loose = client.post("/anomaly-report", json={"include_llm": False, "z_threshold": 1e9}).json()
    assert strict["detection"]["n_flagged"] >= loose["detection"]["n_flagged"] == 0


def test_anomaly_report_with_llm_success(client, monkeypatch):
    seen = {}

    def fake_report(stats):
        seen["detection"] = stats["detection"]
        return "All nominal."

    monkeypatch.setattr("naviguard.api.routes.anomaly.generate_operator_report", fake_report)
    monkeypatch.setattr(
        "naviguard.api.routes.anomaly.assess_anomaly_severity",
        lambda stats: {"severity": "nominal", "reasoning": "ok"},
    )

    resp = client.post("/anomaly-report", json={"include_llm": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_report"] == "All nominal."
    assert body["llm_severity"]["severity"] == "nominal"
    assert body["llm_status"] == "ok"
    assert seen["detection"]["n_scored"] == EXPECTED_TEST_WINDOWS   # LLM is given the detector verdict


def test_anomaly_report_with_llm_unavailable_degrades_gracefully(client, monkeypatch):
    def raise_unavailable(stats):
        raise LMStudioUnavailableError("LM Studio not reachable at http://localhost:1234/v1.")

    monkeypatch.setattr("naviguard.api.routes.anomaly.generate_operator_report", raise_unavailable)
    monkeypatch.setattr("naviguard.api.routes.anomaly.assess_anomaly_severity", raise_unavailable)

    resp = client.post("/anomaly-report", json={"include_llm": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_report"] is None
    assert "not reachable" in body["llm_status"]
    assert body["detection"] is not None            # numeric analysis survives


# ── security / CORS ───────────────────────────────────────────────────────────
def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setenv("NAVIGUARD_API_KEY", "s3cret")
    assert client.get("/health").status_code == 200                       # always open
    assert client.get("/predict/evaluate").status_code == 401
    assert client.get("/predict/evaluate", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/predict/evaluate", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/model/info").status_code == 401


def test_api_open_when_no_key_configured(client, monkeypatch):
    monkeypatch.delenv("NAVIGUARD_API_KEY", raising=False)
    assert client.get("/predict/evaluate").status_code == 200


def test_cors_allows_configured_origin(client):
    resp = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:5173"
    resp = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in resp.headers
