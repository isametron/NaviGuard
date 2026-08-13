"""Tests for the FastAPI service.

Hermetic by design: get_artifacts is overridden via FastAPI's
dependency_overrides with fake model/scaler stubs, and the LLM report/
severity functions are monkeypatched at their call site — no trained
model, real telemetry files, or running LM Studio are required.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from naviguard.api.main import app
from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.llm.client import LMStudioUnavailableError

HORIZON = 3
SEQ_LEN = 5
N_FEATURES = 3
FEATURES = ["clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m"]


class FakeModel:
    """Deterministic stand-in for a trained Keras model."""

    def predict(self, X, verbose=0):
        n = X.shape[0]
        return np.tile(np.arange(1, HORIZON + 1, dtype="float32") * 1e-7, (n, 1))


@pytest.fixture
def fake_artifacts(monkeypatch, tmp_path, fake_scaler):
    n_samples = 10
    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES), dtype="float32")
    y = np.zeros((n_samples, HORIZON), dtype="float32")
    x_path = tmp_path / "X_seq.npy"
    y_path = tmp_path / "y_seq.npy"
    np.save(x_path, X)
    np.save(y_path, y)

    # evaluate_on_test() loads sequences from these module-level path
    # constants directly; monkeypatching them here keeps the test hermetic
    # instead of depending on real generated artifacts.
    monkeypatch.setattr("naviguard.inference.predict.X_SEQ_PATH", str(x_path))
    monkeypatch.setattr("naviguard.inference.predict.Y_SEQ_PATH", str(y_path))

    return Artifacts(
        model=FakeModel(),
        scaler=fake_scaler,
        meta={"seq_len": SEQ_LEN, "horizon": HORIZON, "features": FEATURES, "target": "clock_bias_s"},
    )


@pytest.fixture
def client(fake_artifacts):
    app.dependency_overrides[get_artifacts] = lambda: fake_artifacts
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health_always_200():
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_loaded"], bool)


def test_model_info_503_when_untrained(monkeypatch, tmp_path):
    monkeypatch.setattr("naviguard.api.routes.health.META_PATH", str(tmp_path / "does_not_exist.json"))
    with TestClient(app) as c:
        resp = c.get("/model/info")
    assert resp.status_code == 503
    assert resp.json()["error"] == "model_not_trained"


def test_predict_evaluate_returns_expected_shape(client):
    resp = client.get("/predict/evaluate")
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == HORIZON
    assert body["n_test_samples"] == 2  # 10 samples, 80/20 chronological split -> 2 test
    assert len(body["mae_ns"]) == HORIZON


def test_predict_forecast_with_explicit_window(client):
    window = [[0.0, 0.0, 0.0]] * SEQ_LEN
    resp = client.post("/predict", json={"window": window})
    assert resp.status_code == 200
    body = resp.json()
    assert body["horizon"] == HORIZON
    assert len(body["predicted_clock_bias_ns"]) == HORIZON


def test_predict_forecast_rejects_wrong_window_shape(client):
    resp = client.post("/predict", json={"window": [[0.0, 0.0, 0.0]] * (SEQ_LEN + 1)})
    assert resp.status_code == 422  # malformed client input, not a server error


def test_anomaly_report_without_llm(client):
    resp = client.post("/anomaly-report", json={"include_llm": False})
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_report"] is None
    assert body["llm_status"] == "not requested"


def test_anomaly_report_with_llm_success(client, monkeypatch):
    monkeypatch.setattr("naviguard.api.routes.anomaly.generate_operator_report", lambda stats: "All nominal.")
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


def test_anomaly_report_with_llm_unavailable_degrades_gracefully(client, monkeypatch):
    def raise_unavailable(stats):
        raise LMStudioUnavailableError("LM Studio not reachable at http://localhost:1234/v1.")

    monkeypatch.setattr("naviguard.api.routes.anomaly.generate_operator_report", raise_unavailable)

    resp = client.post("/anomaly-report", json={"include_llm": True})
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_report"] is None
    assert "not reachable" in body["llm_status"]
