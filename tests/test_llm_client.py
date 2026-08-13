"""Tests for the LM Studio client and LLM report/severity generation.

None of these tests require a running LM Studio server — the client test
uses an intentionally unreachable local port, and the report/severity tests
use a fake client stub so parsing logic is verified in isolation.
"""

import pytest

from naviguard.config import LLMSettings
from naviguard.llm.client import LMStudioClient, LMStudioUnavailableError
from naviguard.llm.reports import assess_anomaly_severity, generate_operator_report


def test_client_raises_unavailable_when_server_unreachable():
    # Port 1 is a reserved/unused port that should refuse connections immediately.
    settings = LLMSettings(base_url="http://localhost:1/v1", timeout_s=2.0)
    client = LMStudioClient(settings)

    with pytest.raises(LMStudioUnavailableError):
        client.chat("system", "user")


class FakeClient:
    def __init__(self, response: str):
        self._response = response

    def chat(self, system, user):
        return self._response


def test_generate_operator_report_returns_llm_text():
    fake = FakeClient("Everything looks nominal.")
    report = generate_operator_report({"mae_ns": [10.0], "target_ns": 50.0}, client=fake)
    assert report == "Everything looks nominal."


def test_assess_anomaly_severity_parses_clean_json():
    fake = FakeClient('{"severity": "nominal", "reasoning": "MAE well within target."}')
    result = assess_anomaly_severity({"mae_ns": [10.0]}, client=fake)
    assert result == {"severity": "nominal", "reasoning": "MAE well within target."}


def test_assess_anomaly_severity_extracts_json_from_surrounding_prose():
    fake = FakeClient('Sure, here is my assessment:\n{"severity": "watch", "reasoning": "borderline"}\nHope that helps!')
    result = assess_anomaly_severity({"mae_ns": [10.0]}, client=fake)
    assert result["severity"] == "watch"


def test_assess_anomaly_severity_falls_back_on_unparseable_response():
    fake = FakeClient("I'm not sure how to answer that.")
    result = assess_anomaly_severity({"mae_ns": [10.0]}, client=fake)
    assert result["severity"] == "unknown"
    assert "not sure" in result["reasoning"]
