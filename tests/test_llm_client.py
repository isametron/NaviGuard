"""Tests for the LM Studio client and LLM report/severity generation.

None of these tests require a running LM Studio server — the client test
uses an intentionally unreachable local port, and the report/severity tests
use a fake client stub so parsing logic is verified in isolation.
"""

import pytest

from naviguard.config import LLMSettings
from naviguard.llm.client import LMStudioClient, LMStudioUnavailableError
from naviguard.llm.prompts import format_stats_for_prompt
from naviguard.llm.reports import assess_anomaly_severity, generate_operator_report


def test_client_raises_unavailable_when_server_unreachable():
    # Port 1 is a reserved/unused port that should refuse connections immediately.
    settings = LLMSettings(base_url="http://localhost:1/v1", timeout_s=2.0)
    client = LMStudioClient(settings)

    with pytest.raises(LMStudioUnavailableError):
        client.chat("system", "user")


def _scripted_client(monkeypatch, outcomes):
    """An LMStudioClient whose completions endpoint replays `outcomes` (exceptions or reply text)."""
    import types

    import naviguard.llm.client as client_mod
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    client = LMStudioClient(LLMSettings(base_url="http://localhost:1/v1"))
    calls = []

    def create(**kw):
        calls.append(kw)
        out = outcomes[min(len(calls), len(outcomes)) - 1]
        if isinstance(out, Exception):
            raise out
        msg = types.SimpleNamespace(content=out)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    client._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    return client, calls


def _server_error():
    import httpx
    from openai import InternalServerError
    req = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
    return InternalServerError("<html>Internal Server Error</html>", response=httpx.Response(500, request=req),
                               body=None)


def test_chat_retries_a_server_error_while_the_model_loads(monkeypatch):
    client, calls = _scripted_client(monkeypatch, [_server_error(), _server_error(), "Status: ok"])
    assert client.chat("system", "user") == "Status: ok"
    assert len(calls) == 3


def test_chat_gives_up_on_persistent_server_errors_with_a_clean_message(monkeypatch):
    client, calls = _scripted_client(monkeypatch, [_server_error()])
    with pytest.raises(LMStudioUnavailableError, match="server error") as exc:
        client.chat("system", "user")
    assert len(calls) == 3 and "<html>" not in str(exc.value)


def test_chat_reports_a_timeout_as_a_timeout_and_does_not_retry(monkeypatch):
    import httpx
    from openai import APITimeoutError
    timeout = APITimeoutError(request=httpx.Request("POST", "http://localhost:1/v1/chat/completions"))
    client, calls = _scripted_client(monkeypatch, [timeout])
    with pytest.raises(LMStudioUnavailableError, match="timed out"):
        client.chat("system", "user")
    assert len(calls) == 1


class FakeClient:
    def __init__(self, response: str):
        self._response = response
        self.temperature = "unset"

    def chat(self, system, user, temperature=None):
        self.temperature = temperature
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


def test_report_is_sampled_warmly_but_severity_json_stays_near_deterministic():
    fake = FakeClient('{"severity": "nominal", "reasoning": "ok"}')
    generate_operator_report({"mae_ns": [1.0]}, client=fake)
    assert fake.temperature is None                                  # -> LLMSettings.temperature
    assess_anomaly_severity({"mae_ns": [1.0]}, client=fake)
    assert fake.temperature <= 0.2
    assert LLMSettings().temperature > 0.5


def _navic_stats():
    events = [{"index": 3, "label": "refit", "alarm_index": None, "z": 9.0, "satellite_id": 2},
              {"index": 9, "label": "refit", "alarm_index": None, "z": -7.0, "satellite_id": 10},
              {"index": 20, "label": "sustained", "alarm_index": 22, "z": 6.3, "satellite_id": 2}]
    det = {"n_flagged": 3, "n_scored": 1256, "z_threshold": 12.4, "max_abs_z": 40.0, "severity": "anomalous",
           "refit_aware": {"n_jumps": 3, "z_threshold": 4.0, "n_alarms": 1, "severity": "watch", "events": events,
                           "shape_counts": {"refit": 2, "spike": 0, "sustained": 1, "edge": 0}}}
    return {"mae_ns": [1.25, 3.0, 9.8], "rmse_ns": [2.0, 4.0, 12.0], "horizon": 3, "target_ns": 50.0,
            "pass_step1": True, "n_test_samples": 1256, "skill_vs_persistence": 0.97, "detection": det,
            "baselines": {"persistence": {"mae_ns": [40.0]}, "broadcast_drift": {"mae_ns": [1.5]},
                          "ridge": {"mae_ns": [1.25]}},
            "context": {"profile": "navic", "model": "ridge", "step_s": 912.0, "satellites": [2, 9, 10],
                        "satellite_id": None}}


def test_prompt_gives_context_and_precomputed_comparisons():
    text = format_stats_for_prompt(_navic_stats())
    assert "satellites I02, I09, I10" in text and "model in service: ridge regression" in text
    assert "MAE 1.25 ns (≈ 0.37 m of range)" in text and "~46 min ahead" in text
    assert "best baseline (broadcast-drift extrapolation) at step 1: 17% lower error" in text
    assert "ridge regression 1.25" not in text                       # the served model is not its own baseline
    one = _navic_stats()
    one["context"]["satellite_id"] = 9
    assert "satellite I09" in format_stats_for_prompt(one)


def test_prompt_puts_the_refit_aware_verdict_first_and_names_alarms():
    text = format_stats_for_prompt(_navic_stats())
    assert text.index("Refit-aware detector (primary verdict): severity watch") < text.index("Raw residual detector")
    assert "routine refits by satellite: I02 1, I10 1" in text
    assert "alarm: I02 sustained (z +6.3)" in text
    plain = _navic_stats()
    plain["detection"]["refit_aware"] = None
    plain.pop("context")
    text = format_stats_for_prompt(plain)
    assert "Refit-aware" not in text and "- Residual anomaly detector: severity anomalous" in text
