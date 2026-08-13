"""naviguard.llm.reports — LLM-backed operator report + anomaly severity assessment."""

import json
import re

from naviguard.llm.client import LMStudioClient
from naviguard.llm.prompts import (
    REPORT_SYSTEM_PROMPT,
    SEVERITY_SYSTEM_PROMPT,
    format_stats_for_prompt,
)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def generate_operator_report(stats: dict, client: LMStudioClient | None = None) -> str:
    """Have the local LLM narrate already-computed prediction stats into a
    plain-English operator report. Raises LMStudioUnavailableError if the
    local server isn't reachable — callers should catch that and degrade
    gracefully rather than treat it as fatal."""
    client = client or LMStudioClient()
    return client.chat(REPORT_SYSTEM_PROMPT, format_stats_for_prompt(stats))


def assess_anomaly_severity(stats: dict, client: LMStudioClient | None = None) -> dict:
    """Ask the local LLM for a structured severity verdict as a secondary,
    reasoning-based check alongside the numeric MAE-threshold pass/fail —
    not a replacement for it. Local instruct models aren't perfectly
    JSON-reliable, so parsing is defensive with a safe fallback."""
    client = client or LMStudioClient()
    raw = client.chat(SEVERITY_SYSTEM_PROMPT, format_stats_for_prompt(stats))

    match = _JSON_BLOCK_RE.search(raw)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "severity" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    return {"severity": "unknown", "reasoning": raw.strip() or "No parseable response from local model."}
