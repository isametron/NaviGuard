"""naviguard.llm.prompts — prompt templates for the LLM reasoning layer.

Both prompts deliberately frame the LLM as a narrator/reasoner over numbers
NaviGuard's attention-LSTM has already computed, not as a forecaster in its
own right — it never sees raw telemetry or is asked to predict a value.
"""

REPORT_SYSTEM_PROMPT = (
    "You are an operator-facing assistant for NaviGuard, a NavIC/GNSS satellite "
    "clock bias and ephemeris error monitoring system. You are given prediction "
    "accuracy statistics that have already been computed by an attention-LSTM "
    "model — you do not forecast or compute anything yourself. Write a short, "
    "clear operator report (3-5 sentences) explaining what the numbers mean in "
    "plain language, whether the model is within its accuracy target, and "
    "whether anything looks worth flagging. Do not invent numbers not given to you."
)

SEVERITY_SYSTEM_PROMPT = (
    "You are an anomaly-severity classifier for NaviGuard, a NavIC/GNSS "
    "satellite clock bias prediction system. You are given prediction error "
    "statistics already computed by an attention-LSTM model. Classify the "
    "situation's severity based only on the numbers given. Respond with ONLY "
    "a compact JSON object of the form "
    '{"severity": "nominal|watch|anomalous", "reasoning": "<one sentence>"}. '
    "No prose outside the JSON."
)


def format_stats_for_prompt(stats: dict) -> str:
    mae = stats.get("mae_ns", [])
    rmse = stats.get("rmse_ns", [])
    lines = [
        f"Target MAE (step 1, 15 min ahead): <= {stats.get('target_ns')} ns",
        f"Step 1 MAE: {mae[0]:.3f} ns" if mae else "Step 1 MAE: n/a",
        f"Step 1 RMSE: {rmse[0]:.3f} ns" if rmse else "Step 1 RMSE: n/a",
        f"Step 1 within target: {stats.get('pass_step1')}",
        f"Forecast horizon: {stats.get('horizon')} steps",
        f"Per-step MAE (ns): {[round(v, 2) for v in mae]}",
        f"Test samples evaluated: {stats.get('n_test_samples')}",
    ]
    return "\n".join(lines)
