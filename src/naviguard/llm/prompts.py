"""naviguard.llm.prompts — prompt templates for the LLM reasoning layer.

Both prompts deliberately frame the LLM as a narrator/reasoner over numbers
NaviGuard's forecasting model and detectors have already computed, not as a
forecaster in its own right — it never sees raw telemetry or is asked to predict
a value. Everything it might want to quote (metre equivalents, the best baseline,
lead times) is computed here, so the model never has to do arithmetic.
"""

from collections import Counter

REPORT_SYSTEM_PROMPT = (
    "You are the operator-facing assistant of NaviGuard, which monitors NavIC (IRNSS) satellite clocks: "
    "it forecasts each satellite's broadcast clock bias and watches the forecast errors for anomalies. "
    "You receive statistics NaviGuard has already computed. Never forecast, compute or invent numbers; "
    "quote only the numbers given, rounded to 2-3 significant figures.\n\n"
    "Background: clock error turns directly into ranging error (the metre equivalents are given). "
    "Broadcast clocks receive routine ground-segment refits that make the bias jump once and then run on "
    "smoothly; these are normal operations, not faults. A spike (one bad record that reverts) or a "
    "sustained excursion (several records drifting the same way) deserves an operator's attention. When "
    "both detectors are present, the refit-aware one is the primary verdict; the raw one over-flags "
    "because these errors are heavy-tailed.\n\n"
    "Write four short lines of plain text (no Markdown, bullets or asterisks), each starting with its label:\n"
    "Status: a one-sentence verdict and its main reason.\n"
    "Accuracy: step-1 error against the target and in metres, how the error grows to the last step, and "
    "how the model compares with the best baseline.\n"
    "Anomalies: what the refit-aware detector found, naming satellites and event types, and why the raw "
    "detector differs if it does.\n"
    "Action: one concrete next step for the operator.\n"
    "Open each line with the most important finding for this particular data rather than a stock phrase, "
    "and vary your wording."
)

SEVERITY_SYSTEM_PROMPT = (
    "You are the severity checker of NaviGuard, which monitors NavIC satellite clocks by forecasting the "
    "broadcast clock bias and watching the forecast errors. Give an independent second opinion using only "
    "the statistics provided. Routine ground-segment refits (isolated level shifts) are normal operations, "
    "not faults; spikes and sustained excursions flagged by the refit-aware detector, or a step-1 error "
    "above target, are what raise concern. Respond with ONLY a JSON object of the form "
    '{"severity": "nominal|watch|anomalous", "reasoning": "<one or two sentences citing the specific numbers>"}. '
    "No prose outside the JSON."
)

NS_TO_M = 0.299792458            # 1 ns of clock error = 0.3 m of ranging error (speed of light)
MAX_ALARMS_LISTED = 5
MODEL_NAMES = {
    "attn_lstm": "attention-LSTM", "lstm": "LSTM", "gru": "GRU", "ridge": "ridge regression",
    "broadcast_drift": "broadcast-drift extrapolation", "linear_extrap": "linear extrapolation",
    "linear_extrapolation": "linear extrapolation", "arima_p10": "AR(10)", "persistence": "persistence",
}


def _name(kind: str) -> str:
    return MODEL_NAMES.get(kind, kind)


def _lead(steps: int, step_s: float) -> str:
    minutes = steps * step_s / 60
    return f"~{minutes:.0f} min" if minutes < 90 else f"~{minutes / 60:.1f} h"


def _ns(v: float) -> str:
    return f"{v:.3g} ns (≈ {v * NS_TO_M:.2g} m of range)"


def _sat(s) -> str:
    return f"I{int(s):02d}"


def format_stats_for_prompt(stats: dict) -> str:
    ctx = stats.get("context") or {}
    step_s = ctx.get("step_s") or 900.0
    served = ctx.get("model")
    mae, rmse = stats.get("mae_ns", []), stats.get("rmse_ns", [])
    horizon = stats.get("horizon") or len(mae)

    lines = ["CONTEXT"]
    if ctx.get("profile") == "navic":
        sats = [ctx["satellite_id"]] if ctx.get("satellite_id") is not None else ctx.get("satellites") or []
        lines.append("- Data: real NavIC broadcast clocks, " + (
            f"satellite {_sat(sats[0])}" if len(sats) == 1 else "satellites " + ", ".join(map(_sat, sats))))
    elif ctx.get("profile"):
        lines.append("- Data: synthetic NavIC-like telemetry (controlled test data, not a real satellite)")
    if served:
        lines.append(f"- Forecasting model in service: {_name(served)}")
    lines.append(f"- One record every {_lead(1, step_s)}; forecasts run {horizon} steps ({_lead(horizon, step_s)}) ahead")

    lines += ["", f"FORECAST ACCURACY (held-out test data, {stats.get('n_test_samples')} forecasts)"]
    if mae:
        lines.append(f"- Step 1 ({_lead(1, step_s)} ahead): MAE {_ns(mae[0])}"
                     + (f", RMSE {rmse[0]:.3g} ns" if rmse else ""))
        lines.append(f"- Step-1 target: MAE <= {stats.get('target_ns')} ns, "
                     + ("met" if stats.get("pass_step1") else "NOT met"))
        if len(mae) > 1:
            lines.append(f"- Step {len(mae)} ({_lead(len(mae), step_s)} ahead): MAE {_ns(mae[-1])}")
    base = {k: v["mae_ns"][0] for k, v in (stats.get("baselines") or {}).items()
            if k != served and v.get("mae_ns")}
    if base:
        ranked = sorted(base.items(), key=lambda kv: kv[1])
        lines.append("- Baselines, step-1 MAE: " + ", ".join(f"{_name(k)} {v:.3g} ns" for k, v in ranked))
        best, best_mae = ranked[0]
        if mae and best_mae > 0:
            rel = mae[0] / best_mae - 1
            lines.append(f"- Model vs best baseline ({_name(best)}) at step 1: "
                         f"{abs(rel):.0%} {'lower' if rel < 0 else 'higher'} error")
    skill = stats.get("skill_vs_persistence")
    if skill is not None:
        lines.append(f"- Skill vs naive persistence at step 1: {skill:+.1%}")

    det = stats.get("detection")
    if det:
        lines += ["", "ANOMALY MONITORING (step-1 forecast errors on the test data)"]
        ra = det.get("refit_aware")
        if ra:
            shapes = ra["shape_counts"]
            lines.append(
                f"- Refit-aware detector (primary verdict): severity {ra['severity']}, alarms: {ra['n_alarms']}. "
                f"{ra['n_jumps']} jumps above |z| {ra['z_threshold']:.3g}: {shapes['refit']} routine refits "
                f"(not faults), {shapes['spike']} spikes, {shapes['sustained']} sustained excursions, "
                f"{shapes['edge']} unclassifiable at data gaps")
            refits = Counter(e["satellite_id"] for e in ra["events"]
                             if e["label"] == "refit" and e.get("satellite_id") is not None)
            if refits:
                lines.append("  - routine refits by satellite: "
                             + ", ".join(f"{_sat(s)} {n}" for s, n in sorted(refits.items())))
            alarmed = sorted((e for e in ra["events"] if e.get("alarm_index") is not None),
                             key=lambda e: -abs(e["z"]))
            for e in alarmed[:MAX_ALARMS_LISTED]:
                where = f"{_sat(e['satellite_id'])} " if e.get("satellite_id") is not None else ""
                lines.append(f"  - alarm: {where}{e['label']} (z {e['z']:+.1f})")
            if len(alarmed) > MAX_ALARMS_LISTED:
                lines.append(f"  - ... and {len(alarmed) - MAX_ALARMS_LISTED} more alarms")
        label = ("Raw residual detector (secondary; over-flags heavy-tailed errors)" if ra
                 else "Residual anomaly detector")
        lines.append(f"- {label}: severity {det['severity']}, {det['n_flagged']} of {det['n_scored']} "
                     f"windows flagged at |z| > {det['z_threshold']:.3g}, largest |z| {det['max_abs_z']:.1f}")
    return "\n".join(lines)
