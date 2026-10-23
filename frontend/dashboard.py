"""NaviGuard dashboard (Streamlit).

Reads everything from the FastAPI service over HTTP. Four views: Overview, Forecast, Anomalies and (navic
profile) Satellites. HTML/SVG pieces come from components.py; the look is in style.css.

Run the API first (`naviguard serve`, or `docker compose up`), then:  streamlit run frontend/dashboard.py
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

import components as ui

# ── API config ────────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("NAVIGUARD_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_S = 10
# Optional shared secret; sent as X-API-Key when the API runs with NAVIGUARD_API_KEY set.
API_KEY = os.environ.get("NAVIGUARD_API_KEY")
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}
# /anomaly-report can chain two local-LLM generations (report + severity) of ~60 s each, so give the
# HTTP client enough headroom rather than misreporting a slow answer as "unreachable".
ANOMALY_REPORT_TIMEOUT_S = 150
MAE_TARGET_NS = 50.0

# ── Page config + theme ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="naviguard · navic clock intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(path: str, **params):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, headers=API_HEADERS, timeout=REQUEST_TIMEOUT_S)
        return resp.status_code, resp.json()
    except requests.exceptions.Timeout as e:
        return None, {"error": "timeout", "message": str(e)}
    except requests.exceptions.RequestException as e:
        return None, {"error": "unreachable", "message": str(e)}


def api_post(path: str, json_body: dict, timeout: float = REQUEST_TIMEOUT_S):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, headers=API_HEADERS, timeout=timeout)
        return resp.status_code, resp.json()
    except requests.exceptions.Timeout as e:
        return None, {"error": "timeout", "message": str(e)}
    except requests.exceptions.RequestException as e:
        return None, {"error": "unreachable", "message": str(e)}


# Short-lived cache so switching views doesn't repeat slow calls (the LLM report especially).
# The "Refresh data" button in the sidebar clears it.
@st.cache_data(ttl=45, show_spinner=False)
def cached_get(path: str, params: tuple = ()):
    return api_get(path, **dict(params))


@st.cache_data(ttl=45, show_spinner=False)
def cached_post(path: str, body_json: str, timeout: float = REQUEST_TIMEOUT_S):
    return api_post(path, json.loads(body_json), timeout)


def show(html: str):
    st.markdown(html, unsafe_allow_html=True)


# ── Notices (API problems and empty states) ───────────────────────────────────
def api_unreachable_notice(body: dict | None = None):
    url = f"<code>{ui.esc(API_BASE_URL)}</code>"
    if body and body.get("error") == "timeout":
        show(ui.notice("warn", "The request timed out",
                       f"Waiting on {url}. The local LLM can be slow on its first generation. Try again in a moment."))
        return
    show(ui.notice("error", "Can't reach the NaviGuard API",
                   f"Nothing answered at {url}. Start it with <code>uvicorn naviguard.api.main:app</code> "
                   f"(or <code>docker compose up</code>) and refresh."))


def model_not_trained_notice(message: str, navic: bool = False):
    cmds = ("naviguard fetch", "naviguard train-navic") if navic else ("naviguard preprocess", "naviguard train")
    show(ui.notice("warn", "The model isn't trained yet",
                   f"Run <code>{cmds[0]}</code>, then <code>{cmds[1]}</code>, and refresh. {ui.esc(message)}"))


def unauthorized_notice():
    show(ui.notice("error", "The API rejected the request (401)",
                   "It requires an API key. Set <code>NAVIGUARD_API_KEY</code> in the environment running this "
                   "dashboard, then refresh."))


def eval_error_notice(status, body):
    if status == 401:
        unauthorized_notice()
    elif status == 503:
        model_not_trained_notice(body.get("message", ""), navic=is_navic)
    elif status is None:
        api_unreachable_notice(body)
    else:
        show(ui.notice("error", "Evaluation failed", ui.esc(str(body))))


# ── Profile, model info, satellites ───────────────────────────────────────────
health_status, health_body = api_get("/health")
profile = (health_body.get("profile") or "synthetic") if health_status == 200 else "synthetic"
is_navic = profile == "navic"

info_status, info_body = api_get("/model/info")
model_info = info_body if info_status == 200 else None
served_kind = ((model_info or {}).get("hparams") or {}).get("kind") if is_navic else "attention-lstm"
served_kind = (served_kind or "model").replace("_", "-")

_sat_status, _sat_body = api_get("/telemetry", limit=1)
satellites = (_sat_body.get("satellites") or []) if _sat_status == 200 else []

# Input features come from the model, falling back to the profile's defaults when it isn't trained.
_feature_list = (model_info or {}).get("features") or (["clock_bias_s", "clock_drift_s_per_s"] if is_navic
                                                         else ["bias", "drift", "ephemeris"])
n_features = len(_feature_list)
feature_names = " \u00b7 ".join(
    f.replace("clock_", "").replace("_s_per_s", "").replace("_s", "").replace("_m", "") for f in _feature_list
)

VIEWS = ["Overview", "Forecast", "Anomalies"] + (["Satellites"] if is_navic and len(satellites) > 1 else [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    show(ui.brand())
    satellite_id = None
    sat_note_slot = st.empty()
    if satellites:
        satellite_id = st.selectbox(
            "Satellite", satellites, label_visibility="collapsed",
            format_func=lambda s: f"I{int(s):02d}" if is_navic else f"sat {s}",
        )
    show(ui.side_label("Controls"))
    telemetry_limit = st.slider("Telemetry rows", 10, 200, 60)
    show_drift = st.toggle("Clock drift panel", value=False)
    show_raw = st.toggle("Raw data table", value=True)
    include_llm = st.toggle("Local LLM operator brief", value=True)
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()
    show(ui.side_label("Model"))
    if model_info:
        rows = [("served", served_kind)]
        notes = []
        if is_navic:
            rows += [(n.replace("_", "-"), f"{c['val_mae_ns']:.2f} ns")
                     for n, c in (model_info.get("candidates") or {}).items()]
            notes.append("Validation MAE per candidate. The served model was selected on validation only.")
            notes.append("Real broadcast clocks (DLR BRDM via BKG/ESA): what the satellite broadcasts, not a precise clock product.")
        else:
            notes.append("LSTM(64) \u2192 Dropout(0.2) \u2192 LSTM(32) \u2192 Attention(32) \u2192 Dense(16) \u2192 Dense(horizon)")
        rows += [("window", f"{model_info['seq_len']} samples"), ("horizon", f"{model_info['horizon']} steps")]
        rows.append(("features", str(n_features)))
        notes.append(f"Input features: {feature_names}.")
        show(ui.model_block(rows, notes, MAE_TARGET_NS))
    else:
        show(ui.model_block([("served", "not trained")], ["Run the training command, then refresh."], MAE_TARGET_NS))
    show(ui.side_label("API"))
    show(ui.sat_note(f"{API_BASE_URL} \u00b7 live"))

# ── View switcher (main page, above everything else) ──────────────────────────
show(ui.nav_header("View"))
view = st.radio("View", VIEWS, horizontal=True, label_visibility="collapsed", key="view")

# ── Telemetry (every view needs it) ───────────────────────────────────────────
tele_params = {"limit": telemetry_limit}
if satellite_id is not None:
    tele_params["satellite_id"] = int(satellite_id)
tele_status, tele_body = api_get("/telemetry", **tele_params)

if tele_status is None:
    api_unreachable_notice(tele_body)
    st.stop()
elif tele_status == 401:
    unauthorized_notice()
    st.stop()
elif tele_status == 503:
    show(ui.notice("warn", "No telemetry yet", f"{ui.esc(tele_body.get('message', ''))} Run "
                   f"<code>{'naviguard fetch' if is_navic else 'naviguard generate'}</code> first."))
    st.stop()
elif tele_status != 200:
    show(ui.notice("error", "Telemetry fetch failed", ui.esc(str(tele_body))))
    st.stop()

df = pd.DataFrame(tele_body["rows"])
n_total = tele_body["n_rows"]
sat_label = f"I{int(satellite_id):02d}" if (is_navic and satellite_id is not None) else "navic"
sat_scope = {"satellite_id": int(satellite_id)} if (is_navic and satellite_id is not None) else {}

# Cadence comes from the model or the data, not a constant (real NavIC records arrive every ~912 s).
step_s = (model_info or {}).get("step_s")
if step_s is None and "timestamp_s" in df.columns and len(df) > 1:
    step_s = float(df["timestamp_s"].diff().median())
step_s = step_s or 900

sat_note_slot.markdown(ui.sat_note(f"{n_total:,} samples \u00b7 one every {step_s / 60:.0f} min"), unsafe_allow_html=True)


def load_eval():
    return cached_get("/predict/evaluate", tuple(sat_scope.items()))


def load_forecast():
    body = {"satellite_id": int(satellite_id)} if satellite_id is not None else {}
    return cached_post("/predict", json.dumps(body, sort_keys=True), REQUEST_TIMEOUT_S)


def load_report(with_llm: bool):
    body = {"include_llm": with_llm, **sat_scope}
    return cached_post("/anomaly-report", json.dumps(body, sort_keys=True), ANOMALY_REPORT_TIMEOUT_S)


def summarize_detection(det: dict | None):
    """One shape for both detectors: the refit-aware verdict when the API has it, otherwise the raw one."""
    if not det:
        return None
    ra = det.get("refit_aware")
    base = dict(n_scored=det["n_scored"], raw_flagged=det["n_flagged"], raw_severity=det["severity"],
                raw_thr=det["z_threshold"], raw_max=det["max_abs_z"])
    if ra:
        sc = ra["shape_counts"]
        return dict(base, kind="refit", severity=ra["severity"], alarms=ra["n_alarms"], jumps=ra["n_jumps"],
                    refits=sc.get("refit", 0), spikes=sc.get("spike", 0), sustained=sc.get("sustained", 0),
                    edge=sc.get("edge", 0), z=ra.get("z_scores") or [], thr=ra["z_threshold"],
                    alarm_idx=ra["alarm_indices"], events=ra["events"])
    return dict(base, kind="raw", severity=det["severity"], alarms=det["n_flagged"], jumps=det["n_flagged"],
                refits=0, spikes=0, sustained=0, edge=0, z=det.get("z_scores") or [], thr=det["z_threshold"],
                alarm_idx=det.get("flagged_indices") or [], events=[])


SEV_TONE = {"nominal": "ok", "watch": "warn", "anomalous": "bad"}
SEV_COLOR = {"nominal": ui.TEAL, "watch": ui.PEACH, "anomalous": ui.ALARM}


def mins(k: float) -> str:
    """Minutes for k forecast steps, rounded to the nearest 5 (real NavIC steps are 15.2 min)."""
    return f"{round(k * step_s / 60 / 5) * 5:.0f}"


def page_header(name: str):
    live = f"Live \u00b7 {profile} profile \u00b7 {served_kind}" if model_info else f"Live \u00b7 {profile} profile \u00b7 model not trained"
    # The Satellites view compares all satellites, so the selected one doesn't belong in its breadcrumb.
    crumbs = ["NaviGuard", name] if name == "Satellites" else ["NaviGuard", sat_label, name]
    show(ui.topbar(crumbs, live))


def clock_panels():
    """(title, subtitle, values, unit) for the secondary clock charts."""
    out = []
    if "ephemeris_error_m" in df.columns:
        out.append(("Ephemeris error", "Orbital position drift in metres.", df["ephemeris_error_m"].tolist(), "m"))
        if show_drift:
            out.append(("Clock drift", "Rate of change in clock bias.", (df["clock_drift_s_per_s"] * 1e10).tolist(), "\u00d710\u207b\u00b9\u2070 s/s"))
    elif "clock_drift_s_per_s" in df.columns:
        out.append(("Broadcast drift (af1)", "Rate of change in clock bias, as broadcast.", (df["clock_drift_s_per_s"] * 1e10).tolist(), "\u00d710\u207b\u00b9\u2070 s/s"))
    return out


# ── Views ─────────────────────────────────────────────────────────────────────
def page_overview():
    page_header("Overview")
    ev_status, ev = load_eval()
    fc_status, fc = load_forecast() if ev_status == 200 else (None, {})
    s = None
    if ev_status == 200:
        rep_status, rep = load_report(False)
        if rep_status == 200:
            s = summarize_detection(rep.get("detection"))

    subject = sat_label if (is_navic and satellite_id is not None) else "The forecaster"
    eyebrow = f"{sat_label} \u00b7 {profile} profile \u00b7 {served_kind if model_info else 'no model loaded'}"
    if ev_status == 200:
        mae1 = ev["mae_ns"][0]
        passed = ev["pass_step1"]
        sub = f"{ui.esc(served_kind)} error is {ui.num(f'{mae1:.1f} ns')} at step 1 against the {MAE_TARGET_NS:.0f} ns target"
        if s:
            sub += f", with {ui.num(ui.plural(s['alarms'], 'alarm'))} across {s['n_scored']:,} test windows"
            if s["kind"] == "refit" and s["refits"]:
                sub += f" and {ui.plural(s['refits'], 'routine refit')} set aside"
        show(ui.hero(eyebrow, f"{subject} is {'within' if passed else 'outside'} target.", sub + "."))
    else:
        show(ui.hero(eyebrow, f"{sat_label} telemetry is live.",
                     "The forecast model isn't available yet, so only the raw clock data is shown below."))

    if ev_status == 200:
        horizon = ev["horizon"]
        mae = ev["mae_ns"]
        skill = ev.get("skill_vs_persistence")
        pers = ((ev.get("baselines") or {}).get("persistence") or {}).get("mae_ns")
        resid = [abs(a[0] - p[0]) for a, p in zip(ev["actual_ns"], ev["predicted_ns"])]
        deviation = ui.broadcast_deviation_ns(df)[0] if (is_navic and len(df) >= 4) else df["clock_bias_s"] * 1e6
        cells = [
            dict(label="Telemetry samples", value=f"{n_total:,}", spark=deviation.tolist(), color=ui.TEAL,
                 foot=f"{sat_label} \u00b7 clock vs broadcast drift, last {len(df)}"),
            dict(label="Forecast error \u00b7 step 1", value=f"{mae[0]:.1f}", unit="ns", color=ui.GREY, spark=mae,
                 status="within target" if ev["pass_step1"] else "above target", tone="ok" if ev["pass_step1"] else "bad",
                 foot=f"step {horizon}: {mae[-1]:.1f} ns \u00b7 target \u2264 {MAE_TARGET_NS:.0f} ns"),
            dict(label="Skill vs persistence", value="\u2014" if skill is None else f"{skill:+.0%}", color=ui.TEAL, spark=resid,
                 status=None if skill is None else ("better than persistence" if skill > 0 else "worse than persistence"),
                 tone="ok" if (skill or 0) > 0 else "bad",
                 foot=(f"persistence: {pers[0]:.1f} ns at step 1" if pers else "step-1 residual over the test split")),
        ]
        if s:
            cells.append(dict(label="Alarms" if s["kind"] == "refit" else "Windows flagged", value=str(s["alarms"]),
                              color=SEV_COLOR.get(s["severity"], ui.TEAL), spark=[abs(z) for z in s["z"]], how="max",
                              status=s["severity"], tone=SEV_TONE.get(s["severity"], "ok"),
                              foot=(f"{ui.plural(s['jumps'], 'jump')} \u00b7 {s['refits']} routine refits" if s["kind"] == "refit"
                                    else f"|robust z| above {s['thr']:.1f}")))
        else:
            cells.append(dict(label="Alarms", value="\u2014", foot="anomaly check unavailable"))
        show(ui.kpi_strip(cells))
    else:
        eval_error_notice(ev_status, ev)

    # Main chart: how the clock moves
    ticks, names = ui.time_axis(df)
    stat_items = [("Current bias", f"{df['clock_bias_s'].iloc[-1] * 1e6:,.2f} \u00b5s", "")]
    if is_navic and {"clock_drift_s_per_s", "timestamp_s"} <= set(df.columns) and len(df) >= 4:
        dev, jumps = ui.broadcast_deviation_ns(df)
        peak = float(np.max(np.abs(dev)))
        title = f"The clock strays {peak:,.0f} ns from its broadcast drift line."
        sub = (f"Over the last {len(df)} samples, {ui.plural(len(jumps), 'jump')} exceeded 4\u03c3. "
               f"Routine ground-segment refits show up as steps.")
        main_chart = ui.chart(
            [dict(name="clock minus broadcast drift", values=dev.tolist(), color=ui.TEAL)],
            height=320, x_ticks=ticks, x_names=names, y_unit="ns",
            dots=[dict(series=0, idx=jumps, color=ui.GREY, label=f"jumps > 4\u03c3 ({len(jumps)})")] if jumps else [],
            uid="main", aria="Clock bias minus its broadcast drift line, in nanoseconds",
        )
        stat_items += [("Peak deviation", f"{peak:,.0f} ns", ""), ("Large jumps", str(len(jumps)), "warn" if jumps else "ok")]
    else:
        title, sub = "Clock bias over the window.", "Satellite oscillator offset over time."
        main_chart = ui.chart([dict(name="clock bias", values=(df["clock_bias_s"] * 1e6).tolist(), color=ui.TEAL)],
                              height=320, x_ticks=ticks, x_names=names, y_unit="\u00b5s", uid="main",
                              aria="Clock bias in microseconds")
    if fc_status == 200 and "clock_bias_s" in df.columns:
        last_ns = float(df["clock_bias_s"].iloc[-1]) * 1e9
        d1 = fc["predicted_clock_bias_ns"][0] - last_ns
        stat_items.append((f"Forecast +{mins(1)} min", f"{d1:+.1f} ns", ""))
    show(ui.panel(title, sub, ui.stats(stat_items) + main_chart))

    # Secondary row: raw rows and the drift / ephemeris charts
    panels = clock_panels()
    side_html = [ui.panel(t, sb, ui.chart([dict(values=v, color=ui.TEAL)], height=230, y_unit=u, uid=f"side{i}",
                                          aria=t, x_ticks=ticks, x_names=names, width=520, font=13))
                 for i, (t, sb, v, u) in enumerate(panels)]
    side_html = [h.replace('class="ng-panel"', 'class="ng-panel compact"') for h in side_html]
    if show_raw:
        fmt = {"clock_bias_s": "{:.4e}", "clock_drift_s_per_s": "{:.4e}", "ephemeris_error_m": "{:.4f}", "tgd_s": "{:.3e}"}
        left, right = st.columns([3, 2], gap="large")
        with left:
            show(ui.panel("Latest telemetry", f"The {min(8, len(df))} most recent rows from {sat_label}.", ui.data_table(df.tail(8).drop(columns=["satellite_id"], errors="ignore"), fmt)))
        with right:
            for h in side_html:
                show(h)
    elif side_html:
        for col, h in zip(st.columns(len(side_html), gap="large"), side_html):
            with col:
                show(h)


def page_forecast():
    page_header("Forecast")
    ev_status, ev = load_eval()
    if ev_status != 200:
        eval_error_notice(ev_status, ev)
        return
    fc_status, fc = load_forecast()
    horizon = ev["horizon"]
    mae = ev["mae_ns"]
    eyebrow = f"{sat_label} \u00b7 {served_kind} \u00b7 {ev['n_test_samples']:,} test windows"

    deltas = []
    if fc_status == 200 and "clock_bias_s" in df.columns:
        last_ns = float(df["clock_bias_s"].iloc[-1]) * 1e9
        deltas = [v - last_ns for v in fc["predicted_clock_bias_ns"]]
    if deltas:
        show(ui.hero(eyebrow, f"The clock is forecast to move {deltas[-1]:+.0f} ns in about {mins(len(deltas))} minutes.",
                     f"Relative to the last observed bias on {ui.esc(sat_label)}, from the {ui.esc(served_kind)} forecaster. "
                     f"Typical error at the first step is {ui.num(f'{mae[0]:.1f} ns')}."))
        show(ui.stats([(f"t+{mins(i + 1)} min", f"{d:+.1f} ns", "") for i, d in enumerate(deltas)]))
    else:
        show(ui.hero(eyebrow, "Forecast accuracy on the held-out test split.",
                     f"Step-1 error is {ui.num(f'{mae[0]:.1f} ns')}; error at step {horizon} is {ui.num(f'{mae[-1]:.1f} ns')}."))
        if fc_status == 422:
            st.caption(f"Forecast unavailable for {sat_label}: {fc.get('detail', 'invalid input')}")

    # Actual vs predicted (step 1) and its residual
    actual = [r[0] / 1e3 for r in ev["actual_ns"]]
    predicted = [r[0] / 1e3 for r in ev["predicted_ns"]]
    resid = [(a - p) * 1e3 for a, p in zip(actual, predicted)]
    n = len(actual)
    wticks = [(int(i), f"window {int(i)}") for i in np.linspace(0, n - 1, 5).round()] if n > 1 else []
    body = ui.chart(
        [dict(name="actual", values=actual, color=ui.TEAL, w=1.6), dict(name="predicted", values=predicted, color=ui.GREY, dash=True, w=1.6)],
        height=250, x_ticks=wticks, y_unit="\u00b5s", uid="avp", aria="Actual versus predicted clock bias at step 1",
    ) + ui.chart([dict(name="residual (actual \u2212 predicted)", values=resid, color=ui.ALARM, w=1.3)],
                 height=170, x_ticks=wticks, y_unit="ns", uid="res", aria="Forecast residual in nanoseconds")
    if is_navic:
        body += (f'<p class="ng-note">The clock drifts by tens of \u00b5s across the test period while the model\'s error is only about '
                 f'{mae[0]:.1f} ns, so the two lines overlap. The residual shows what is left over; judge the forecaster against the baselines below.</p>')
    pill = ui.pill("within target" if ev["pass_step1"] else "exceeds target", "ok" if ev["pass_step1"] else "bad")
    show(ui.panel("Predictions track the clock to within " + f"{mae[0]:.1f} ns.", f"Step 1 of {horizon}, scored on {n:,} test windows. "
                  f"MAE {mae[0]:.2f} ns \u00b7 RMSE {ev['rmse_ns'][0]:.2f} ns.", body, aside=pill))

    # Error by step vs baselines
    baselines = ev.get("baselines") or {}
    steps = list(range(1, len(mae) + 1))
    series = [dict(name=f"{served_kind} (served)", values=mae, color=ui.TEAL, w=2.2)]
    for k, color in (("broadcast_drift", ui.ALARM), ("ridge", ui.GREY)):
        if k in baselines and k.replace("_", "-") != served_kind:
            series.append(dict(name=k.replace("_", " "), values=baselines[k]["mae_ns"], color=color, dash=(k == "ridge"), w=1.6))
    left_out = " Persistence is left out because its error is off this scale; it is in the table below." if "persistence" in baselines else ""
    show(ui.panel("Error grows with the forecast horizon.", "Mean absolute error at each step, served model against reference forecasters." + left_out,
                  ui.chart(series, height=260, x_ticks=[(i, f"step {s}") for i, s in enumerate(steps)], y_unit="ns", uid="steps",
                           fill=False, aria="Mean absolute error by forecast step")))

    # Baseline table
    if baselines:
        rows = [dict(name=served_kind, m1=mae[0], m_last=mae[-1], served=True)]
        rows += [dict(name=k.replace("_", " "), m1=v["mae_ns"][0], m_last=v["mae_ns"][-1], served=False)
                 for k, v in baselines.items() if k.replace("_", "-") != served_kind]
        skill = ev.get("skill_vs_persistence")
        aside = ui.pill(f"{skill:+.0%} vs persistence", "ok" if skill > 0 else "bad") if skill is not None else ""
        show(ui.panel("How does it compare?", "Lower is better. Sorted by step-1 error.", ui.baseline_table(rows, horizon), aside=aside))


def page_anomalies():
    page_header("Anomalies")
    ev_status, ev = load_eval()
    if ev_status != 200:
        eval_error_notice(ev_status, ev)
        st.caption("The anomaly check needs a trained model first.")
        return
    with st.spinner("Running the numeric check" + (" and the local LLM report\u2026" if include_llm else "\u2026")):
        rep_status, rep = load_report(include_llm)
    if rep_status is None:
        api_unreachable_notice(rep)
        return
    if rep_status == 401:
        unauthorized_notice()
        return
    if rep_status != 200:
        show(ui.notice("error", "The anomaly report failed", ui.esc(str(rep))))
        return

    s = summarize_detection(rep.get("detection"))
    breach = rep["threshold_breach"]
    eyebrow = f"{sat_label} \u00b7 residual anomaly detector"
    if s:
        head = (f"No alarms in {s['n_scored']:,} test windows." if s["alarms"] == 0
                else f"{ui.plural(s['alarms'], 'alarm')} in {s['n_scored']:,} test windows.")
        if s["kind"] == "refit":
            sub = (f"{ui.num(ui.plural(s['refits'], 'routine refit'))} and {ui.num(ui.plural(s['edge'], 'edge jump'))} were set aside. "
                   f"The raw detector alone would have flagged {ui.num(ui.plural(s['raw_flagged'], 'window'))}.")
        else:
            largest = ui.num(f"{s['raw_max']:.1f}")
            sub = f"{ui.num(s['alarms'])} of {s['n_scored']:,} windows exceeded |robust z| {s['thr']:.1f}; the largest was {largest}."
        show(ui.hero(eyebrow, head, sub))
        verdict = ui.pill(s["severity"], SEV_TONE.get(s["severity"], "neutral")) + " " + \
            ui.pill("threshold breached" if breach else "numeric check nominal", "bad" if breach else "ok")
        show(f'<div style="padding:6px 6px 2px">{verdict}</div>')

        if s["z"]:
            z = [abs(v) for v in s["z"]]
            grey = [e["index"] for e in s["events"] if e["label"] in ("refit", "edge")]
            n = len(z)
            wticks = [(int(i), f"window {int(i)}") for i in np.linspace(0, n - 1, 6).round()] if n > 1 else []
            if s["kind"] == "refit":
                summary = (f"{ui.plural(s['jumps'], 'jump')}: {ui.plural(s['refits'], 'refit')}, {ui.plural(s['spikes'], 'spike')}, "
                           f"{s['sustained']} sustained" + (f", {s['edge']} edge" if s["edge"] else "")
                           + f" \u2192 {ui.plural(s['alarms'], 'alarm')}")
                dots = [dict(series=0, idx=grey, color=ui.GREY, label=f"refit / edge ({len(grey)})")] if grey else []
                dots += [dict(series=0, idx=s["alarm_idx"], color=ui.ALARM, label=f"alarm ({len(s['alarm_idx'])})", r=5)] if s["alarm_idx"] else []
                note = ("Scored on the broadcast-drift residual. Jumps shaped like routine ground-segment refits (isolated level "
                        "shifts, grey) are reported but do not alarm; only spikes and sustained excursions (red) do. "
                        "Alarms are candidates for review, not confirmed faults.")
            else:
                summary = f"{s['alarms']} of {s['n_scored']:,} test windows flagged"
                dots = [dict(series=0, idx=s["alarm_idx"], color=ui.ALARM, label=f"flagged ({len(s['alarm_idx'])})", r=5)] if s["alarm_idx"] else []
                note = "Windows are flagged when |robust z| crosses the threshold."
            chart_html = ui.chart(
                [dict(name="|robust z|", values=z, color=ui.TEAL, w=1.4)], height=320, x_ticks=wticks, y_unit="|z|", dots=dots,
                hlines=[dict(y=s["thr"], label=f"threshold {s['thr']:.1f}", color=ui.ALARM)], uid="zt",
                aria="Absolute robust z-score for each test window with the alarm threshold",
            )
            raw_line = ""
            if s["kind"] == "refit":
                raw_line = (f'<p class="ng-note">Raw detector without refit handling: {ui.esc(s["raw_severity"])}, {s["raw_flagged"]} of '
                            f'{s["n_scored"]:,} windows above |z| {s["raw_thr"]:.1f}, largest |z| {s["raw_max"]:.1f}. It over-flags routine refits.</p>')
            show(ui.panel("Residual timeline", summary, chart_html + f'<p class="ng-note">{ui.esc(note)}</p>' + raw_line,
                          aside=ui.pill(s["severity"], SEV_TONE.get(s["severity"], "neutral"))))
    else:
        show(ui.hero(eyebrow, "The numeric check ran.", "The API returned no detector output for this satellite."))

    # Operator brief (local LLM, optional)
    if rep.get("llm_report"):
        body = ui.brief(rep["llm_report"], rep.get("llm_severity"))
        show(ui.panel("Operator brief", "Written by the local model from the numbers above. It describes the results and never computes them.", body))
    else:
        show(ui.panel("Operator brief", f"Unavailable: {rep['llm_status']}.",
                      '<p class="ng-note">The numeric results above do not depend on the language model. '
                      'Turn on \u201cLocal LLM operator brief\u201d in the sidebar and start LM Studio to get a written summary.</p>'))


def page_satellites():
    page_header("Satellites")
    compare_ids = [s for s in (2, 9, 10) if s in satellites] or list(satellites)
    results = {}
    with st.spinner("Loading satellites\u2026"):
        for sid in compare_ids:
            results[sid] = (cached_get("/telemetry", (("limit", telemetry_limit), ("satellite_id", int(sid)))),
                            cached_get("/predict/evaluate", (("satellite_id", int(sid)),)))

    rows, errs = [], {}
    for sid, ((t_status, t_body), (e_status, e_body)) in results.items():
        r = dict(satellite=f"I{int(sid):02d}", samples=t_body.get("n_rows") if t_status == 200 else None,
                 mae1=None, mae_last=None, rmse1=None, skill=None)
        if e_status == 200:
            r.update(mae1=e_body["mae_ns"][0], mae_last=e_body["mae_ns"][-1], rmse1=e_body["rmse_ns"][0],
                     skill=e_body.get("skill_vs_persistence"))
            errs[sid] = e_body["mae_ns"]
        rows.append(r)

    scored = [r for r in rows if r["mae1"] is not None]
    if scored:
        best = min(scored, key=lambda r: r["mae1"])
        worst = max(scored, key=lambda r: r["mae1"])
        lasts = [r["mae_last"] for r in scored]
        sub = (f"Step-1 error runs from {ui.num(format(best['mae1'], '.1f') + ' ns')} on {ui.esc(best['satellite'])} to "
               f"{ui.num(format(worst['mae1'], '.1f') + ' ns')} on {ui.esc(worst['satellite'])}. Error at the last step runs from "
               f"{ui.num(format(min(lasts), '.1f') + ' ns')} to {ui.num(format(max(lasts), '.1f') + ' ns')}.")
        show(ui.hero(f"{len(compare_ids)} satellites \u00b7 {served_kind}", f"{best['satellite']} has the lowest forecast error.", sub))
    else:
        show(ui.hero("Satellites", "Satellite comparison", "Evaluation isn't available yet. Train the model first."))
    show(ui.panel("Summary", "One row per satellite. Lower error is better; skill is measured against persistence.", ui.sat_table(rows)))

    cols = st.columns(len(compare_ids), gap="medium")
    for i, (col, sid) in enumerate(zip(cols, compare_ids)):
        (t_status, t_body), _ = results[sid]
        with col:
            label = f"I{int(sid):02d}"
            if t_status == 200 and t_body["rows"]:
                sdf = pd.DataFrame(t_body["rows"])
                try:
                    dev, jumps = ui.broadcast_deviation_ns(sdf)
                    ticks, names = ui.time_axis(sdf, 3)
                    show(ui.panel(label, f"Clock vs broadcast drift \u00b7 {ui.plural(len(jumps), 'jump')} over 4\u03c3 \u00b7 peak {np.max(np.abs(dev)):,.0f} ns",
                                  ui.chart([dict(values=dev.tolist(), color=ui.LINE_COLORS[i % 4])], height=210, x_ticks=ticks, width=380, font=14,
                                           x_names=names, y_unit="ns", uid=f"sat{i}", aria=f"{label} clock versus broadcast drift",
                                           dots=[dict(series=0, idx=jumps, color=ui.GREY, label="")] if jumps else [])
                                  ).replace('class="ng-panel"', 'class="ng-panel compact"'))
                except (KeyError, ValueError):
                    show(ui.notice("warn", f"{label}: no drift data", "This satellite's rows lack the drift or timestamp columns."))
            else:
                show(ui.notice("warn", f"{label}: no telemetry", f"The API answered with status {t_status}."))

    # Raw clock bias, one panel per satellite: change from the first sample (absolute bias is ~500 \u00b5s and looks flat).
    bias_cols = st.columns(len(compare_ids), gap="medium")
    for i, (col, sid) in enumerate(zip(bias_cols, compare_ids)):
        (t_status, t_body), _ = results[sid]
        with col:
            label = f"I{int(sid):02d}"
            if t_status == 200 and t_body["rows"]:
                sdf = pd.DataFrame(t_body["rows"])
                if "clock_bias_s" in sdf.columns and len(sdf) >= 2:
                    bias_ns = ((sdf["clock_bias_s"] - sdf["clock_bias_s"].iloc[0]) * 1e9).tolist()
                    ticks, names = ui.time_axis(sdf, 3)
                    show(ui.panel(f"{label} clock bias", "Change from the first sample, in ns.",
                                  ui.chart([dict(values=bias_ns, color=ui.LINE_COLORS[i % 4])], height=210, x_ticks=ticks,
                                           x_names=names, width=380, font=14, y_unit="ns", uid=f"bias{i}",
                                           aria=f"{label} clock bias change from the first sample")
                                  ).replace('class="ng-panel"', 'class="ng-panel compact"'))
                else:
                    show(ui.notice("warn", f"{label}: no bias data", "This satellite's rows lack the clock bias column."))

    if errs:
        series = [dict(name=f"I{int(sid):02d}", values=v, color=ui.LINE_COLORS[i % 4], w=2.0) for i, (sid, v) in enumerate(errs.items())]
        h = max(len(v) for v in errs.values())
        show(ui.panel("Forecast error by step", "Mean absolute error on each satellite's test split.",
                      ui.chart(series, height=280, fill=False, y_unit="ns", x_ticks=[(i, f"step {i + 1}") for i in range(h)],
                               uid="cmp", aria="Mean absolute error by forecast step for each satellite")))


{"Overview": page_overview, "Forecast": page_forecast, "Anomalies": page_anomalies, "Satellites": page_satellites}[view]()

show(ui.footer(
    f"naviguard \u00b7 navic/irnss \u00b7 {served_kind} \u00b7 level-free windows \u00b7 real broadcast clocks \u00b7 dept. of ai & ds, bmsce 2026\u201327"
    if is_navic else
    "naviguard \u00b7 navic/irnss \u00b7 attention-lstm(64\u219232) \u00b7 minmax scaler \u00b7 synthetic telemetry \u00b7 dept. of ai & ds, bmsce 2026\u201327"
))