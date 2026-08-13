"""
dashboard.py  ·  NaviGuard — Pudding-inspired presentation layer
──────────────────────────────────────────────────────────────────
Charcoal grid · colorful cards · bold lowercase serif · lightweight UI
Visual design by Pratyush Narain. Talks to the live naviguard FastAPI
service (attention-LSTM predictions + local-LLM anomaly reports) instead
of reading pre-generated files off disk.

Run:
    uvicorn naviguard.api.main:app          # in one terminal
    streamlit run frontend/dashboard.py     # in another
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

# ── API config ───────────────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("NAVIGUARD_API_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_S = 10

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="naviguard · navic clock intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pudding palette ───────────────────────────────────────────────────────────
BG_CHARCOAL = "#282828"
CARD_YELLOW = "#f9e87a"
CARD_LAVENDER = "#e4dff5"
CARD_SKY = "#9dd4f0"
CARD_PINK = "#ff8fab"
CARD_CREAM = "#f5f0e1"
CARD_MINT = "#b8e8c8"
INK = "#282828"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Libre+Baskerville:ital,wght@0,700;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    .stApp {{
        background-color: {BG_CHARCOAL};
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }}
    .block-container {{
        padding-top: 1.5rem;
        max-width: 1080px;
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    hr {{ border: none; margin: 0; visibility: hidden; }}

    [data-testid="stSidebar"] {{
        background-color: #1e1e1e;
        border-right: none;
    }}
    [data-testid="stSidebar"] h3 {{
        font-family: 'Libre Baskerville', serif !important;
        font-weight: 700 !important;
        color: #fff !important;
        text-transform: lowercase;
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] .stCaption {{
        color: #a8a8a8 !important;
        font-family: 'Inter', sans-serif !important;
    }}
    [data-testid="stSidebar"] code {{
        background: #333 !important;
        color: #f9e87a !important;
        border-radius: 6px;
    }}

    h1, h2, h3 {{
        font-family: 'Libre Baskerville', serif !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-transform: lowercase;
    }}

    .pudding-logo {{
        font-family: 'Fredoka', sans-serif;
        font-size: 3.4rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
        line-height: 1.1;
    }}
    .pudding-sub {{
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-size: 0.86rem;
        color: #888;
        margin: 8px 0 32px 0;
        line-height: 1.55;
    }}
    .section-label {{
        font-family: 'Libre Baskerville', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #fff;
        margin: 32px 0 14px 0;
        text-transform: lowercase;
    }}
    .pudding-card {{
        border-radius: 16px;
        padding: 18px 20px 16px;
        margin-bottom: 4px;
        min-height: 130px;
    }}
    .card-meta {{
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        opacity: 0.5;
        margin-bottom: 10px;
        font-family: 'Inter', sans-serif;
    }}
    .card-title {{
        font-family: 'Libre Baskerville', serif;
        font-weight: 700;
        font-size: 1.12rem;
        line-height: 1.25;
        margin-top: 12px;
    }}
    .card-sub {{
        font-size: 0.8rem;
        opacity: 0.65;
        margin-top: 4px;
        font-family: 'Inter', sans-serif;
    }}
    .card-value {{
        font-family: 'Fredoka', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
    }}
    .chart-wrap {{
        border-radius: 16px;
        padding: 14px 16px 12px;
        margin-bottom: 4px;
    }}
    .chart-footer {{
        padding: 0 4px 4px;
    }}
    .status-pill {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        text-transform: lowercase;
    }}

    [data-testid="stDataFrame"] {{
        border: none !important;
        border-radius: 12px;
        overflow: hidden;
    }}
    [data-testid="stPyplot"] {{
        background: transparent !important;
    }}
    [data-testid="stImage"] {{
        background: transparent !important;
    }}
    .stSlider label, .stToggle label {{
        font-family: 'Inter', sans-serif !important;
        text-transform: lowercase;
    }}
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(meta_l: str, meta_r: str, bg: str, value: str, title: str, subtitle: str) -> str:
    return f"""
    <div class="pudding-card" style="background:{bg}; color:{INK};">
        <div class="card-meta"><span>{meta_l}</span><span>{meta_r}</span></div>
        <div class="card-value">{value}</div>
        <div class="card-title">{title}</div>
        <div class="card-sub">{subtitle}</div>
    </div>
    """


def pudding_plot(series, ylabel: str, bg: str, line_color: str = INK):
    fig, ax = plt.subplots(figsize=(5.4, 2.7), facecolor=bg)
    ax.set_facecolor(bg)
    ax.plot(series.values, color=line_color, lw=2.2, solid_capstyle="round")
    ax.fill_between(range(len(series)), series.values, alpha=0.14, color=line_color)
    ax.set_xlabel("sample", fontsize=8, color=INK, alpha=0.45)
    ax.set_ylabel(ylabel, fontsize=8, color=INK, alpha=0.45)
    ax.tick_params(colors=INK, labelsize=7, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=0.12, linestyle="-", color=INK)
    fig.tight_layout(pad=0.8)
    return fig


def pudding_compare_plot(actual, predicted, ylabel: str, bg: str):
    fig, ax = plt.subplots(figsize=(11, 3.2), facecolor=bg)
    ax.set_facecolor(bg)
    ax.plot(actual, color=INK, lw=2.0, label="actual", solid_capstyle="round")
    ax.plot(predicted, color="#c0392b", lw=2.0, ls="--", label="predicted", solid_capstyle="round")
    ax.fill_between(range(len(actual)), actual, predicted, alpha=0.12, color=INK)
    ax.set_xlabel("test time step", fontsize=8, color=INK, alpha=0.45)
    ax.set_ylabel(ylabel, fontsize=8, color=INK, alpha=0.45)
    ax.tick_params(colors=INK, labelsize=7, length=0)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(alpha=0.12, linestyle="-", color=INK)
    fig.tight_layout(pad=0.8)
    return fig


# ── API helpers ────────────────────────────────────────────────────────────────
# Anomaly-report calls up to two sequential local-LLM generations on the
# backend (report + severity), each with its own ~60s budget — give the
# HTTP client enough headroom to see that through rather than giving up
# early and misreporting a slow response as "unreachable".
ANOMALY_REPORT_TIMEOUT_S = 150


def api_get(path: str, **params):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=REQUEST_TIMEOUT_S)
        return resp.status_code, resp.json()
    except requests.exceptions.Timeout as e:
        return None, {"error": "timeout", "message": str(e)}
    except requests.exceptions.RequestException as e:
        return None, {"error": "unreachable", "message": str(e)}


def api_post(path: str, json_body: dict, timeout: float = REQUEST_TIMEOUT_S):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=json_body, timeout=timeout)
        return resp.status_code, resp.json()
    except requests.exceptions.Timeout as e:
        return None, {"error": "timeout", "message": str(e)}
    except requests.exceptions.RequestException as e:
        return None, {"error": "unreachable", "message": str(e)}


def api_unreachable_banner(body: dict | None = None):
    if body and body.get("error") == "timeout":
        st.markdown(
            f"""
            <div style="background:{CARD_YELLOW};border-radius:14px;padding:18px 22px;color:{INK};
                        font-family:Inter,sans-serif;font-size:0.92rem;line-height:1.6;">
                <b>request timed out</b> waiting on <code style="background:rgba(0,0,0,0.08);
                padding:2px 8px;border-radius:6px;">{API_BASE_URL}</code>.<br/>
                the local llm can be slow on first generation — try again in a moment.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    st.markdown(
        f"""
        <div style="background:{CARD_PINK};border-radius:14px;padding:18px 22px;color:{INK};
                    font-family:Inter,sans-serif;font-size:0.92rem;line-height:1.6;">
            <b>can't reach the naviguard api</b> at <code style="background:rgba(0,0,0,0.08);
            padding:2px 8px;border-radius:6px;">{API_BASE_URL}</code>.<br/>
            start it with <code style="background:rgba(0,0,0,0.08);padding:2px 8px;
            border-radius:6px;">uvicorn naviguard.api.main:app</code> and refresh.
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_not_trained_banner(message: str):
    st.markdown(
        f"""
        <div style="background:{CARD_YELLOW};border-radius:14px;padding:18px 22px;color:{INK};
                    font-family:Inter,sans-serif;font-size:0.92rem;line-height:1.5;">
            model isn't trained yet.<br/>
            run <code style="background:rgba(0,0,0,0.08);padding:3px 8px;border-radius:6px;
            font-size:0.85rem;">naviguard preprocess</code> then
            <code style="background:rgba(0,0,0,0.08);padding:3px 8px;border-radius:6px;
            font-size:0.85rem;">naviguard train</code>.
            <div style="opacity:0.6;font-size:0.78rem;margin-top:8px;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="pudding-logo">naviguard</div>
<div class="pudding-sub">
    navic clock bias & ephemeris prediction · attention-lstm · isro · bmsce 2025–26
</div>
""",
    unsafe_allow_html=True,
)

# ── Model info (drives sidebar + chart labels) ──────────────────────────────────
info_status, info_body = api_get("/model/info")
model_info = info_body if info_status == 200 else None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### controls")
    st.caption(API_BASE_URL)
    telemetry_limit = st.slider("telemetry rows", 10, 200, 60)
    show_drift = st.toggle("clock drift panel", value=False)
    show_raw = st.toggle("raw data table", value=True)
    include_llm = st.toggle("local llm anomaly report", value=True)
    st.markdown("---")
    st.markdown("**model**")
    if model_info:
        st.code(
            f"LSTM(64) -> Drop(0.2)\nLSTM(32) -> Attention(32)\nDense(16) -> Dense({model_info['horizon']})",
            language="text",
        )
        st.caption(f"seq_len={model_info['seq_len']}  ·  horizon={model_info['horizon']} steps")
    else:
        st.code(
            "LSTM(64) -> Drop(0.2)\nLSTM(32) -> Attention(32)\nDense(16) -> Dense(horizon)",
            language="text",
        )
        st.caption("model not trained yet")
    st.markdown(
        f"""
        <div style="background:{CARD_YELLOW};color:{INK};padding:10px 14px;
                    border-radius:10px;font-size:0.84rem;font-weight:600;
                    font-family:Inter,sans-serif;">
            mae target ≤ 50 ns (step 1)
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{API_BASE_URL} · live")

# ── Telemetry ────────────────────────────────────────────────────────────────
tele_status, tele_body = api_get("/telemetry", limit=telemetry_limit)

if tele_status is None:
    api_unreachable_banner(tele_body)
    st.stop()
elif tele_status != 200:
    st.error(f"telemetry fetch failed: {tele_body}")
    st.stop()

df = pd.DataFrame(tele_body["rows"])
n_total = tele_body["n_rows"]

if "timestamp_s" in df.columns:
    elapsed_h = df["timestamp_s"].iloc[-1] / 3600
    month_label = f"{elapsed_h:.0f}h track"
else:
    month_label = "navic"

# ── Metric cards ───────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(
        metric_card(f"#{n_total}", month_label, CARD_YELLOW, str(n_total),
                    "telemetry samples", "live from naviguard api"),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        metric_card("15m", "interval", CARD_SKY, "15<span style='font-size:1rem;opacity:0.5'> min</span>",
                    "sampling rate", "900 second steps"),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        metric_card("3×", "features", CARD_PINK, "3",
                    "input features", "bias · drift · ephemeris"),
        unsafe_allow_html=True,
    )
with m4:
    horizon_val = model_info["horizon"] if model_info else "—"
    st.markdown(
        metric_card(str(horizon_val), "steps", CARD_LAVENDER, str(horizon_val),
                    "forecast horizon", "attention-lstm output"),
        unsafe_allow_html=True,
    )

# ── Raw table ─────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown('<div class="section-label">latest telemetry</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background:{CARD_CREAM};border-radius:16px 16px 0 0;padding:14px 18px 0;">
            <div class="card-meta" style="color:{INK};">
                <span>#samples</span><span>{month_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    fmt = {}
    if "clock_bias_s" in df.columns:
        fmt["clock_bias_s"] = "{:.4e}"
    if "clock_drift_s_per_s" in df.columns:
        fmt["clock_drift_s_per_s"] = "{:.4e}"
    if "ephemeris_error_m" in df.columns:
        fmt["ephemeris_error_m"] = "{:.4f}"
    st.dataframe(df.tail(12).style.format(fmt), width="stretch", height=260)

# ── Time-series ────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">time-series</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f'<div class="chart-wrap" style="background:{CARD_SKY};color:{INK};">'
        f'<div class="card-meta"><span>live</span><span>navic</span></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(pudding_plot(df["clock_bias_s"] * 1e6, "μs", CARD_SKY), width="stretch")
    st.markdown(
        f"""
        <div class="chart-footer" style="color:{INK};">
            <div class="card-title">clock bias</div>
            <div class="card-sub">satellite oscillator offset over time</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="chart-wrap" style="background:{CARD_YELLOW};color:{INK};">'
        f'<div class="card-meta"><span>live</span><span>navic</span></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(pudding_plot(df["ephemeris_error_m"], "m", CARD_YELLOW), width="stretch")
    st.markdown(
        f"""
        <div class="chart-footer" style="color:{INK};">
            <div class="card-title">ephemeris error</div>
            <div class="card-sub">orbital position drift in meters</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

if show_drift:
    st.markdown(
        f'<div class="chart-wrap" style="background:{CARD_MINT};color:{INK};margin-top:8px;">'
        f'<div class="card-meta"><span>live</span><span>drift</span></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(pudding_plot(df["clock_drift_s_per_s"] * 1e10, "×10⁻¹⁰ s/s", CARD_MINT), width="stretch")
    st.markdown(
        f"""
        <div class="chart-footer" style="color:{INK};">
            <div class="card-title">clock drift</div>
            <div class="card-sub">rate of change in clock bias</div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )

# ── Prediction ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">attention-lstm prediction</div>', unsafe_allow_html=True)

eval_status, eval_body = api_get("/predict/evaluate")

if eval_status == 200:
    step1_actual = [row[0] / 1e3 for row in eval_body["actual_ns"]]      # ns -> us
    step1_predicted = [row[0] / 1e3 for row in eval_body["predicted_ns"]]
    mae_ns = eval_body["mae_ns"][0]
    rmse_ns = eval_body["rmse_ns"][0]
    passed = eval_body["pass_step1"]

    st.markdown(
        f"""
        <div style="background:{CARD_LAVENDER};border-radius:16px;padding:18px 20px 8px;color:{INK};">
            <div class="card-meta">
                <span>step 1 / {eval_body['horizon']}</span>
                <span>{eval_body['n_test_samples']} test samples</span>
            </div>
        """,
        unsafe_allow_html=True,
    )
    st.pyplot(pudding_compare_plot(step1_actual, step1_predicted, "μs", CARD_LAVENDER), width="stretch")
    pill_bg = CARD_MINT if passed else CARD_PINK
    st.markdown(
        f"""
        <div style="color:{INK};padding:4px 4px 14px;">
            <div class="card-title">actual vs predicted</div>
            <div class="card-sub">mae {mae_ns:.2f} ns · rmse {rmse_ns:.2f} ns
                <span class="status-pill" style="background:{pill_bg};margin-left:8px;">
                    {'within target' if passed else 'exceeds target'}
                </span>
            </div>
        </div></div>
        """,
        unsafe_allow_html=True,
    )
elif eval_status == 503:
    model_not_trained_banner(eval_body.get("message", ""))
elif eval_status is None:
    api_unreachable_banner(eval_body)
else:
    st.error(f"evaluation fetch failed: {eval_body}")

# ── Anomaly report (local LLM via LM Studio) ────────────────────────────────────
st.markdown('<div class="section-label">anomaly report</div>', unsafe_allow_html=True)

if eval_status == 200:
    with st.spinner("running numeric check" + (" + local llm reasoning..." if include_llm else "...")):
        report_status, report_body = api_post(
            "/anomaly-report", {"include_llm": include_llm}, timeout=ANOMALY_REPORT_TIMEOUT_S
        )

    if report_status == 200:
        llm_status = report_body["llm_status"]
        breach = report_body["threshold_breach"]
        pill_bg = CARD_PINK if breach else CARD_MINT

        st.markdown(
            f"""
            <div style="background:{CARD_CREAM};border-radius:16px;padding:18px 22px;color:{INK};">
                <div class="card-meta">
                    <span>numeric check</span>
                    <span class="status-pill" style="background:{pill_bg};">
                        {'threshold breached' if breach else 'nominal'}
                    </span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        if report_body["llm_report"]:
            st.markdown(
                f'<div class="card-title">operator report</div>'
                f'<div class="card-sub" style="opacity:0.85;margin-top:6px;line-height:1.5;">'
                f'{report_body["llm_report"]}</div>',
                unsafe_allow_html=True,
            )
            if report_body["llm_severity"]:
                sev = report_body["llm_severity"]
                st.markdown(
                    f'<div class="card-sub" style="margin-top:10px;">'
                    f'<b>llm severity:</b> {sev.get("severity", "unknown")} — {sev.get("reasoning", "")}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="card-sub" style="opacity:0.7;margin-top:6px;">'
                f'llm report unavailable — {llm_status}</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    elif report_status is None:
        api_unreachable_banner(report_body)
    else:
        st.error(f"anomaly report fetch failed: {report_body}")
else:
    st.caption("anomaly report needs a trained model first.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;font-size:0.76rem;color:#666;margin-top:36px;
            font-family:Inter,sans-serif;line-height:1.6;">
    naviguard · navic/irnss · attention-lstm(64→32) · minmax scaler · dept. of ai & ds, bmsce 2025–26
</div>
""",
    unsafe_allow_html=True,
)
