"""
dashboard.py  ·  NaviGuard — Presentation Layer
─────────────────────────────────────────────────
ISRO-themed lightweight Streamlit dashboard
Primary  : #0E88D3  (ISRO Electric Blue)
Accent   : #F47216  (ISRO Pumpkin Orange)
Run      : streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NaviGuard · ISRO NavIC Clock Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── ISRO Theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0b0f1a; color: #e8edf5; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1526;
        border-right: 2px solid #0E88D3;
    }
    [data-testid="stSidebar"] * { color: #c8d8e8 !important; }

    /* Top header bar */
    .isro-header {
        background: linear-gradient(90deg, #0b1a2e 0%, #0E88D3 60%, #F47216 100%);
        padding: 16px 24px;
        border-radius: 8px;
        margin-bottom: 18px;
    }
    .isro-header h1 {
        color: #ffffff;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .isro-header p {
        color: #d0e8f8;
        font-size: 0.82rem;
        margin: 4px 0 0 0;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #0d1a2e;
        border: 1px solid #0E88D3;
        border-radius: 8px;
        padding: 10px 14px;
    }
    [data-testid="stMetricLabel"] { color: #7ab8d8 !important; font-size:0.8rem; }
    [data-testid="stMetricValue"] { color: #F47216 !important; font-weight:700; }

    /* Section headers */
    h2, h3 { color: #0E88D3 !important; }

    /* Divider */
    hr { border-color: #1a3050 !important; }

    /* Status box */
    .status-ok {
        background: #0a2010;
        border-left: 4px solid #F47216;
        padding: 8px 14px;
        border-radius: 4px;
        color: #F47216;
        font-size: 0.85rem;
        font-family: monospace;
    }

    /* Streamlit default overrides */
    .stDataFrame { border: 1px solid #1a3050 !important; }
    .stToggle label { color: #7ab8d8 !important; }
    .stSlider label { color: #7ab8d8 !important; }
    p, li { color: #c8d8e8; }
    code { background: #0d1a2e !important; color: #F47216 !important; }
</style>
""", unsafe_allow_html=True)

# ── ISRO Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="isro-header">
  <h1>🛰️ NaviGuard &nbsp;·&nbsp; AI-Driven Satellite Clock Intelligence</h1>
  <p>Indian Space Research Organisation &nbsp;·&nbsp; NavIC / IRNSS Constellation &nbsp;·&nbsp;
     LSTM Clock Bias & Ephemeris Error Prediction &nbsp;·&nbsp;
     Dept. of AI & DS, BMSCE &nbsp;·&nbsp; 2025–26</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pipeline Controls")
    st.caption("Source: `satellite_telemetry.csv`")
    st.divider()
    lookback   = st.slider("Lookback Window (steps)", 5, 40, 20)
    show_drift = st.toggle("Clock Drift Panel", value=False)
    show_raw   = st.toggle("Raw Data Table",    value=True)
    st.divider()
    st.markdown("**Model**")
    st.code(
        "LSTM(64) → Drop(0.2)\n"
        "LSTM(32) → Drop(0.2)\n"
        "Dense(16) → Dense(1)",
        language="text"
    )
    st.caption("Adam · MSE · patience=10")
    st.divider()
    st.markdown(
        "<div class=\'status-ok\'>MAE Target &nbsp; ≤ 50 ns</div>",
        unsafe_allow_html=True
    )
    st.markdown("&nbsp;")
    st.caption("localhost:8501 · Read-only")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/satellite_telemetry.csv")

df = load_data()

# ── Metrics ────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Telemetry Samples",  f"{len(df)}")
c2.metric("Sampling Interval",  "15 min")
c3.metric("Input Features",     "3")
c4.metric("Lookback Window",    f"{lookback} steps")
st.divider()

# ── Raw table ─────────────────────────────────────────────────────────────────
if show_raw:
    st.markdown("### 📡 NavIC Telemetry — Latest Samples")
    st.dataframe(
        df.tail(12).style.format({
            "clock_bias_s":        "{:.4e}",
            "clock_drift_s_per_s": "{:.4e}",
            "ephemeris_error_m":   "{:.4f}"
        }),
        use_container_width=True, height=260
    )
    st.divider()

# ── Plot helper ───────────────────────────────────────────────────────────────
ISRO_BLUE   = "#0E88D3"
ISRO_ORANGE = "#F47216"
BG          = "#0b0f1a"
GRID        = "#1a2a3a"

def isro_plot(series, ylabel, title, color):
    fig, ax = plt.subplots(figsize=(6, 2.6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.plot(series, color=color, lw=1.5)
    ax.fill_between(range(len(series)), series, alpha=0.15, color=color)
    ax.set_xlabel("Sample Index", fontsize=8, color="#7ab8d8")
    ax.set_ylabel(ylabel, fontsize=8, color="#7ab8d8")
    ax.set_title(title, fontsize=9, fontweight="bold", color="#e8edf5", pad=8)
    ax.tick_params(colors="#7ab8d8", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.grid(alpha=0.2, linestyle="--", color=GRID)
    fig.tight_layout(pad=1.0)
    return fig

# ── Time-series charts ────────────────────────────────────────────────────────
st.markdown("### 📈 Telemetry Time-Series")
col1, col2 = st.columns(2)

with col1:
    st.pyplot(
        isro_plot(df["clock_bias_s"] * 1e6,
                  "Bias (μs)", "Satellite Clock Bias", ISRO_BLUE),
        use_container_width=True
    )

with col2:
    st.pyplot(
        isro_plot(df["ephemeris_error_m"],
                  "Error (m)", "Ephemeris Error", ISRO_ORANGE),
        use_container_width=True
    )

if show_drift:
    st.pyplot(
        isro_plot(df["clock_drift_s_per_s"] * 1e10,
                  "Drift (×10⁻¹⁰ s/s)", "Clock Drift", "#8ecfee"),
        use_container_width=True
    )

st.divider()

# ── Prediction Output ─────────────────────────────────────────────────────────
st.markdown("### 🤖 LSTM Prediction Output")

plot_path = "outputs/prediction_plot.png"
if os.path.exists(plot_path):
    st.image(plot_path,
             caption="LSTM Prediction — Actual vs Predicted Clock Bias  |  Residuals (ns)",
             use_container_width=True)
else:
    st.warning("Run `python predict.py` first to generate the prediction plot.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🇮🇳 NaviGuard  ·  Indian Space Research Organisation (ISRO)  ·  "
    "NavIC/IRNSS Clock & Ephemeris Prediction  ·  "
    "LSTM(64→32) + Dropout  ·  MinMaxScaler  ·  20-step window  ·  "
    "Dept. of AI & DS, BMSCE 2025–26"
)
