"""naviguard.config — shared paths, model constants, and service settings."""

import os
from dataclasses import dataclass
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR      = os.path.join(ROOT_DIR, "data")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR   = os.path.join(ROOT_DIR, "outputs")

TELEMETRY_CSV = os.path.join(DATA_DIR, "satellite_telemetry.csv")
SEQUENCES_PATH = os.path.join(DATA_DIR, "sequences.npz")   # X, y, split, satellite

SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH    = os.path.join(MODELS_DIR, "lstm_attention_satellite.keras")
META_PATH     = os.path.join(MODELS_DIR, "model_meta.json")
PLOT_PATH     = os.path.join(OUTPUTS_DIR, "prediction_plot.png")

# ── Data / model constants ────────────────────────────────────────────────────
FEATURES   = ["clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m"]
TARGET_COL = "clock_bias_s"
TARGET_IDX = FEATURES.index(TARGET_COL)
N_FEATURES = len(FEATURES)

SEQ_LEN      = 20
HORIZON      = 6
RANDOM_SEED  = 42
MAE_TARGET_NS = 50.0

# Chronological split, applied per satellite: first TRAIN_FRAC of rows train,
# next VAL_FRAC validate (early stopping / LR decay / anomaly calibration),
# remainder is the held-out test set that training never sees.
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

# ── Training defaults ─────────────────────────────────────────────────────────
DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 16
DEFAULT_PATIENCE   = 10


@dataclass(frozen=True)
class ModelHParams:
    """Architecture/optimizer hyperparameters (persisted in model_meta.json)."""
    lstm_units: tuple[int, int] = (64, 32)
    dropout: float = 0.2
    dense_units: int = 16
    learning_rate: float = 1e-3
    residual_skip: bool = True     # predict change from last observed bias (see LastValueSkip)


# ── Anomaly detection ─────────────────────────────────────────────────────────
ANOMALY_Z_THRESHOLD = 4.0     # robust z-score on step-1 residuals
ANOMALY_ANOMALOUS_FLAG_RATE = 0.05
ANOMALY_ANOMALOUS_Z_MULT = 2.0


# ── LLM (LM Studio) settings ─────────────────────────────────────────────────
class LLMSettings(BaseSettings):
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"          # LM Studio ignores the value; openai SDK requires non-empty
    model: str = "llama-3.2-3b-instruct"  # override via LLM_MODEL if a different model is loaded in LM Studio
    timeout_s: float = 60.0             # local inference (esp. on CPU) can genuinely take tens of seconds
    max_tokens: int = 400

    model_config = SettingsConfigDict(env_prefix="LLM_")


# ── API service settings ─────────────────────────────────────────────────────
class APISettings(BaseSettings):
    """Env-configurable (NAVIGUARD_*) service settings.

    api_key: if set, every endpoint except /health requires a matching
             `X-API-Key` header. Unset (default) = open, for local dev.
    cors_origins: browser origins allowed to call the API (JSON list in env).
    """
    api_key: Optional[str] = None
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:8501",   # streamlit
        "http://localhost:3000",   # react (CRA / next)
        "http://localhost:5173",   # vite
    ])

    model_config = SettingsConfigDict(env_prefix="NAVIGUARD_")
