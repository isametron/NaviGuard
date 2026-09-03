"""naviguard.config — shared paths, model constants, and LLM settings."""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR      = os.path.join(ROOT_DIR, "data")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR   = os.path.join(ROOT_DIR, "outputs")

TELEMETRY_CSV = os.path.join(DATA_DIR, "satellite_telemetry.csv")
X_SEQ_PATH    = os.path.join(DATA_DIR, "X_seq.npy")
Y_SEQ_PATH    = os.path.join(DATA_DIR, "y_seq.npy")

SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH    = os.path.join(MODELS_DIR, "lstm_attention_satellite.keras")
META_PATH     = os.path.join(MODELS_DIR, "model_meta.json")
PLOT_PATH     = os.path.join(OUTPUTS_DIR, "prediction_plot.png")

# ── Model constants ───────────────────────────────────────────────────────────
FEATURES   = ["clock_bias_s", "clock_drift_s_per_s", "ephemeris_error_m"]
TARGET_COL = "clock_bias_s"
TARGET_IDX = FEATURES.index(TARGET_COL)
N_FEATURES = len(FEATURES)

SEQ_LEN      = 20
HORIZON      = 6
RANDOM_SEED  = 42
MAE_TARGET_NS = 50.0


# ── LLM (LM Studio) settings ─────────────────────────────────────────────────
class LLMSettings(BaseSettings):
    base_url: str = "http://localhost:1234/v1"
    api_key: str = "lm-studio"          # LM Studio ignores the value; openai SDK requires non-empty
    model: str = "llama-3.2-3b-instruct"  # override via LLM_MODEL if a different model is loaded in LM Studio
    timeout_s: float = 60.0             # local inference (esp. on CPU) can genuinely take tens of seconds
    max_tokens: int = 400

    model_config = SettingsConfigDict(env_prefix="LLM_")
