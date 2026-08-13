"""naviguard.api.schemas — request/response models for the FastAPI service."""

from typing import Optional

from pydantic import BaseModel


class TelemetryResponse(BaseModel):
    n_rows: int
    columns: list[str]
    rows: list[dict]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    llm_configured: bool


class ModelInfoResponse(BaseModel):
    seq_len: int
    horizon: int
    features: list[str]
    target: str
    trained_at: str
    best_val_loss: float
    epochs_run: int
    n_train: int
    n_val: int


class EvaluateResponse(BaseModel):
    n_test_samples: int
    horizon: int
    target_ns: float
    mae_ns: list[float]
    rmse_ns: list[float]
    pass_step1: bool
    actual_ns: list[list[float]]
    predicted_ns: list[list[float]]
    residual_ns: list[list[float]]


class PredictRequest(BaseModel):
    window: Optional[list[list[float]]] = None


class PredictResponse(BaseModel):
    horizon: int
    predicted_clock_bias_ns: list[float]
    generated_at: str


class AnomalyReportRequest(BaseModel):
    include_llm: bool = True


class AnomalyReportResponse(BaseModel):
    evaluation: EvaluateResponse
    threshold_breach: bool
    llm_report: Optional[str] = None
    llm_severity: Optional[dict] = None
    llm_status: str
