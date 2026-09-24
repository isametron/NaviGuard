"""naviguard.api.schemas — request/response models for the FastAPI service.

Changes since v0.2 are additive only (new optional fields / optional request
params), so existing clients keep working unchanged.
"""

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
    telemetry_available: Optional[bool] = None
    llm_reachable: Optional[bool] = None   # only probed with ?check_llm=true


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
    n_test: Optional[int] = None
    test_mae_ns: Optional[list[float]] = None
    persistence_mae_ns: Optional[list[float]] = None
    hparams: Optional[dict] = None


class BaselineMetrics(BaseModel):
    mae_ns: list[float]
    rmse_ns: list[float]


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
    split: str = "test"
    baselines: Optional[dict[str, BaselineMetrics]] = None
    skill_vs_persistence: Optional[float] = None


class PredictRequest(BaseModel):
    window: Optional[list[list[float]]] = None
    satellite_id: Optional[int] = None    # used only when `window` is omitted


class PredictResponse(BaseModel):
    horizon: int
    predicted_clock_bias_ns: list[float]
    generated_at: str


class AnomalyDetection(BaseModel):
    z_threshold: float
    center: float            # ns — median residual on the nominal (validation) reference
    scale: float             # ns — MAD-derived sigma of that reference
    n_scored: int
    n_flagged: int
    flag_rate: float
    max_abs_z: float
    severity: str            # nominal | watch | anomalous (deterministic, numeric)
    flagged_indices: list[int]
    z_scores: list[float]


class AnomalyReportRequest(BaseModel):
    include_llm: bool = True
    z_threshold: Optional[float] = None


class AnomalyReportResponse(BaseModel):
    evaluation: EvaluateResponse
    threshold_breach: bool
    detection: Optional[AnomalyDetection] = None
    llm_report: Optional[str] = None
    llm_severity: Optional[dict] = None
    llm_status: str
