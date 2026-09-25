"""naviguard.api.routes.predict — evaluation and forecasting endpoints."""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from naviguard.api.schemas import EvaluateResponse, PredictRequest, PredictResponse
from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.inference.predict import evaluate_on_test, forecast

router = APIRouter()


@router.get("/predict/evaluate", response_model=EvaluateResponse)
def predict_evaluate(
    satellite_id: Optional[int] = Query(None, description="navic profile only: evaluate one satellite"),
    artifacts: Artifacts = Depends(get_artifacts),
) -> EvaluateResponse:
    """Per-horizon-step MAE/RMSE on the untouched test split, the raw
    actual/predicted/residual series (nanoseconds), and classical-baseline
    comparison. Results are cached, so repeat calls are cheap."""
    try:
        return EvaluateResponse(**evaluate_on_test(artifacts, satellite_id))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/predict", response_model=PredictResponse)
def predict_forecast(
    req: PredictRequest = PredictRequest(), artifacts: Artifacts = Depends(get_artifacts)
) -> PredictResponse:
    """Forecast `horizon` steps ahead. If `window` is omitted, uses the
    latest seq_len rows from the telemetry CSV (optionally for one satellite)."""
    try:
        result = forecast(window=req.window, artifacts=artifacts, satellite_id=req.satellite_id)
    except ValueError as e:
        # Malformed client input (wrong window shape, unknown satellite) — not a server error.
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(
        horizon=result["horizon"],
        predicted_clock_bias_ns=result["predicted_clock_bias_ns"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
