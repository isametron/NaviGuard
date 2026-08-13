"""naviguard.api.routes.predict — evaluation and forecasting endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.inference.predict import evaluate_on_test, forecast
from naviguard.api.schemas import EvaluateResponse, PredictRequest, PredictResponse

router = APIRouter()


@router.get("/predict/evaluate", response_model=EvaluateResponse)
def predict_evaluate(artifacts: Artifacts = Depends(get_artifacts)) -> EvaluateResponse:
    """JSON form of the original PNG-only evaluation: per-horizon-step MAE/RMSE
    plus the raw actual/predicted/residual series, all in nanoseconds."""
    result = evaluate_on_test(artifacts)
    return EvaluateResponse(**result)


@router.post("/predict", response_model=PredictResponse)
def predict_forecast(
    req: PredictRequest = PredictRequest(), artifacts: Artifacts = Depends(get_artifacts)
) -> PredictResponse:
    """Forecast `horizon` steps ahead. If `window` is omitted, uses the
    latest seq_len rows from the telemetry CSV."""
    try:
        result = forecast(window=req.window, artifacts=artifacts)
    except ValueError as e:
        # Malformed client input (wrong window shape) — not a server error.
        raise HTTPException(status_code=422, detail=str(e)) from e
    return PredictResponse(
        horizon=result["horizon"],
        predicted_clock_bias_ns=result["predicted_clock_bias_ns"],
        generated_at=datetime.now(timezone.utc).isoformat(),
    )
