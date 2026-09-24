"""naviguard.api.routes.health — liveness + model-info endpoints."""

import json
import os

from fastapi import APIRouter, Depends, Query

from naviguard.api.schemas import HealthResponse, ModelInfoResponse
from naviguard.api.security import require_api_key
from naviguard.config import META_PATH, MODEL_PATH, SCALER_PATH, TELEMETRY_CSV, LLMSettings
from naviguard.inference.artifacts import ArtifactsNotFoundError
from naviguard.llm.client import LMStudioClient

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(
    check_llm: bool = Query(False, description="Also probe the LM Studio server (adds latency)"),
) -> HealthResponse:
    """Always returns 200 — does its own file-existence checks rather than
    depending on get_artifacts(), so it never 503s even with no trained model."""
    settings = LLMSettings()
    return HealthResponse(
        status="ok",
        model_loaded=os.path.exists(MODEL_PATH),
        scaler_loaded=os.path.exists(SCALER_PATH),
        llm_configured=bool(settings.base_url and settings.model),
        telemetry_available=os.path.exists(TELEMETRY_CSV),
        llm_reachable=LMStudioClient(settings).is_reachable() if check_llm else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse, dependencies=[Depends(require_api_key)])
def model_info() -> ModelInfoResponse:
    if not os.path.exists(META_PATH):
        raise ArtifactsNotFoundError(
            f"No model metadata at {META_PATH}. Run `naviguard preprocess` then `naviguard train` first."
        )
    with open(META_PATH) as f:
        meta = json.load(f)
    return ModelInfoResponse(**meta)
