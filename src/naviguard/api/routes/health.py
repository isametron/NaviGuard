"""naviguard.api.routes.health — liveness + model-info endpoints."""

import json
import os

from fastapi import APIRouter, Depends, Query

from naviguard.api.schemas import HealthResponse, ModelInfoResponse
from naviguard.api.security import require_api_key
from naviguard.config import (
    META_PATH, MODEL_PATH, NAVIC_META_PATH, NAVIC_TELEMETRY_CSV, SCALER_PATH, TELEMETRY_CSV, LLMSettings,
    active_profile,
)
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
    navic = active_profile() == "navic"
    model_loaded = os.path.exists(NAVIC_META_PATH) if navic else os.path.exists(MODEL_PATH)
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        scaler_loaded=model_loaded if navic else os.path.exists(SCALER_PATH),   # navic windows are level-free: no scaler
        llm_configured=bool(settings.base_url and settings.model),
        telemetry_available=os.path.exists(NAVIC_TELEMETRY_CSV if navic else TELEMETRY_CSV),
        profile=active_profile(),
        llm_reachable=LMStudioClient(settings).is_reachable() if check_llm else None,
    )


@router.get("/model/info", response_model=ModelInfoResponse, dependencies=[Depends(require_api_key)])
def model_info() -> ModelInfoResponse:
    meta_path = NAVIC_META_PATH if active_profile() == "navic" else META_PATH
    if not os.path.exists(meta_path):
        raise ArtifactsNotFoundError(
            f"No model metadata at {meta_path}. Run `naviguard preprocess` then `naviguard train` first "
            "(or `naviguard fetch` then `naviguard train-navic` for the navic profile)."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return ModelInfoResponse(**meta)
