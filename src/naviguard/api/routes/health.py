"""naviguard.api.routes.health — liveness + model-info endpoints."""

import json
import os

from fastapi import APIRouter

from naviguard.config import MODEL_PATH, SCALER_PATH, META_PATH, LLMSettings
from naviguard.inference.artifacts import ArtifactsNotFoundError
from naviguard.api.schemas import HealthResponse, ModelInfoResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Always returns 200 — does its own file-existence checks rather than
    depending on get_artifacts(), so it never 503s even with no trained model."""
    return HealthResponse(
        status="ok",
        model_loaded=os.path.exists(MODEL_PATH),
        scaler_loaded=os.path.exists(SCALER_PATH),
        llm_configured=LLMSettings() is not None,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    if not os.path.exists(META_PATH):
        raise ArtifactsNotFoundError(
            f"No model metadata at {META_PATH}. Run `naviguard preprocess` then `naviguard train` first."
        )
    with open(META_PATH) as f:
        meta = json.load(f)
    return ModelInfoResponse(**meta)
