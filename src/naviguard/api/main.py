"""naviguard.api.main — FastAPI application factory."""

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from naviguard.api.routes import anomaly, health, predict, telemetry
from naviguard.api.security import require_api_key
from naviguard.config import APISettings
from naviguard.errors import ArtifactsInvalidError, TelemetryNotFoundError
from naviguard.inference.artifacts import ArtifactsNotFoundError


def create_app() -> FastAPI:
    app = FastAPI(
        title="NaviGuard API",
        version="0.3.0",
        description="NavIC/GNSS satellite clock bias & ephemeris error intelligence service",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=APISettings().cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    guard = [Depends(require_api_key)]
    app.include_router(health.router)                       # /health open; /model/info guarded per-route
    app.include_router(telemetry.router, dependencies=guard)
    app.include_router(predict.router, dependencies=guard)
    app.include_router(anomaly.router, dependencies=guard)

    def _503(code: str):
        async def handler(request: Request, exc: Exception):
            return JSONResponse(status_code=503, content={"error": code, "message": str(exc)})
        return handler

    app.add_exception_handler(ArtifactsNotFoundError, _503("model_not_trained"))
    app.add_exception_handler(ArtifactsInvalidError, _503("model_artifacts_invalid"))
    app.add_exception_handler(TelemetryNotFoundError, _503("telemetry_missing"))

    return app


app = create_app()
