"""naviguard.api.main — FastAPI application factory."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from naviguard.inference.artifacts import ArtifactsNotFoundError
from naviguard.api.routes import anomaly, health, predict, telemetry


def create_app() -> FastAPI:
    app = FastAPI(
        title="NaviGuard API",
        version="0.2.0",
        description="NavIC/GNSS satellite clock bias & ephemeris error intelligence service",
    )

    app.include_router(health.router)
    app.include_router(telemetry.router)
    app.include_router(predict.router)
    app.include_router(anomaly.router)

    @app.exception_handler(ArtifactsNotFoundError)
    async def artifacts_not_found_handler(request: Request, exc: ArtifactsNotFoundError):
        return JSONResponse(
            status_code=503,
            content={"error": "model_not_trained", "message": str(exc)},
        )

    return app


app = create_app()
