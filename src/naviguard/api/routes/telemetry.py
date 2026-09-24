"""naviguard.api.routes.telemetry — raw telemetry read endpoint for frontends/dashboards."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from naviguard.api.schemas import TelemetryResponse
from naviguard.config import TELEMETRY_CSV
from naviguard.preprocessing.sequences import load_telemetry

router = APIRouter()


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry(
    limit: int = Query(50, ge=1, le=1000),
    satellite_id: Optional[int] = Query(None, description="Filter to one satellite (multi-satellite data only)"),
) -> TelemetryResponse:
    """Latest `limit` rows of raw telemetry, for frontends that want to chart
    or table the source data without reading data/*.csv directly."""
    df = load_telemetry(TELEMETRY_CSV)   # raises TelemetryNotFoundError -> 503
    if satellite_id is not None:
        if "satellite_id" not in df.columns:
            raise HTTPException(status_code=422, detail="telemetry has no satellite_id column")
        df = df[df["satellite_id"] == satellite_id]
        if df.empty:
            raise HTTPException(status_code=404, detail=f"no rows for satellite_id {satellite_id}")
    tail = df.tail(limit)
    return TelemetryResponse(
        n_rows=len(df),
        columns=list(tail.columns),
        rows=tail.to_dict(orient="records"),
    )
