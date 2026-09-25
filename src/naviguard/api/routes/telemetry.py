"""naviguard.api.routes.telemetry — raw telemetry read endpoint for frontends/dashboards."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from naviguard.api.schemas import TelemetryResponse
from naviguard.config import NAVIC_TELEMETRY_CSV, TELEMETRY_CSV, active_profile
from naviguard.preprocessing.sequences import load_telemetry

router = APIRouter()


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry(
    limit: int = Query(50, ge=1, le=1000),
    satellite_id: Optional[int] = Query(None, description="Filter to one satellite (multi-satellite data only)"),
) -> TelemetryResponse:
    """Latest `limit` rows of raw telemetry, for frontends that want to chart
    or table the source data without reading data/*.csv directly."""
    navic = active_profile() == "navic"
    # raises TelemetryNotFoundError -> 503; the navic profile has no ephemeris_error_m column
    df = load_telemetry(NAVIC_TELEMETRY_CSV, required=["clock_bias_s"]) if navic else load_telemetry(TELEMETRY_CSV)
    all_sats = sorted(int(s) for s in df["satellite_id"].unique()) if "satellite_id" in df.columns else None
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
        satellites=all_sats,
    )
