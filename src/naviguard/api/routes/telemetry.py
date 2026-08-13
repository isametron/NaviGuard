"""naviguard.api.routes.telemetry — raw telemetry read endpoint for frontends/dashboards."""

import pandas as pd
from fastapi import APIRouter, Query

from naviguard.config import TELEMETRY_CSV
from naviguard.api.schemas import TelemetryResponse

router = APIRouter()


@router.get("/telemetry", response_model=TelemetryResponse)
def get_telemetry(limit: int = Query(50, ge=1, le=1000)) -> TelemetryResponse:
    """Latest `limit` rows of raw telemetry, for frontends that want to chart
    or table the source data without reading data/*.csv directly."""
    df = pd.read_csv(TELEMETRY_CSV)
    tail = df.tail(limit)
    return TelemetryResponse(
        n_rows=len(df),
        columns=list(tail.columns),
        rows=tail.to_dict(orient="records"),
    )
