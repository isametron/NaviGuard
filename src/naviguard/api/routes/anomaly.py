"""naviguard.api.routes.anomaly — numeric evaluation + optional local-LLM report/severity."""

from fastapi import APIRouter, Depends

from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.inference.predict import evaluate_on_test
from naviguard.llm.client import LMStudioUnavailableError
from naviguard.llm.reports import assess_anomaly_severity, generate_operator_report
from naviguard.api.schemas import AnomalyReportRequest, AnomalyReportResponse, EvaluateResponse

router = APIRouter()


@router.post("/anomaly-report", response_model=AnomalyReportResponse)
def anomaly_report(
    req: AnomalyReportRequest = AnomalyReportRequest(),
    artifacts: Artifacts = Depends(get_artifacts),
) -> AnomalyReportResponse:
    """Runs the numeric evaluation + MAE-threshold check unconditionally, then
    optionally layers a local-LLM narrative report + severity second-opinion
    on top. Always returns 200: if LM Studio isn't running, the numeric
    result is still returned with llm_report=None and an explanatory
    llm_status, instead of failing the whole request."""
    result = evaluate_on_test(artifacts)
    threshold_breach = not result["pass_step1"]

    llm_report = None
    llm_severity = None
    llm_status = "not requested"

    if req.include_llm:
        try:
            llm_report = generate_operator_report(result)
            llm_severity = assess_anomaly_severity(result)
            llm_status = "ok"
        except LMStudioUnavailableError as e:
            llm_status = str(e)

    return AnomalyReportResponse(
        evaluation=EvaluateResponse(**result),
        threshold_breach=threshold_breach,
        llm_report=llm_report,
        llm_severity=llm_severity,
        llm_status=llm_status,
    )
