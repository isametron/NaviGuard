"""naviguard.api.routes.anomaly — residual anomaly detection + optional local-LLM report/severity."""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException

from naviguard.api.schemas import AnomalyReportRequest, AnomalyReportResponse, EvaluateResponse
from naviguard.inference.artifacts import Artifacts, get_artifacts
from naviguard.inference.predict import detect_anomalies, evaluate_on_test
from naviguard.llm.client import LMStudioUnavailableError
from naviguard.llm.reports import assess_anomaly_severity, generate_operator_report

router = APIRouter()


@router.post("/anomaly-report", response_model=AnomalyReportResponse)
def anomaly_report(
    req: AnomalyReportRequest = AnomalyReportRequest(),
    artifacts: Artifacts = Depends(get_artifacts),
) -> AnomalyReportResponse:
    """Runs the numeric evaluation, the MAE-threshold check and the residual
    anomaly detector unconditionally (deterministic severity), then optionally
    layers a local-LLM narrative report + severity second-opinion on top.
    Always returns 200: if LM Studio isn't running, the numeric result is
    still returned with llm_report=None and an explanatory llm_status."""
    try:
        result = evaluate_on_test(artifacts, req.satellite_id)
        detection = detect_anomalies(artifacts, req.z_threshold, req.satellite_id)
    except ValueError as e:                      # e.g. unknown satellite_id
        raise HTTPException(status_code=422, detail=str(e)) from e
    threshold_breach = not result["pass_step1"]

    llm_report = None
    llm_severity = None
    llm_status = "not requested"

    if req.include_llm:
        # The LLM is told about the detector's verdict but never asked to
        # compute anything; the two calls are independent, so run them together.
        stats = {**result, "detection": detection}
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                report_f = pool.submit(generate_operator_report, stats)
                severity_f = pool.submit(assess_anomaly_severity, stats)
                llm_report = report_f.result()
                llm_severity = severity_f.result()
            llm_status = "ok"
        except LMStudioUnavailableError as e:
            llm_report, llm_severity = None, None
            llm_status = str(e)

    return AnomalyReportResponse(
        evaluation=EvaluateResponse(**result),
        threshold_breach=threshold_breach,
        detection=detection,
        llm_report=llm_report,
        llm_severity=llm_severity,
        llm_status=llm_status,
    )
