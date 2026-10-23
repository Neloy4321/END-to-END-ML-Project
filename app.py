"""FastAPI interface for dataset-grounded, non-medical sleep survey analysis."""
import hmac
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from sleep_project.behavioral_assessment import InvalidAssessmentInput
from sleep_project.duration_v2 import SleepDurationAssessmentV2
from sleep_project.operations import RequestCounters
from sleep_project.survey_insights_v3 import SurveyInsightsV3

BASE = Path(__file__).resolve().parent
assessor = SleepDurationAssessmentV2(BASE / "models" / "sleep_duration_multiclass_v2.json")
survey_insights = SurveyInsightsV3(BASE / "models" / "survey_cohorts_v3.json")
templates = Jinja2Templates(directory=str(BASE / "templates"))
app = FastAPI(title="Sleep Behaviour Survey Explorer", version="3.0.0")
request_counters = RequestCounters()
MAX_FORM_BYTES = 16384


@app.middleware("http")
async def no_personal_response_cache(request, call_next):
    started = time.monotonic()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Request-ID"] = uuid.uuid4().hex
        return response
    finally:
        request_counters.observe(request.url.path, status, time.monotonic() - started)


def page_context(request, selected=None, result=None, error=None,
                 selected_extra=None, insight_report=None):
    return {
        "request": request,
        "questions": assessor.questions(selected),
        "survey_sections": survey_insights.sections(selected_extra),
        "insight_report": insight_report,
        "result": result,
        "error": error,
        "holdout": assessor.model["evaluation"]["held_out_test"],
        "sample_size": assessor.model["evaluation"]["total_records"],
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request, name="assessment.html", context=page_context(request)
    )


@app.post("/assess", response_class=HTMLResponse)
async def assess(request: Request):
    if len(await request.body()) > MAX_FORM_BYTES:
        raise HTTPException(status_code=413, detail="Survey submission is too large.")
    form = await request.form()
    answers = dict(form)
    try:
        result = assessor.predict(answers)
        extra_answers = survey_insights.parse(form)
        insight_report = survey_insights.report(result["chosen"], extra_answers)
    except InvalidAssessmentInput as exc:
        return templates.TemplateResponse(
            request=request, name="assessment.html",
            context=page_context(
                request, selected=answers, error=str(exc), selected_extra=form
            ),
            status_code=422,
        )
    return templates.TemplateResponse(
        request=request, name="assessment.html",
        context=page_context(
            request, selected=answers, result=result,
            selected_extra=extra_answers, insight_report=insight_report
        ),
    )


@app.post("/api/compare")
async def compare(request: Request):
    if len(await request.body()) > 4096:
        raise HTTPException(status_code=413, detail="Request too large.")
    try:
        payload = await request.json()
    except ValueError:
        raise HTTPException(status_code=400, detail="Malformed JSON.")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Expected a JSON object.")
    try:
        return assessor.compare(
            payload.get("answers"),
            payload.get("bed_time"),
            payload.get("wake_time"),
        )
    except InvalidAssessmentInput as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict")
async def obsolete_medical_prediction():
    raise HTTPException(
        status_code=410,
        detail="The unsupported medical-condition endpoint was retired. Use /assess.",
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "task": assessor.model["task"],
            "version": 2, "dashboard_version": 3}


@app.get("/readyz")
async def readyz():
    """Fail-fast artifact validation occurs at import; no user input needed."""
    return {"status": "ready", "model_schema_version": assessor.model["schema_version"],
            "cohort_schema_version": survey_insights.data["schema_version"]}


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics(request: Request):
    """Disabled until a bearer token is configured; no PII/answer logging."""
    token = os.environ.get("SLEEP_METRICS_TOKEN", "")
    if not token:
        raise HTTPException(status_code=404, detail="Not found")
    incoming = request.headers.get("Authorization", "")
    if not hmac.compare_digest(incoming, "Bearer " + token):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return PlainTextResponse(request_counters.exposition(), media_type="text/plain; version=0.0.4")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
