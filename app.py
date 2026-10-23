"""FastAPI interface for dataset-grounded, non-medical sleep survey analysis."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from sleep_project.behavioral_assessment import InvalidAssessmentInput
from sleep_project.duration_v2 import SleepDurationAssessmentV2

BASE = Path(__file__).resolve().parent
assessor = SleepDurationAssessmentV2(BASE / "models" / "sleep_duration_multiclass_v2.json")
templates = Jinja2Templates(directory=str(BASE / "templates"))
app = FastAPI(title="Sleep Behaviour Survey Explorer", version="2.0.0")


@app.middleware("http")
async def no_personal_response_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


def page_context(request, selected=None, result=None, error=None):
    return {
        "request": request,
        "questions": assessor.questions(selected),
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
    form = await request.form()
    answers = dict(form)
    try:
        result = assessor.predict(answers)
    except InvalidAssessmentInput as exc:
        return templates.TemplateResponse(
            request=request, name="assessment.html",
            context=page_context(request, selected=answers, error=str(exc)),
            status_code=422,
        )
    return templates.TemplateResponse(
        request=request, name="assessment.html",
        context=page_context(request, selected=answers, result=result),
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
    return {"status": "ok", "task": assessor.model["task"], "version": 2}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
