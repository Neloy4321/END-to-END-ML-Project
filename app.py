"""Non-diagnostic questionnaire-answer prediction backed by Neloy's survey."""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from sleep_project.behavioral_assessment import (
    BehavioralSleepAssessment,
    InvalidAssessmentInput,
)

BASE = Path(__file__).resolve().parent
assessor = BehavioralSleepAssessment(BASE / "models" / "short_sleep_logistic.json")
templates = Jinja2Templates(directory=str(BASE / "templates"))
app = FastAPI(title="Sleep Behaviour Survey Assessment", version="1.0.0")


def page_context(request, selected=None, result=None, error=None):
    return {
        "request": request,
        "questions": assessor.questions(selected),
        "result": result,
        "error": error,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "assessment.html", page_context(request)
    )


@app.post("/assess", response_class=HTMLResponse)
async def assess(request: Request):
    form = await request.form()
    submitted = dict(form)
    try:
        result = assessor.predict(submitted)
    except InvalidAssessmentInput as exc:
        return templates.TemplateResponse(
            "assessment.html",
            page_context(request, selected=submitted, error=str(exc)),
            status_code=422,
        )
    return templates.TemplateResponse(
        "assessment.html",
        page_context(request, selected=submitted, result=result),
    )


@app.post("/predict")
async def obsolete_medical_prediction():
    # Do not serve the old, target-leaked medical-condition classifier.
    raise HTTPException(
        status_code=410,
        detail="The unsupported medical-condition endpoint was retired. Use /assess.",
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "task": assessor.model["task"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
