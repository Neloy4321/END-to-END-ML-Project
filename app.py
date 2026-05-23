from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
from uvicorn import run as app_run

from sleep_project.constants import APP_HOST, APP_PORT

from sleep_project.pipeline.prediction_pipeline import (
    SleepData,
    SleepClassifier
)

from sleep_project.entity.estimator import (
    TargetValueMapping
)


# =========================
# FASTAPI APP
# =========================
app = FastAPI()


# =========================
# STATIC FILES
# =========================
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================
# TEMPLATE
# =========================
templates = Jinja2Templates(
    directory="templates"
)


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HOME
# =========================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):

    return templates.TemplateResponse(
        "sleep.html",
        {
            "request": request,
            "context": "Fill the form and click Predict"
        }
    )


# =========================
# PREDICT
# =========================
@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request):

    try:

        form = await request.form()

        print("\n========== FORM DATA ==========")
        print(dict(form))

        # =========================
        # CREATE DATA OBJECT
        # =========================
        sleep_data = SleepData(

            age=form.get("age"),

            weight=form.get("weight"),

            height=form.get("height"),

            gender=form.get("gender"),

            occupation=form.get("occupation"),

            bed_time=form.get("bed_time"),

            wake_time=form.get("wake_time"),

            sleep_hours=form.get("sleep_hours"),

            difficulty_sleep=form.get(
                "difficulty_sleep"
            ),

            breathing_problem=form.get(
                "breathing_problem"
            ),

            restless_legs=form.get(
                "restless_legs"
            ),

            concentration_problem=form.get(
                "concentration_problem"
            ),

            sleep_environment=form.get(
                "sleep_environment"
            ),

            # =========================
            # NEW FIELDS
            # =========================
            sleep_reason=form.get(
                "sleep_reason"
            ),

            medical_condition=form.get(
                "medical_condition"
            ),

            coping_strategy=form.get(
                "coping_strategy"
            ),
        )

        # =========================
        # DATAFRAME
        # =========================
        sleep_df = (
            sleep_data
            .get_sleep_input_data_frame()
        )

        print("\n========== INPUT DATAFRAME ==========")
        print(sleep_df)

        # =========================
        # LOAD MODEL
        # =========================
        model_predictor = SleepClassifier()

        # =========================
        # PREDICT
        # =========================
        prediction = model_predictor.predict(
            dataframe=sleep_df
        )[0]

        print("\n========== RAW PREDICTION ==========")
        print(prediction)

        # =========================
        # LABEL MAPPING
        # =========================
        mapping = (
            TargetValueMapping()
            .reverse_mapping()
        )

        status = mapping.get(
            int(prediction),
            "Unknown"
        )

        print("\n========== FINAL STATUS ==========")
        print(status)

        # =========================
        # RETURN RESULT
        # =========================
        return templates.TemplateResponse(
            "sleep.html",
            {
                "request": request,
                "context": f"Prediction: {status}"
            }
        )

    except Exception as e:

        print("\n========== ERROR ==========")
        print(str(e))

        return templates.TemplateResponse(
            "sleep.html",
            {
                "request": request,
                "context": f"Error: {str(e)}"
            }
        )


# =========================
# RUN
# =========================
if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )