# Sleep Survey Explorer V3 — End-to-End ML Project

Owner: **Neloy Pramanik Supto**  
Data: **52,610 owner-collected questionnaire records** in notebooks/final_sleep_dataDev.csv.

The FastAPI app estimates one of **four self-reported sleep-duration answer categories** (less than 4, 4–6, 6–8, and more than 8 hours). It does **not** measure sleep or diagnose diseases. V3 preserves the identical V2 trained model and test performance, adding optional questionnaire detail and descriptive comparisons.

## P0–P3 external evidence and local operations update (2026-10-24)

**Read [P0–P3 evidence, limitations and readiness ledger](docs/P0_P3_EVIDENCE_AND_READINESS.md) before citing accuracy or using this as a health assessment.** The frozen model's 86.10% **internal** questionnaire-answer accuracy drops to **64.17% accuracy**, **54.96% balanced accuracy**, and **24.54% expected calibration error** on an independent **CDC NHANES survey-source** mapping test (6,104 eligible records). The CDC outcome is itself derived from reported schedule times, so this is *not* independently measured or clinical validation. This degradation must not be hidden; the model was not retrained or retuned against NHANES.

P0 quality audit found 10 complete duplicates and 1,511 repeated six-answer profiles with conflicting reported duration labels; independent participants, consent and original data rights remain unverified. P1 independently observed health outcome is not available, so no disease prediction is enabled. P3 adds readiness checks, bearer-protected in-memory non-PII metrics, security headers, form size bounds, Docker health check and synthetic local/CI smoke tests. AWS remains disabled. **This is still a non-diagnostic portfolio/research demonstrator, not a production-accepted clinical tool.**

After starting your local app, test it in a second PowerShell terminal from the project folder:

~~~powershell
python -m scripts.smoke_test_local
~~~

Optional research audits (not required to run the app):

~~~powershell
python -m pip install pandas==2.2.3 numpy==2.3.5
python -m scripts.audit_dataset_quality
python -m scripts.evaluate_external_nhanes
~~~

For a local observability collector only, set `SLEEP_METRICS_TOKEN` to a strong secret in the deployment environment and make an authenticated request to `/metrics`. Keep that token out of the repository; `/metrics` is disabled without it. See the operations section in the evidence ledger.

## V3 — More survey questions, unchanged accuracy

The app now accepts **17 questions**: six original required answers and **eleven optional survey-context questions** (age, gender, occupation, weekend bed/wake times, sleep-onset latency, self-reported medical condition, concentration difficulty, late-night reasons, side effects, and coping strategies).

- Weekday/weekend timing shifts in **survey answer bands** (no invented hours).
- Symptom, coping, lifestyle and questionnaire-profile summaries.
- Descriptive four-category distributions for selected age, gender, occupation, and **self-reported medical condition** compared with the full collected survey.
- No individual disease prediction: the original medical answers are not verified diagnoses, and prior models for medical labels were unreliable/target-leaked.
- All optional answers are *not* V2 predictor features. The same six answers always produce the same four ML probabilities, regardless of what is entered in V3.
- Aggregate comparison cohorts below 100 records are excluded. The raw dataset remains outside the serving container.

**Accuracy stays 86.10% on the same 9,207-record internal holdout; macro-F1 stays 85.66%.** This preserves the existing result, not a guarantee of accuracy for outside populations.

For the detailed field map, privacy/provenance safeguards, and regression tests see **[V3 methods and acceptance](docs/V3_SURVEY_INSIGHTS.md)**.

## V2 model and interactive prediction (preserved)



- Four response bars, most likely answer, runner-up, and combined under-6 / 6+ model scores.
- Coefficient-level explanation of which schedule answers favour the chosen category versus runner-up.
- Self-reported sleep experiences (difficulty, breathing difficulty, leg movements) and environment comfort; **these are not additional predictions**.
- Interactive comparison of different bedtime and wake-up choices with change in category probabilities. **Associations only, not an intervention or guarantee.**
- Responsive layout, validation against exact dataset categories, Print / Save as PDF, and no persistence of user-submitted answers.

## Reproducible model evidence

Training and evaluation used the original owner-collected dataset. An exact-profile grouped 70/15/15 split gave 36,136 train, 7,267 validation, 9,207 test records. Validation selected the **bedtime and working-day wake-up** only model. The other four input questions remain descriptive.

| Four-class held-out metric | Result |
|---|---:|
| Accuracy | 86.10% |
| Balanced accuracy | 83.62% |
| Macro-F1 | 85.66% |
| Log-loss | 0.4442 |
| 10-bin expected calibration error | 0.0993 |

This is *internal survey-answer reconstruction*, not externally verified respondent independence, clinical validation, or calibrated health risk. See [V2 methods and limitations](docs/V2_ASSESSMENT.md), [earlier binary model evidence](docs/SHORT_SLEEP_ASSESSMENT.md), and [target feasibility audit](docs/TARGET_FEASIBILITY_AUDIT.md).

## Run on Windows PowerShell (Python 3.11)

1. Download the latest **main** branch ZIP and extract it.
2. In the extracted project directory run:

~~~powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-app.txt
python -m uvicorn app:app --reload
~~~

3. Open http://127.0.0.1:8000 and fill out the six questions. The detailed result and What-if section appear after submitting.

If PowerShell blocks virtualenv activation, use the executables directly:
~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
~~~

**No retraining or AWS deployment is required** to run locally. Your Python 3.11.9 works.

### Optional reproduction and tests

~~~powershell
python -m pip install pandas==2.2.3 numpy==2.3.5 scikit-learn==1.8.0 "httpx>=0.27,<1"
python scripts/train_sleep_duration_v2.py
python -m unittest discover -s tests -p "test_duration_v2.py" -v
~~~

### Optional Docker

~~~bash
docker build -t sleep-survey-v2 .
docker run --rm -p 8000:8000 sleep-survey-v2
~~~

## Safety of prior results

The old medical-category target was derived from a question also used as an input (target leakage). Its reported 97–99% numbers are **not valid proof of medical prediction**. The old endpoint POST /predict is retired (HTTP 410). Historical code and notebooks are preserved but are excluded from the V2 container runtime.

## Original pipeline (historical)

Legacy artifacts and modules under `sleep_project/components/`, `sleep_project/pipeline/`, `artifact/`, and notebooks document earlier experiments with MongoDB, model training, Docker, and AWS CI/CD. They are not evidence of verified medical performance and are deliberately excluded from the new runtime image.
