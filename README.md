# Sleep Behaviour Self-Assessment — End-to-End ML Project

Owner: **Neloy Pramanik Supto**  
Source: the owner's collected sleep survey, 52,610 questionnaire rows in `notebooks/final_sleep_dataDev.csv`.

## Current honest prediction task

A six-input model predicts which **self-reported sleep-duration category** a respondent is more likely to select: less than 4 / 4–6 hours vs 6–8 / over 8 hours. This is **not a clinical sleep-health score**, sleep disorder detector, diagnosis, or objective measurement of duration.

The original repository's medical-category target was generated from a question that also appeared as a feature, causing target leakage. The old model and notebooks remain for historical reproducibility; the new application does not serve them. `POST /predict` is retired with HTTP 410.

## Main evidence

Internally tested on the owner's CSV with a seeded profile-group train/holdout split (no exact matching six-input survey profile on both sides):

| Metric | New model | Bed/wake-only comparator |
|---|---:|---:|
| Accuracy | 93.62% | 93.41% |
| Balanced accuracy | 93.79% | 93.72% |
| ROC AUC | 0.9748 | 0.9741 |
| Positive recall | 91.39% | 89.34% |

This is an **internal response-reconstruction result**, not external-generalization acceptance. Bedtime and wake time dominate; the gain from additional answers is small. Respondent identities, independent sites, externally measured sleep, and clinical outcomes were not verified. Model probabilities are not clinically calibrated.

For validation methodology, confusion matrices, limitations, and instructions, see [Step 3 evidence](docs/SHORT_SLEEP_ASSESSMENT.md) and [Step 2 target feasibility audit](docs/TARGET_FEASIBILITY_AUDIT.md).

## New application

- FastAPI questionnaire with six dataset-matched dropdowns, backend option validation, and clear non-medical language
- Reproducible Logistic Regression with one-hot encoding fitted only on training data
- Profile-group holdout with simple schedule-only comparator
- JSON-exported weights and pure-Python serving; scoring parity asserted against scikit-learn on holdout
- Automated contract/leakage checks and full training reproducibility test
- Lightweight non-root Docker runtime containing **no raw survey CSV or legacy pickle**
- No permissive wildcard CORS and no questionnaire-answer logging or storage in the new application

### Run the app

Requires Python 3.12.

```bash
pip install -r requirements-app.txt
uvicorn app:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Retrain with the same dataset

```bash
pip install pandas==2.2.3 numpy==2.3.5 scikit-learn==1.8.0
python scripts/train_short_sleep.py --data notebooks/final_sleep_dataDev.csv
python -m unittest discover -s tests -p "test_short_sleep.py" -v
```

The model is committed as readable `models/short_sleep_logistic.json`. A separate CI workflow runs tests and validates container build before review.

### Docker

```bash
docker build -t sleep-behaviour .
docker run --rm -p 8000:8000 sleep-behaviour
```

## Project status and next steps

This feature is proposed on a **draft branch/PR** and is not verified as deployed or validated for healthcare use. Remaining production steps: deployment smoke test, independent reviewer, privacy/collection documentation, tests with independently sourced respondent groups, and fuller probability-calibration analysis. Do not promote the legacy 97–99% medical-category figures.

## Original pipeline (historical)

Legacy artifacts and modules under `sleep_project/components/`, `sleep_project/pipeline/`, `artifact/`, and notebooks document earlier experiments with MongoDB, model training, Docker, and AWS CI/CD. They are not evidence of verified medical performance and are deliberately excluded from the new runtime image.
