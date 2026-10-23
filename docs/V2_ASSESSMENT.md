# Sleep Survey Explorer V2 — Model Evidence

Owner: Neloy Pramanik Supto. Source: notebooks/final_sleep_dataDev.csv (52,610 collected survey records).

## Scope and features

Four-way survey answer reconstruction: Less than 4 hours, 4-6 hours, 6-8 hours, More than 8 hours. The app displays category distributions, runner-up, combined under-six/6+ score, model explanation, the respondent's self-reported symptoms and sleeping-environment rating, and an interactive bedtime/wake comparison. Browser Print / Save to PDF works without a third-party report service.

The V2 model uses only usual bedtime and working-day wake-up time. The other four survey fields are validated and displayed as **self-reported context**, not used to predict health outcomes. There is **no medical classification or clinical risk scoring**.

## Provenance and no-leakage protocol

- A 70/15/15 seeded split grouped by exact six-answer profile: train 36,136 records, validation 7,267, final untouched test 9,207. There are 5,792 unique six-answer profiles. Seeds: 42, 73.
- The target is the survey's sleep-duration *answer*, and is excluded from feature columns.
- One-hot encoders and Logistic Regression were fitted only on training records. Two candidates (bed/wake-only, all six) were compared using validation log-loss, with no test selection. The simpler bed/wake-only model won (0.532056 versus 0.537093 validation log-loss).
- A small fixed temperature grid was selected on validation log-loss only. Selected temperature: 1.0.
- Frozen held-out test metrics follow. The JSON coefficients were checked against sklearn on held-out records and are reproducible from the checked-in survey CSV.

| Metric | V2 held-out result |
|---|---:|
| Accuracy (four classes) | 86.10% |
| Balanced accuracy | 83.62% |
| Macro-F1 | 85.66% |
| Multiclass log-loss | 0.4442 |
| Top-label ECE (10 bins) | 0.0993 |

Confusion matrix. Rows are actual source-survey answers; columns are predicted, in order: Less than 4, 4–6, 6–8, More than 8:

| Actual response | Less than 4 | 4–6 | 6–8 | More than 8 |
|---|---:|---:|---:|---:|
| Less than 4 | 1198 | 464 | 23 | 16 |
| 4–6 | 11 | 2424 | 431 | 20 |
| 6–8 | 16 | 38 | 3191 | 24 |
| More than 8 | 8 | 10 | 219 | 1114 |

For scientific transparency, the full-six-field comparator yielded 86.27% accuracy, 85.80% macro-F1, and 0.4442 log-loss on the same untouched test. It did not outperform the simpler deployed model on the **prespecified validation selection metric**; these test results were not used for selection.

## How to run and reproduce

With Python 3.11+ installed, create a virtual environment, install requirements-app.txt, and run python -m uvicorn app:app --reload. Open http://127.0.0.1:8000. No model retraining or AWS service is needed.

To retrain, install pandas==2.2.3 numpy==2.3.5 scikit-learn==1.8.0 and run:
- python scripts/train_sleep_duration_v2.py
- python -m unittest discover -s tests -p "test_duration_v2.py" -v

Comparison API: POST /api/compare, JSON object with "answers" (all six form keys) and "bed_time", "wake_time" replacement strings. Responds with "original", "scenario", and "short_response_change_pp". Invalid values return 422. The prior medical-condition POST /predict remains retired (410).

## Limitations

The labels and predictors are collected self-reports, not objective sleep measurement or doctor-reviewed diagnosis. The exact-profile split **does not** prove independent respondents, sites, collection periods, or external generalization; overlapping schedules and correlated answers may appear across partitions. Internal calibration is not clinical calibration (ECE ≈0.10 on the held-out test). Model probability and what-if differences describe statistical associations, **not causes or expected effects of changing a real person's bedtime**.

Form answers are processed in-memory and not persisted by the V2 app; browser caching is disabled through Cache-Control: no-store. Docker excludes the raw dataset, notebooks, and old serialized medical models.
