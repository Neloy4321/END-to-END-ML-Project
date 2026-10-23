# Step 3 — Owner-collected survey: short-sleep response model

**Dataset:** `notebooks/final_sleep_dataDev.csv`, 52,610 survey records, collected by the repository owner (per owner, 2026-10-23). Collection protocol, consent to redistribute, participant independence and external-site provenance have **not** been independently verified.

## What the new model actually does

The model predicts the **survey answer category**, not true measured hours slept, future risk of illness, a sleep disorder, or overall sleep health.

- Positive answer: "Less than 4 hours" or "4-6 hours".
- Negative answer: "6-8 hours" or "More than 8 hours".
- It deliberately **does not take sleep duration as an input**, nor any medical-condition answer or fabricated side-effect field.
- Input questions: working-day bedtime, working-day wake time, frequency of difficulty falling asleep, breathing difficulty, restless movements, and comfort score.
- No gender, age, weight, height, or medical-history input required.
- Form options are generated from exactly the fitted training category schema.
- Inference is a safe, lightweight JSON logistic model; no untrusted pickle deserialization and no sklearn serving-version mismatch.
- The unsupported old medical-condition prediction endpoint now returns HTTP 410.

## Holdout evidence (internal, not clinical)

Same exact-predictor-profile responses are kept together in a seeded **GroupShuffleSplit**, holding out approximately 20% of *profiles*, not necessarily 20% of respondents. The model is fitted only on the training side. Its one-hot encoder is fitted inside the training pipeline. No SMOTE, feature selection across splits, or target-derived input variables.

| Measure | 6-feature logistic model | Bed/wake-only logistic baseline |
|---|---:|---:|
| Records in holdout | 11,596 | 11,596 |
| Accuracy | 0.936185 | 0.934115 |
| Balanced accuracy | 0.937851 | 0.937156 |
| ROC AUC | 0.974802 | 0.974148 |
| Recall for short-sleep response | 0.913885 | 0.893404 |
| Confusion matrix, true 0/1 | [[5189,206],[534,5667]] | [[5292,103],[661,5540]] |

Training records: 41,014; total: 52,610; distinct six-feature profiles: 5,792; positive responses: 52.3056%.

**Key limitation:** bedtime and wake-up time explain almost all of the prediction; the other four responses give only minor improvements. This is survey-response reconstruction, **not independent evidence of sleep-health risk prediction**. Exact-profile separation does not exclude the same person appearing with multiple profiles. Neither this test nor the model has independent respondent/time/site or medical ground-truth validation. The logistic score is not proven clinically calibrated.

## Reproduce and verify

Use Python 3.12 with pandas 2.2.3, NumPy 2.3.5, scikit-learn 1.8.0 for the exact reproduced numbers:

```bash
python scripts/train_short_sleep.py --data notebooks/final_sleep_dataDev.csv --output models/short_sleep_logistic.json
python -m unittest discover -s tests -p "test_short_sleep.py" -v
```

The training script writes metadata and coefficients to transparent JSON, and asserts that the served pure-Python sigmoid gives the **same probabilities** as fitted scikit-learn on every held-out record. The unit test retrains the model on the original dataset and checks the committed artifact.

## Application

```bash
pip install -r requirements-app.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

- `GET /` — accessible six-question responsive interface.
- `POST /assess` — validated questionnaire-response classification.
- `POST /predict` — HTTP 410; retired because the earlier medical target leaked.
- `GET /healthz` — lightweight readiness check.

Questionnaire answers are processed for an individual request and not stored by the new application. Deployment and independent human assessment remain pending; changes are in a draft branch and **not on main**.

## Legacy warning

Legacy training code, notebooks, and old artifacts are kept for reproducibility, but the medical category model is **not used** by the new app and its former accuracy must not be advertised. The README has been updated to distinguish the legacy experiments from this honest benchmark.
