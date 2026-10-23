# V3 — Expanded Survey Insights Without Reducing ML Accuracy

## Outcome

The existing V2 four-category duration predictor is **frozen and unchanged** in V3. Its JSON weights, training algorithm, feature list, held-out test partition and measured model scores are unchanged. V3 introduces **11 additional optional questions** and derived/descriptive analysis beside the existing six-field survey assessment.

| Frozen duration model metric | V2 | V3 |
|---|---:|---:|
| Four-class accuracy (9,207 held-out responses) | 86.10% | **86.10%** |
| Macro-F1 | 85.66% | **85.66%** |
| Balanced accuracy | 83.62% | **83.62%** |
| Model predictors | Bedtime + workday wake | **Unchanged** |

These are internal *survey-answer reconstruction* metrics, not verified sleep measurement or clinical outcomes. Accuracy cannot be promised for arbitrary new respondents or external populations.

## Added questions — original dataset, optional

Eight single-choice questions: age group, gender, occupation, weekend bedtime, weekend wake-up time, time to fall asleep, reported pre-existing medical condition (optional), and concentration difficulty.

Three multiple-choice questions: reasons for late sleeping, self-reported side effects, and coping strategies. Their category labels come from the original survey. The source uses semicolon-delimited multi-choice answers including internal semicolons in the category **Entertainment (TV;movies;games)**; the aggregate generator detects full category labels with exact substring matching, rather than incorrectly splitting that category.

No user is required to disclose medical history. All new answers are excluded from frozen model inputs.

## New analyses

1. **Weekday vs weekend timing**: ordinal category-band comparison for bedtime and wake-up responses (e.g., one time band later). This is *not* a measured hour shift, a social-jetlag diagnosis, or a health-risk calculation.
2. **Survey-reported context**: optional respondent background, sleep onset latency, concentration difficulty, known medical history, reasons, side effects, and coping.
3. **Survey answer frequencies**: for selected reasons, side effects, or coping habits, show the fraction of respondents **who answered that question** and selected that label.
4. **Aggregated cohort comparison**: four-category duration distributions for self-reported age, gender, occupation, and medical condition against whole-survey distribution. Groups are evaluated *one attribute at a time*; **no joint sensitive subgroups** or cohorts with fewer than **100** records are exported.
5. Existing four-category predictor explanation, probability bars, What-if API and print-to-PDF remain available and unmodified in their core logic.

## Model versus descriptive evidence

- **Model result:** frozen two-feature multinomial Logistic Regression (JSON-weight serving).
- **Descriptive results:** aggregations of the original 52,610 *survey records* stored in models/survey_cohorts_v3.json and served without the raw CSV.
- **Medical values:** the original column has multiple response spellings and does not contain independently confirmed clinical diagnoses. V3 shows selected *self-reported condition* only and the corresponding duration-answer distribution. It **never infers or diagnoses a disease**, nor estimates an individual risk.
- **Privacy:** cohort groups smaller than 100 are excluded. Only aggregate counts, rounded proportions, and common survey labels are included in the Docker image; the source survey rows are not shipped. User-submitted answers are not persisted by this app, and its responses use Cache-Control: no-store.
- **No causal evidence:** changing bedtime in the What-if simulator substitutes survey input values in the original trained model. It is not a predicted treatment effect.

## Reproducibility and acceptance

Regenerate aggregated statistics after installing pandas==2.2.3:

    python scripts/build_survey_cohorts_v3.py

Then run:

    python -m unittest discover -s tests -p "test_short_sleep.py" -v
    python -m unittest discover -s tests -p "test_duration_v2.py" -v
    python -m unittest discover -s tests -p "test_survey_insights_v3.py" -v

V3 tests assert:
- Fixed V2 model architecture and exact 0.860975 test accuracy, 0.856607 macro-F1.
- The optional medical and survey answers do not enter prediction input and cannot change outputs for the same six core answers.
- Aggregate counts and percentages reproduce from the owner-collected CSV.
- Every exported cohort has at least 100 records, with no raw rows or combined sensitive groups.
- The 17-field form stays compatible with old six-answer requests; malformed/unseen answers fail validation.
- Existing What-if and retired medical endpoint continue to behave as before.

The main predictor artifact is intentionally NOT retrained because extended source questions did not beat the simpler model on the preregistered validation log-loss. See docs/V2_ASSESSMENT.md for measured model selection and test details, and docs/TARGET_FEASIBILITY_AUDIT.md for the historical medical-label leakage and weak alternative outcomes.

## Local run

Python 3.11.9 is supported for inference. No AWS deployment or training is needed.

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements-app.txt
    python -m uvicorn app:app --reload

Open http://127.0.0.1:8000 and fill the six required questions; optional V3 fields add richer descriptive insights. The app still reports 86.10% **internal test accuracy**, unchanged from the frozen V2 model.
