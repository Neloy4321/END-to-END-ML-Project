# P1 — Meaningful prediction eligibility gate (2026-10-24)

**Status: reproducible feasibility screen, not a new medically validated model.**
This step deliberately preserves the working local V2/V3 assessment and the
frozen 86.10% *internal* self-reported sleep-duration model. No API prediction
card, weights, deploy steps, or AWS settings are changed.

## Why a new disease predictor was not shipped

The 20-item owner-collected survey is cross-sectional and self-reported. It has no
independently established clinical diagnosis, validated sleep-scale total, later
health outcome, objective sleep measurement, participant ID, or collection-site field.
The old medical-category label was constructed from the medical-condition answer
included among its own predictors: fixing this requires genuine new outcomes,
not another tuning pass.

The existing [target feasibility audit](TARGET_FEASIBILITY_AUDIT.md) found
roughly chance-level discrimination of concentration difficulty, sleep-onset
difficulty, breathing difficulty, and fatigue, whereas duration was classifiable
largely from related sleep/wake schedule answers. A second evaluation on this
same source does *not* become an untouched independent confirmatory test.

## Executable P1 gate

From the repository root with pandas, NumPy and scikit-learn installed:

~~~powershell
py -3.11 -m scripts.screen_p1_targets --output p1-local-results.json
py -3.11 -m unittest discover -s tests -p "test_p1_target_screen.py" -v
~~~

The JSON contains **aggregate metrics only, no participant-level rows**.
CI runs the same screen against the committed survey file.

Five outcomes are evaluated, always named **self-reported**:

1. Frequent concentration difficulty.
2. Frequent difficulty falling asleep.
3. Frequent breathing difficulty during sleep.
4. Fatigue selected among side effects.
5. Academic/work difficulty selected among side effects.

All share eight allowed predictors: age band, gender, occupation,
weekday/weekend bedtime and waking-time bands, and rated sleep environment.
Medical-condition answers, symptom answers, side effects and coping responses
are excluded from all predictors.

- Exact identical entire-response records are conservatively excluded;
  they are **not claimed to be proven repeat participants**.
- The approximately 70/15/15 seeded split groups matching eight-answer
  predictor profiles into train, validation, or test. This checks exact
  profile overlap, not person/time/geography independence.
- A prior-only dummy, one-hot logistic regression, and capped random forest
  are compared on validation log loss.
- Imputation of categorical missing values uses a fixed missing category;
  feature encoding/model fitting uses training rows only. The candidate
  and its 0.50 classification threshold are fixed before the final test.
- Selected-vs-prior results include accuracy, balanced accuracy, macro-F1,
  ROC AUC, log loss, Brier score, and bootstrap AUC 95% interval.
  Raw accuracy alone does not earn acceptance.
- The **internal exploratory screen** requires: non-prior selection, test
  AUC interval lower bound at least 0.60, balanced accuracy at least 0.60,
  and test log-loss gain over prior of at least 0.02. These are arbitrary
  engineering screen preferences and **not** clinical acceptance thresholds.

Even if an internal candidate meets these conditions, this gate explicitly
returns medical_or_clinical_validation=false, externally_validated=false,
and eligible_for_app_model_replacement=false.
Never display its model outputs as disease probability, clinical screening,
sleep-disorder risk, causal benefit, or independent population validation.

## Observed results on the owner-collected dataset

GitHub Actions PR #10, commit bbf8093, executed the new screen against the
52,610-row survey on 2026-10-24. This is **retrospective internal evidence**
using labels previously explored on the same data, not a new independent test.
ROC AUC = 0.50 means essentially chance ranking.

| Self-reported candidate | Selected test ROC AUC | Internal screen |
| --- | ---: | --- |
| Frequent concentration difficulty | 0.52181 | FAIL |
| Frequent difficulty falling asleep | 0.55208 | FAIL |
| Frequent breathing difficulty | 0.56943 | FAIL |
| Fatigue among reported effects | 0.55727 | FAIL |
| Academic/work impact among reported effects | 0.50158 | FAIL |

All five screens failed: **zero new models accepted for the app**.
These are *not* externally verified results, and the selected models must not
be used as an individual symptom forecast. The original duration inference
artifact stays unchanged. Re-run the CLI to see complete class support,
balanced accuracy, log loss, prior comparison, Brier score, and AUC uncertainty.

## Outstanding genuine P1 requirements

A genuinely accepted new health prediction requires an independently
assessed outcome from a permitted validated instrument (or an appropriate
clinical/objective protocol), de-identified stable respondent identifiers,
source/time/site separation and a held-out prospective confirmatory sample.
The owner reported verbal participant and under-18 guardian permission for
academic and research purposes. This testimony is recorded, but independent
verification or public redistribution rights are not assumed.

A university final-year defense reflects academic evaluation, but does not
substitute for clinical labels, prospective measurement, or independent
review of source-level predictions.

**Go/no-go:** if no candidate earns reliable validation, keep the descriptive
survey dashboard and the V2 duration model (labeled questionnaire-response
estimate) without adding another model card.
