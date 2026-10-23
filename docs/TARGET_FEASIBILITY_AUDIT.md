# Step 2 — Sleep-health target feasibility audit

Audit date: 2026-10-23  
Repository baseline: `main` at `4537c43ca331189cc63e5bbfce0261da56b78d32`  
Input: `notebooks/final_sleep_dataDev.csv` (Git blob `06a78bd846b424c74f078c8df64dd98ed65797a3`)

## Decision

**Do not select a supervised clinical sleep-health or sleep-disorder label from this CSV.**
The file has self-reported questions, not an independently verified clinical or
validated sleep-health outcome. A model may still be used experimentally to
forecast a clearly named self-reported survey answer, but it must not be
represented as a medical diagnosis or validated health-risk assessment.

The previous `Medical Condition Category` is **not a valid independent
prediction label** for the current training features: it is computed from
`Do you have any medical conditions that might affect your sleep?`, and that
same question remains in the original predictors. The committed model
artifact also selected that predictor. Its high reported accuracy is
contaminated by target leakage.

## Dataset inspection

- 52,610 rows and 20 survey columns.
- 10 exact duplicate complete rows.
- 52,583 usable responses for the concentration-difficulty outcome;
  frequent difficulty (`Often` or `Always`) is 50.01%.
- 2,806 duplicate **11-feature predictor profiles** among those responses.
  These are not necessarily duplicate respondents; respondent identifiers,
  timing, and site identifiers are absent.
- No verified provenance, independent participant count, data collection
  protocol, consent/license review, or independent clinical labels were
  found in the repository materials inspected.
- Treat the 52,610 rows as records, not as verified independent subjects.

## Leakage-conscious baseline screen

Exploratory 80/20 stratified random holdout, random seed 42,
one-hot-encoded survey predictors, RandomForestClassifier
(100 trees, maximum depth 12). We excluded the direct target answer,
medical-condition responses, downstream multi-select side effects and
coping-response fields; ambiguous height/weight units were also excluded.
These are **internal tests only**, not external-generalization results.

| Candidate self-reported outcome | Positive rate | RF balanced accuracy | RF ROC AUC | Interpretation |
|---|---:|---:|---:|---|
| Frequent concentration difficulty | 50.01% | 0.5192 | 0.5181 | Near chance |
| Frequent difficulty falling asleep | 22.50% | 0.5000 | 0.5705 | Predicted majority at default threshold |
| Frequent breathing difficulties | 6.71% | 0.5000 | 0.6160 | Predicted majority at default threshold |
| Fatigue listed as a side effect | 76.02% | 0.5000 | 0.5807 | Predicted majority at default threshold |
| Academic/work performance side effect | 35.80% | 0.5003 | 0.5055 | Near chance |
| Short sleep (less than 4 or 4–6-hour response) | 52.31% | 0.9431 | 0.9793 | **Not an independent health endpoint:** reported bed/wake times strongly encode duration, which is also directly asked in the form |

For concentration difficulty, an additional GroupShuffleSplit keeping
identical 11-feature predictor profiles in one side of the split gave
**balanced accuracy 0.5284 and ROC AUC 0.5282** (random forest).
This check removes exact-profile overlap but does not establish independent
participant/site generalization.

**Why raw accuracy is misleading:** 93.29% accuracy on rare frequent
breathing difficulties and 77.50% on sleep-onset difficulty are just the
majority-class rates. The corresponding balanced accuracy is 0.50.

## Reproduce

```bash
python scripts/audit_sleep_targets.py --data notebooks/final_sleep_dataDev.csv
```

Execution used Python 3.12.14, scikit-learn 1.8.0,
pandas 2.2.3, NumPy 2.3.5. The audit script was run successfully with
the checked-in dataset through a streamed read. No model was accepted,
deployed, or substituted based on these exploratory numbers.

## Requirements before a new model can be accepted

1. Choose an independently observed and meaningful outcome (for example,
   a correctly administered, separately scored validated sleep questionnaire),
   or acquire a suitable licensed and documented dataset containing it.
2. Keep the outcome, its component scoring questions, and post-outcome
   responses out of predictors unless the prediction task explicitly
   justifies their use and prohibits circular evaluation.
3. Verify respondent/source independence and split appropriately by
   participant, time, and/or site. Keep a final test set untouched.
4. Run imputation, encoding, feature selection, and any resampling
   **inside cross-validation folds**. Report classwise metrics and
   comparisons against meaningful baselines.
5. Make training and inference share one preprocessing contract.
   Add regression tests for label decoding, leakage and input schema.
6. Document informed-consent/privacy and data redistribution rights,
   plus a model card stating intended non-diagnostic use and limitations.

## Scope of this change

This PR adds a transparent, repeatable feasibility audit. It intentionally
does **not** change the production application, retrain, claim ML acceptance,
or deploy a replacement model. The old medical-classification training
path remains unsuitable and must not be used as evidence of valid health
prediction until redesigned.
