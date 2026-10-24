# P0–P3 evidence and readiness ledger — 2026-10-24

Project: **Sleep Survey Explorer** · Dataset: `notebooks/final_sleep_dataDev.csv`

## Gate status

| Priority | Delivered | Acceptance / remaining limitation |
| --- | --- | --- |
| P0 source and cleaning audit | Repeatable aggregate-only scanner, missingness, exact duplicates, valid answer schema, conflicting answer profiles | **Partial.** Respondent IDs, sampling frame, consent, redistribution permission, date/site and original instrument are not independently documented. Not possible to assert 52,610 independent participants. |
| P1 meaningful prediction | Earlier clinical-target feasibility screen retained, medical prediction blocked, current non-diagnostic four-category predictor frozen | **Clinical goal blocked.** There is no separately measured health outcome in owner CSV. No invented clinical label or fake accuracy. |
| P2 independent-source evaluation | Reproducible frozen-model CDC NHANES 2015–16 external survey stress test, classwise metrics, calibration, two predefined boundary mappings, four subgroups | **External survey-source stress test completed; clinical generalization NOT accepted.** NHANES sleep hours are calculated from sleep/wake times. |
| P3 operational readiness | Bounded non-PII in-memory counters behind bearer auth, readiness/health checks, request IDs, size limits, security headers, Docker health check, synthetic HTTP and CI tests | **Local/container engineering only.** No hosted staging, uptime, production load, ongoing alerts, or security review demonstrated. AWS intentionally off. |

Owner later clarified that collection occurred around **mid-2025**, and participants were directly told their responses would be used for **research and academic purposes**; permission was reportedly obtained verbally. This remains an owner account, with exact dates, public-redistribution scope and corroboration of guardian permission (for Under-18 responses) unverified; the owner states verbal guardian permission was obtained. Owner supplied a collection-method statement after the first audit: Daffodil International University hall, Kushtia, Sirajganj and locations across Dhaka; in-person discussions, Google Form/QR and online groups; respondent permission stated. This is **owner-attested**, not independently checked against collection logs or the original consent notice. See [owner-reported collection provenance and consent](DATA_COLLECTION_PROVENANCE.md). The original dataset also contains 12,925 responses in the "Under 18" age band; age-appropriate consent and public redistribution scope need specific verification. A university final defense is not external ML validation.

## P0: original owner-collected CSV quality

Reproduce with `python -m scripts.audit_dataset_quality` (requires pandas). The scanner emits aggregate JSON only. It does not export subject rows or infer participant identities.

| Check | Result |
| --- | ---: |
| Questionnaire records | 52,610 |
| Fields per record | 20 |
| Exact entire-row duplicates | 10 |
| Distinct combinations of the six original answer questions | 5,792 |
| Combinations occurring in more than one row | 3,786 |
| Records that share six-answer combinations | 50,604 |
| Six-answer combinations with differing reported sleep durations | 1,511 |
| Invalid categorical answers for six inputs + duration label | 0 |
| Missing late-night reasons | 2,356 |
| Missing reported medical-condition answer | 336 |
| Missing self-reported side effects | 1,046 |
| Missing coping strategy | 7,843 |
| Missing concentration answer | 27 |

**Interpretation:** Most records repeat a *categorical combination* because the questionnaire has few answer options. This is not evidence of duplicated people. Conversely, only ten exact full-record matches does not establish 52,600 unique people. The conflicting target labels in 1,511 profile groups demonstrate irreducible ambiguity when treating categorical schedule answers as precise duration measurements. The V2 test split keeps *exact six-question profiles* together, but cannot separate people without IDs. Raw height and weight fields have ambiguous units and remain excluded from predictors. Missing optional fields are not imputed as a medical status.

**Provenance and privacy gate:** Owner collection is stated, but this repository has no independently inspected recruitment protocol, respondent consent record, questionnaire dates, geographic sampling strategy, collection-system audit trail, respondent identifiers, signed anonymization/redistribution decision, or external clinical measurement. The original survey CSV and large historical notebooks are presently in the public Git repository and its history. Removing a file in a future commit alone does not erase prior Git history. Before sharing this project as clinical research or publicly releasing a new dataset version, the owner should document permission, consent scope, de-identification and retention practices. We have **not** rewritten history or deleted original data.

## P1: honest label decision

The current 86.10% internal test accuracy (macro-F1 85.66%) is for a **self-reported duration category**, mostly encoded by bedtime and weekday wake-up answer bands. It is not diagnosis, objective hours of sleep, or a clinically useful disease score. Historical medical categories directly leaked their source question into their predictors; they cannot be reinstated. The existing exploratory target audit also found near-chance discrimination for concentration, breathing and sleep-onset proxies. No suitable independent clinical target exists in this file.

P1 becomes eligible only after collecting an independently assessed outcome (for example a properly administered validated sleep scale with no questionnaire item leakage, or PSG/actigraphy/clinician-verified outcomes) with permitted data rights, de-identified participant linkage and a locked subject/site split. A model trained against another public survey's **self-reported symptom** would constitute a new non-diagnostic task, not clinical validation of this model.

## P2: independent NHANES 2015–2016 survey stress test

Reference: [CDC NHANES 2015–2016 SLQ_I codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.htm). Official public XPT: [SLQ_I.XPT](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.XPT). Tested file SHA-256: `42dc4ce269704791223954e8513d91bd9b1da4cca89224b3ba87f043a9fa736a`.

**Critical limitation in the CDC documentation:** `SLD012` (reported weekday sleep hours) is **derived directly** by CDC from `SLQ300` (usual weekday bedtime) and `SLQ310` (usual weekday wake time); it is not independently observed sleep. Thus these metrics test whether the original survey-answer mapping transports across surveys. They do not validate a medical outcome. No NHANES responses were added to owner model training or calibration.

- Frozen input model: `models/sleep_duration_multiclass_v2.json`, unchanged.
- Predefined timing bands: bed 18:00–22:00, 22:00–00:00, 00:00–02:00, 02:00–06:00; wake 03:00–06:00, 06:00–08:00, 08:00–10:00, 10:00–14:00. Times outside these ranges are excluded rather than forced into misleading owner response options.
- Predeclared primary duration bins: under 4, 4 to under 6, 6 to 8 inclusive, over 8 hours. A second report assigns exact 6 hours to the 4–6 band to show boundary ambiguity.
- Dataset respondents: 6,327; eligible 6,104; excluded 223 for missing or out-of-range time mapping. Survey weights not used, so these are **unweighted exploratory** statistics, not population estimates.

| Metric | Owner internal profile-group test | External primary | External boundary sensitivity |
| --- | ---: | ---: | ---: |
| Evaluated records | 9,207 | 6,104 | 6,104 |
| Accuracy | 86.10% | **64.17%** | 63.63% |
| Balanced accuracy | 83.62% | **54.96%** | 55.80% |
| 10-bin ECE (lower is better) | 9.93% | **24.54%** | 18.33% |
| Log loss | 0.4442 | 0.9853 | 0.9219 |
| Multiclass Brier | Not reported | 0.5691 | 0.5318 |

Primary external class supports: under 4 = **34**, 4–6 = **411**, 6–8 = **3,630**, over 8 = **2,029**. Recalls, respectively: **55.88%, 47.93%, 84.10%, 31.94%**. Four-way class balance is poor; notably the under-four category has only 34 observations. Never use external raw accuracy alone as evidence of reliability.

External subgroup stress tests (predefined, no retraining):

| NHANES group | n | Accuracy | Balanced accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Ever told doctor about sleep trouble: yes | 1,569 | 62.78% | 56.44% | 24.51% |
| Ever told doctor about sleep trouble: no | 4,531 | 64.67% | 53.97% | 24.54% |
| Daytime sleepiness: rare (response 0–1) | 2,533 | 65.34% | 57.84% | 23.64% |
| Daytime sleepiness: frequent (response 3–4) | 1,577 | 63.35% | 52.84% | 24.61% |

Groups are self-reported response strata in the *same NHANES collection*, not independent clinical sites or a diagnosis. Cell counts for rare duration classes can be very small.

Reproduce without a dataset in Git:

```powershell
py -3.11 -m pip install pandas==2.2.3 numpy==2.3.5
py -3.11 -m scripts.evaluate_external_nhanes
```

Script downloads only the official CDC public XPT in memory, verifies SHA-256, and prints aggregate JSON. Alternatively `--file path\to\SLQ_I.XPT` reads an official downloaded copy. The external data is **never used for training, retuning thresholds, or choosing model weights**. Because ECE worsened, do not advertise externally calibrated probabilities.

## P3: operations, safety and local verification

`/healthz`: basic liveness. `/readyz`: artifact-validated start readiness. `/metrics`: **404 unless** `SLEEP_METRICS_TOKEN` is set, then requires exact `Authorization: Bearer <token>` and emits only fixed-route/status-class counts and duration sums. Tokens must be provided via environment secret; do not commit them. Counters reset at restart and are per worker, so aggregation across processes requires external monitoring.

The app now adds per-response no-cache, frame-denial, `nosniff`, no-referrer, and random request-ID headers; 16-KiB form-body limit; previous JSON comparison-body 4-KiB limit; and Docker health check. No user answers, medical text, IP, query parameters or identity enter request metrics. **This is basic operational instrumentation, not proven compliance, authorization for the assessment page, durable telemetry, or production security acceptance.** Deployment, remote synthetic probes, load tests, alerting and rollback drill remain not demonstrated (AWS is intentionally disabled by owner).

**AWS auto-deploy safety:** The older `.github/workflows/cicd.yaml` formerly triggered AWS/ECR and a self-hosted deployment on *every main push*. Because AWS deployment is paused by the owner, its trigger was changed to manual `workflow_dispatch` only; no deployment or secret change was executed. The separate non-AWS unit/container CI still runs on pull requests and main.

Windows local smoke run after the app starts:

```powershell
py -3.11 -m uvicorn app:app --reload
# In a second PowerShell terminal in the same project directory:
py -3.11 -m scripts.smoke_test_local
```

The smoke script uses only synthetic dataset-allowed answers; it exercises the form, richer self-report, comparison, invalid and oversized submissions, obsolete endpoint, readiness and metrics-access boundary. GitHub CI runs regression tests, dataset-profile assertions, compilation, Docker build and container HTTP smoke. Historical ML pipelines are deliberately left **unchanged but unsupported**, not deployed, and cannot be substituted for this non-diagnostic runtime.

## Final acceptance gates still open

1. **P0:** Collection metadata, consent/license/redistribution rights and independently corroborated participant identities/sampling frame.
2. **P1:** Separately observed, meaningful outcome and genuine leakage-safe, participant-level validation.
3. **P2:** Compatible independently *measured* dataset, predeclared acceptance thresholds, confidence intervals and externally acceptable calibration. Existing frozen model currently fails robust generalization expectations.
4. **P3:** User-approved non-AWS deployment target (if deployment ever desired), repeatable load and security tests, real uptime evidence, safe telemetry, operator review.

**Decision: research/portfolio demonstrator with stronger evidence and local operations — NOT clinically validated or fully production accepted.**
