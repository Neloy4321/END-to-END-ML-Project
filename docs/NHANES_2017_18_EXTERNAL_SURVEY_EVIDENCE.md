# External survey-source check: NHANES 2017–2018 SLQ_J

Evaluation date: 2026-10-24. Official CDC questionnaire and reference:
https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.htm

Input: owner-uploaded unmodified official SLQ_J.XPT, SHA-256
5486e1c51d56a959c94106d131a11b2e58bdec8ff30917ad7748917c6fa09f3a.

This file is **not** included in Git. Its separately delivered CSV conversion has
6,161 source rows and 11 columns. Four SAS time byte fields were decoded as
ASCII, numeric tiny-zero XPT representations were normalized to coded zero
where appropriate, and original columns and survey codes were retained.
A CSV is only a **format conversion**, not a new clinical dataset or new
independent observation.

## What was measured

Frozen deployed artifact: models/sleep_duration_multiclass_v2.json,
unchanged, with only the usual weekday bedtime and wake band used by its
selected classifier. No training, validation tuning, or temperature fitting
used this NHANES cycle. We reused the pre-declared V2 response mapping and
the code from scripts/evaluate_external_nhanes.py; this cycle gets its own
wrapper, SHA guard and metadata in
scripts/evaluate_external_nhanes_2017_18.py.

**Important mismatch:** The owner survey asks when respondents usually
*go to bed*; 2017–18 NHANES SLQ300 asks when they usually *fall asleep*.
Additionally CDC constructs weekday duration (SLD012) **from reported SLQ300
and SLQ310**, rather than measuring it separately. Thus this tests
transport of questionnaire schedule-response relationships across source
populations with non-identical instruments, **not** objectively measured
sleep or clinical disease performance.

## Aggregate measurements

| Metric | Frozen owner internal profile-group test | NHANES 2015–16 | Uploaded NHANES 2017–18 |
| --- | ---: | ---: | ---: |
| Evaluated records | 9,207 | 6,104 | 5,860 |
| Accuracy | 86.10% | 64.17% | **64.78%** |
| Balanced accuracy | 83.62% | 54.96% | **57.55%** |
| 10-bin ECE (lower is better) | 9.93% | 24.54% | **24.29%** |
| Log loss | 0.4442 | 0.9853 | **0.9728** |
| Multiclass Brier | — | 0.5691 | **0.5634** |

Source rows: 6,161; excluded 301 for outside-map or invalid clock,
leaving 5,860 mappable records. No additional eligible rows were excluded
for invalid duration. No survey weights were applied, so this is **not a
nationally representative population estimate**.

Class support / per-class recall on NHANES 2017–18:

| Reported duration | n | Recall |
| --- | ---: | ---: |
| Under 4 hours | 40 | 62.50% |
| 4–6 hours | 445 | 50.56% |
| 6–8 hours | 3,495 | 83.20% |
| Over 8 hours | 1,880 | 33.94% |

There are only 40 observations in the first band, so that recall is
highly unstable. The positive-looking overall accuracy largely reflects
the large 6–8-hour group; bacc and poor probability calibration expose
cross-source weaknesses. Reassigning exactly 6 hours to the 4–6 band
(predeclared sensitivity) gives 64.04% accuracy, 58.02% bacc, 17.07% ECE.

Predeclared self-report subgroups (each >=100; same source, no new sites):

| Subgroup | n | Accuracy | Balanced accuracy | ECE |
| --- | ---: | ---: | ---: | ---: |
| Told a doctor about sleep trouble: yes | 1,525 | 63.80% | 58.10% | 24.72% |
| Told a doctor about sleep trouble: no | 4,330 | 65.15% | 57.34% | 24.14% |
| Daytime sleepiness rare (0–1) | 2,408 | 65.57% | 61.13% | 23.93% |
| Daytime sleepiness frequent (3–4) | 1,466 | 63.17% | 55.39% | 24.34% |

## Reproduce on Windows (Python 3.11)

Obtain SLQ_J.XPT from CDC or use the uploaded official file:

~~~powershell
python -m scripts.evaluate_external_nhanes_2017_18 --file SLQ_J.XPT
~~~

Run from repository root; pandas and NumPy must be installed (see existing
README setup). The checksum must match before inference. The evaluator
prints aggregate JSON only and never logs participant records.

**Decision:** No new medical prediction, no model replacement, no AWS
deployment and **no claim of external clinical validation**. The second
CDC survey cycle corroborates that the original 86.10% owner-internal score
does not directly transport, so a clinically meaningful independent
measured outcome still needs separate data collection or approved access.
