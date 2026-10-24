# CDC NHANES 2017–2018 Sleep Disorders public dataset

- Original official source: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.XPT
- CDC questionnaire, methods and variable codebook:
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.htm
- File: SLQ_J.XPT (original SAS Transport format; do not edit or relabel)
- SHA-256: 5486e1c51d56a959c94106d131a11b2e58bdec8ff30917ad7748917c6fa09f3a
- Original file size: 470,560 bytes
- Public-release responses: 6,161 records, 11 columns.
- Supplied/downloaded by project owner on 2026-10-24.

This **external US survey dataset** is separate from the owner's
Bangladesh sleep questionnaire; row identities and survey instruments are
different. Do not concatenate them as if they are the same respondents or
pretend they share a diagnosis label.

In this NHANES cycle, SLQ300 describes when the respondent usually
*falls asleep*, not when they *go to bed* as asked in the owner questionnaire.
CDC computes SLD012 weekday sleep hours from self-reported SLQ300 and
SLQ310, so it is not an objectively measured independent sleep outcome.

Existing reproducible evaluation (frozen owner V2 model, no retraining):

~~~powershell
python -m scripts.evaluate_external_nhanes_2017_18 --file datasets/external/SLQ_J.XPT
~~~

Aggregate results and limitations:
[2017–2018 transfer evaluation](../../docs/NHANES_2017_18_EXTERNAL_SURVEY_EVIDENCE.md).

This repository copy preserves the original CDC bytes. No clinical,
diagnostic, prospective, nationally representative, or model deployment
acceptance is claimed.
