# Owner-reported collection provenance and consent — 2026-10-24

Project: Sleep Survey Explorer · Source: owner-collected 20-question sleep-behaviour questionnaire · 52,610 CSV rows.

## Source statement and verification level

The following collection details were provided directly by project owner **Neloy Pramanik Supto** on 24 October 2026. They are recorded as **owner-attested provenance**, not independently verified recruitment logs or proof that each row is a unique participant.

| Field | Owner statement / observed fact | Verification |
| --- | --- | --- |
| Recruitment locations | Daffodil International University hall; Kushtia; Sirajganj; multiple places in Dhaka | Owner-attested; no site identifiers in checked-in CSV |
| Contact modes | Person-to-person discussion, sharing a Google Form link, QR code, distribution through online groups | Owner-attested; no per-response recruitment-channel log inspected |
| Survey link | https://forms.gle/KW8c3opuEco7GHwz5 | Provided by owner; the short URL redirected to a Google Forms public view-form address, but its live questions and consent text were not retrievable in this environment |
| Nature of data | Primarily survey responses collected by the project owner | Owner-attested; 52,610 rows and 20 CSV columns were directly audited |
| Respondents | Owner believes responses came from different people; acknowledges some duplicate responses are possible | Not independently verifiable: no stable respondent ID is present in the checked-in CSV |
| Respondent permission | Owner states respondents were told directly that their data would be used for **research and academic purposes**, and that they agreed to participate | Owner-attested verbal disclosure and permission; no contemporaneous consent script or response log independently inspected; this does not establish permission for unrestricted public GitHub redistribution |
| Under-18 guardian permission | Owner additionally states that parents/guardians of under-18 respondents gave **verbal permission** | Owner-attested only; no guardian-consent wording, participation logs, identity verification, or publication-scope documentation independently reviewed |
| Dataset redistribution | Owner grants permission to use/share their collected dataset | Permission is an owner statement; third-party respondents' consent scope and relevant university policies remain separate verification questions |
| Approximate collection time | **Mid-2025** | Owner-reported approximation; exact start/end dates and timestamps not supplied |
| Sampling frame | Multi-location, in-person/online convenience recruitment | No per-site population denominators, recruitment counts or probability-sampling records inspected |
| Independent academic review | Owner presented the dataset-based ML project at their university final defense | Academic assessment reported by owner; **not** independent label verification, separate-site test, clinical endorsement, or peer-reviewed external acceptance |

## Quantitative source checks

- 52,610 survey records, **not 52,610 verified distinct people**.
- 10 exact complete-row duplicates.
- 5,792 distinct combinations of six categorical model-input answers; many people naturally share categorical profiles.
- 1,511 six-answer profiles have conflicting reported duration categories.
- Age-band distribution contains **12,925 records reporting Under 18** (owner dataset analysis). Owner reports verbal guardian permission for those records, but the consent process, wording and permission for public release are not independently documented.
- No verified recruitment-site, collection timestamp, respondent identity, consent version, or geographic weighting field appears in the committed 20-column dataset.

Do not equate identical survey answers with the same person. Do not infer independently verified independence from the owner's recollection. Documented exact-six-profile splitting **reduces direct profile overlap** between train/test but cannot guarantee independent persons.

## Data release and privacy caution

The original CSV and historical artefacts were already committed publicly. An owner can authorize use of their own research work, but we have **not established** that every respondent (including respondents reporting Under 18) agreed to public GitHub distribution or long-term reuse beyond the original survey. We neither deleted nor republished row-level records during this provenance update.

Before *any broader redistribution or marketing as a public health dataset*, verify and retain privately: the original questionnaire and any written consent notice (or document the verbal permission process as accurately as possible), exact collection dates if available, whether collecting responses from minors was covered, permitted publication scope, institutional guidance where applicable, and a de-identification review. Do not upload respondent names, emails, phone numbers, IPs, or other direct identifiers as evidence.

## Internal holdout is not an independent external dataset

The original CSV was already split into internal train/validation/test groups for V2; that is useful **internal** validation. Reserving another part of the *same collected survey* does not make a new external source or independently measured clinical outcome. Avoid reusing its original final test for model selection.

A meaningful future model needs a separately observed outcome. Public external data may be assessed only after checking comparable questions, consent/license, intended population and locked evaluation plans. The earlier CDC NHANES survey-source stress test is a useful transportability warning, not independent clinical measurement.

## Evidence still requested from owner

1. **Verbal consent protocol details**, especially how parents/guardians were informed, what they agreed to, and whether public raw-data sharing was mentioned. Owner reports verbal consent; corroborating records are not available in the inspected materials.
2. **Original survey introduction or accurate consent wording**, including whether unrestricted public data distribution was described; redact personally identifiable information.
3. If available, **exact collection months** and whether the original response export has timestamps, anonymous identifiers or recruitment location/channel, without publishing identifiers.

This document deliberately distinguishes observed repository facts from owner declarations, and is not a certification of consent, participant independence or clinical validity.
