"""CDC NHANES 2017-18 SLQ_J transfer check of the frozen V2 survey-answer model.

Evaluates a DISTINCT survey cycle without retraining. The CDC's sleep duration
(SLD012) is calculated from reported falling-asleep and waking times; this is
NOT independent clinical ground truth, and the owner's bedtime question is
not exactly the same as SLQ_J's falling-asleep question.

Usage:
  python -m scripts.evaluate_external_nhanes_2017_18 --file SLQ_J.XPT
  python -m scripts.evaluate_external_nhanes_2017_18
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

from scripts.evaluate_external_nhanes import evaluate
from sleep_project.duration_v2 import SleepDurationAssessmentV2

SOURCE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.XPT"
CODEBOOK = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/SLQ_J.htm"
# SHA-256 independently calculated from the user's official downloaded XPT.
SOURCE_SHA256 = "5486e1c51d56a959c94106d131a11b2e58bdec8ff30917ad7748917c6fa09f3a"


def score(frame, model):
    """No tuning/fit on this dataset; return aggregate measurements only."""
    output = evaluate(frame, model)
    output.update({
        "dataset": "CDC NHANES 2017-2018 Sleep Disorders SLQ_J",
        "source_codebook": CODEBOOK,
        "external_2015_16_survey": False,
        "external_2017_18_survey": True,
        "survey_weighted_population_inference": False,
        "independently_measured_outcome": False,
        "medical_validation": False,
        "important_questionnaire_mismatch": (
            "Owner survey asks time usually going to bed, while NHANES "
            "2017-2018 SLQ300 asks time usually falling asleep. "
            "This comparison is a schedule-band transfer stress test."
        ),
        "interpretation": (
            "US observational questionnaire-source transfer test only. "
            "CDC derives SLD012 from SLQ300 and SLQ310, so reported "
            "weekday duration is not a separately measured target. "
            "Weekday timing wording, population, site, and exact duration "
            "boundaries differ from the owner's Bangladesh survey. "
            "No external data was used to train, tune, or recalibrate "
            "the frozen model. No survey weights were used; these are "
            "unweighted sample metrics, not representative estimates."
        ),
    })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Official NHANES 2017-18 SLQ_J.XPT file")
    parser.add_argument("--model", default="models/sleep_duration_multiclass_v2.json")
    args = parser.parse_args()
    if args.file:
        raw = Path(args.file).read_bytes()
    else:
        with urlopen(SOURCE, timeout=45) as response:
            raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Official XPT checksum mismatch; do not evaluate altered file")
    frame = pd.read_sas(io.BytesIO(raw), format="xport")
    result = score(frame, SleepDurationAssessmentV2(args.model))
    result.update({"official_file_url": SOURCE, "file_sha256": digest})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
