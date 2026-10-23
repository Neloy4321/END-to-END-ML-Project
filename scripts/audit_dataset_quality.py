"""Aggregate-only quality audit for the owner-collected questionnaire.

No row-level survey answers are printed or persisted. This audit cannot infer
participant uniqueness, collection consent, or a clinical ground truth.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from sleep_project.behavioral_assessment import FORM_FIELDS
from sleep_project.duration_v2 import SleepDurationAssessmentV2


def audit(data, model):
    columns = list(data.columns)
    response = model.model["target_column"]
    required = [column for _, column, _ in FORM_FIELDS] + [response]
    missing_cols = sorted(set(required) - set(columns))
    if missing_cols:
        raise ValueError("Required source columns missing: " + ", ".join(missing_cols))
    features = [column for _, column, _ in FORM_FIELDS]
    invalid = {}
    for column in features:
        values = data[column].astype("string").str.strip()
        invalid[column] = int((~values.isin(model.model["category_options"][column])).sum())
    invalid[response] = int((~data[response].astype("string").str.strip().isin(model.model["response_classes"])).sum())
    profile = data[features].fillna("<missing>").astype(str)
    sizes = profile.groupby(features, dropna=False).size()
    with_target = data[features + [response]].fillna("<missing>").astype(str)
    distinct_labels = with_target.groupby(features, dropna=False)[response].nunique()
    # Repeated profiles represent answer combinations, never verified people.
    discordant_profiles = int((distinct_labels > 1).sum())
    return {
        "source_rows": int(len(data)),
        "source_columns": int(len(columns)),
        "exact_complete_row_duplicates": int(data.duplicated().sum()),
        "missing_by_column": {col: int(n) for col, n in data.isna().sum().items()},
        "invalid_core_answer_counts": invalid,
        "six_answer_distinct_profiles": int(len(sizes)),
        "six_answer_profiles_repeated": int((sizes > 1).sum()),
        "records_in_repeated_six_answer_profiles": int(sizes.loc[sizes > 1].sum()),
        "six_answer_profiles_with_disagreeing_duration_answers": discordant_profiles,
        "no_verified_participant_identifier": True,
        "participant_independence": "NOT VERIFIABLE from this CSV",
        "collection_protocol_and_consent": "NOT VERIFIED from this CSV",
        "clinical_outcome": "NOT AVAILABLE: respondent-provided categorical sleep duration only",
        "warning": "Matching responses are not proof of repeat participants. Do not claim clinical accuracy or independence.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="notebooks/final_sleep_dataDev.csv")
    parser.add_argument("--model", default="models/sleep_duration_multiclass_v2.json")
    args = parser.parse_args()
    source = Path(args.data)
    model = SleepDurationAssessmentV2(args.model)
    data = pd.read_csv(args.data)
    report = audit(data, model)
    if source.is_file():
        report["source_file_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
