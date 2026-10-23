"""Create privacy-conscious aggregate survey comparisons for the V3 dashboard.

No raw respondent rows, joined cohorts, or fabricated diagnoses are exported.
python scripts/build_survey_cohorts_v3.py
git show HEAD:notebooks/final_sleep_dataDev.csv | python scripts/build_survey_cohorts_v3.py --data - --output -
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

TARGET = "How many hours of sleep do you get on average per night?"
CLASSES = ("Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours")
AGE = "Your Age"
GENDER = "What is your gender?"
OCCUPATION = "What is your occupation?"
MEDICAL = "Do you have any medical conditions that might affect your sleep?"
REASONS = "What are the main reasons you sleep late?"
EFFECTS = "Do you experience any of the following side effects from late sleeping?"
COPING = "What strategies do you use to cope with the side effects of late sleeping?"

REASON_ITEMS = (
    "Social Media/Internet", "Entertainment (TV;movies;games)",
    "Study & Exam Purpose", "Stress or anxiety", "professional Work",
    "Insomnia or other sleep disorder", "Others",
)
EFFECT_ITEMS = (
    "Fatigue", "Difficulty concentrating", "Mood swings or irritability",
    "Decreased academic or work performance", "Health issues", "Back Pain", "Others",
)
COPING_ITEMS = (
    "Naps during the day", "Caffeine or other stimulants",
    "Relaxation techniques", "Exercise", "Others",
)
MEDICAL_CATEGORIES = (
    "No", "Insomnia", "Anxiety Disorders", "GERD/Acidity", "Allergies",
    "Asthma", "Depression", "Chronic Pain", "Obstructive Sleep Apnea",
    "Restless Legs Syndrome", "Thyroid Disorders", "Diabetes",
    "Hypertension", "Narcolepsy",
)
MIN_GROUP = 100


def distribution(series):
    counts = series.value_counts()
    n = int(len(series))
    return {
        "n": n,
        "counts": {class_name: int(counts.get(class_name, 0)) for class_name in CLASSES},
        "percent": {
            class_name: round(100 * float(counts.get(class_name, 0)) / n, 1)
            for class_name in CLASSES
        },
    }


def categories(frame, column, allowed=None):
    choices = allowed if allowed is not None else sorted(frame[column].dropna().astype(str).unique())
    result = {}
    for choice in choices:
        subset = frame.loc[frame[column] == choice, TARGET]
        if len(subset) >= MIN_GROUP:
            result[str(choice)] = distribution(subset)
    return result


def prevalence(frame, column, items):
    # Multi-select answers are serialized as semicolon-separated strings.
    # Substring matching preserves "Entertainment (TV;movies;games)" intact.
    answered = frame[column].dropna().astype(str)
    denom = int(len(answered))
    return {
        value: {
            "n": int(answered.str.contains(value, case=False, regex=False).sum()),
            "percent_of_answers": round(
                100 * float(answered.str.contains(value, case=False, regex=False).sum()) / denom, 1
            ) if denom else 0,
        } for value in items
    }


def build(data):
    needed = (TARGET, AGE, GENDER, OCCUPATION, MEDICAL, REASONS, EFFECTS, COPING)
    if any(name not in data.columns for name in needed):
        raise ValueError("Survey source columns missing")
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != set(CLASSES):
        raise ValueError("Invalid survey duration labels")
    output = {
        "schema_version": 3,
        "task": "descriptive_non_diagnostic_survey_cohorts",
        "source_record_count": int(len(data)),
        "class_labels": list(CLASSES),
        "minimum_cohort_size": MIN_GROUP,
        "overall": distribution(data[TARGET]),
        "by_age": categories(data, AGE),
        "by_gender": categories(data, GENDER),
        "by_occupation": categories(data, OCCUPATION),
        "by_medical_self_report": categories(data, MEDICAL, MEDICAL_CATEGORIES),
        "reason_prevalence": prevalence(data, REASONS, REASON_ITEMS),
        "effect_prevalence": prevalence(data, EFFECTS, EFFECT_ITEMS),
        "coping_prevalence": prevalence(data, COPING, COPING_ITEMS),
        "single_select_options": {
            "age": sorted(data[AGE].dropna().astype(str).unique()),
            "gender": sorted(data[GENDER].dropna().astype(str).unique()),
            "occupation": sorted(data[OCCUPATION].dropna().astype(str).unique()),
            "weekend_bed": sorted(data["What time do you usually go to bed on weekends?"].dropna().astype(str).unique()),
            "weekend_wake": sorted(data["What time do you usually wake up on weekends?"].dropna().astype(str).unique()),
            "sleep_latency": sorted(data["How long does it take you to fall asleep after going to bed?"].dropna().astype(str).unique()),
            "medical_self_report": list(MEDICAL_CATEGORIES),
            "concentration": sorted(data["How often do you find it hard to concentrate due to lack of sleep?"].dropna().astype(str).unique()),
        },
        "multi_select_options": {
            "late_reasons": list(REASON_ITEMS),
            "side_effects": list(EFFECT_ITEMS),
            "coping": list(COPING_ITEMS),
        },
        "limitations": (
            "Descriptive frequencies from unverified self-report records, not a clinical cohort; "
            "no independent respondent IDs, diagnosis verification, temporal ordering, "
            "patient-level disease risk or causal interpretation. "
            "All groups with fewer than 100 source records are omitted; "
            "no raw survey rows or respondent combinations are exported."
        ),
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="notebooks/final_sleep_dataDev.csv")
    parser.add_argument("--output", default="models/survey_cohorts_v3.json")
    args = parser.parse_args()
    df = pd.read_csv(sys.stdin if args.data == "-" else args.data)
    text = json.dumps(build(df), ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print("Saved " + str(target), file=sys.stderr)


if __name__ == "__main__":
    main()
