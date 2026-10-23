"""Frozen-model external-source stress test using the official CDC NHANES SLQ_I.

The NHANES 'sleep hours' value SLD012 is *calculated from* SLQ300/SLQ310;
therefore this is NOT independent-outcome, clinical, or health-risk validation.
It tests transportability of grouped questionnaire schedule/answer mappings only.
No participant-level records are committed or printed.

Usage: python -m scripts.evaluate_external_nhanes
       python -m scripts.evaluate_external_nhanes --file downloaded/SLQ_I.XPT
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from sleep_project.duration_v2 import CLASSES, SleepDurationAssessmentV2

SOURCE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.XPT"
SOURCE_CODEBOOK = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/SLQ_I.htm"
SOURCE_SHA256 = "42dc4ce269704791223954e8513d91bd9b1da4cca89224b3ba87f043a9fa736a"


def clock_minutes(value):
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        return None
    try:
        hours, minutes = int(value[:2]), int(value[3:])
    except ValueError:
        return None
    if not (0 <= hours < 24 and 0 <= minutes < 60):
        return None
    return 60 * hours + minutes


def bed_band(minutes):
    if minutes is None:
        return None
    if 18 * 60 <= minutes < 22 * 60:
        return "Before 10 PM"
    if 22 * 60 <= minutes < 24 * 60:
        return "10 PM - 12 AM"
    if 0 <= minutes < 2 * 60:
        return "12 AM - 2 AM"
    if 2 * 60 <= minutes < 6 * 60:
        return "After 2 AM"
    return None  # Day/shift sleepers cannot be mapped responsibly.


def wake_band(minutes):
    if minutes is None:
        return None
    if 3 * 60 <= minutes < 6 * 60:
        return "Before 6 AM"
    if 6 * 60 <= minutes < 8 * 60:
        return "6 AM - 8 AM"
    if 8 * 60 <= minutes < 10 * 60:
        return "8 AM - 10 AM"
    if 10 * 60 <= minutes < 14 * 60:
        return "After 10 AM"
    return None


def duration_class(hours, six_in_upper_band=True):
    if not np.isfinite(hours) or hours < 0 or hours > 24:
        return None
    if hours < 4:
        return CLASSES[0]
    if hours < 6 or (hours == 6 and not six_in_upper_band):
        return CLASSES[1]
    if hours <= 8:
        return CLASSES[2]
    return CLASSES[3]


def summarize(y, predicted, probabilities):
    n = len(y)
    if n == 0:
        raise ValueError("No eligible independent-source rows")
    truth = np.asarray([CLASSES.index(label) for label in y], dtype=int)
    guesses = np.asarray([CLASSES.index(label) for label in predicted], dtype=int)
    p = np.asarray(probabilities, dtype=float)
    support = np.bincount(truth, minlength=4)
    confusion = np.zeros((4, 4), dtype=int)
    for actual, guess in zip(truth, guesses):
        confusion[actual, guess] += 1
    class_recalls = [round(float(confusion[i, i] / support[i]), 4) if support[i] else None for i in range(4)]
    balanced = float(np.mean([confusion[i, i] / support[i] for i in range(4) if support[i]]))
    correct = (truth == guesses).astype(float)
    ece = 0.0
    confidences = p.max(axis=1)
    for lo, hi in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)):
        mask = (confidences >= lo) & (confidences < hi if hi < 1 else confidences <= hi)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidences[mask].mean()))
    log_loss = float(-np.log(np.clip(p[np.arange(n), truth], 1e-12, 1)).mean())
    brier = float(np.mean(np.sum((p - np.eye(4)[truth]) ** 2, axis=1)))
    return {
        "n": n,
        "accuracy": round(float(correct.mean()), 4),
        "balanced_accuracy": round(balanced, 4),
        "log_loss": round(log_loss, 4),
        "multiclass_brier": round(brier, 4),
        "ece_10_bins": round(ece, 4),
        "support_by_class": dict(zip(CLASSES, map(int, support))),
        "recall_by_class": dict(zip(CLASSES, class_recalls)),
        "confusion_matrix_actual_rows_predicted_columns": confusion.tolist(),
    }


def evaluate(frame, model):
    needed = {"SLQ300", "SLQ310", "SLD012", "SLQ050", "SLQ120"}
    if not needed.issubset(frame.columns):
        raise ValueError("Missing NHANES SLQ columns: " + str(sorted(needed - set(frame.columns))))
    base = {
        question["key"]: question["options"][0]
        for question in model.questions()
    }
    examples, excluded = [], {"invalid_or_unmapped_clock": 0, "missing_or_invalid_hours": 0}
    for bed, wake, hours, trouble, sleepy in frame[["SLQ300", "SLQ310", "SLD012", "SLQ050", "SLQ120"]].itertuples(index=False, name=None):
        bed_value = bed_band(clock_minutes(bed))
        wake_value = wake_band(clock_minutes(wake))
        if bed_value is None or wake_value is None:
            excluded["invalid_or_unmapped_clock"] += 1
            continue
        if pd.isna(hours) or duration_class(hours) is None:
            excluded["missing_or_invalid_hours"] += 1
            continue
        answers = dict(base, bed_time=bed_value, wake_time=wake_value)
        result = model.predict(answers)
        examples.append({
            "hours": float(hours),
            "guess": result["predicted_category"],
            "p": [item["probability"] for item in result["category_probabilities"]],
            "bed": bed_value, "wake": wake_value,
            "trouble": int(trouble) if pd.notna(trouble) else None,
            "sleepy": int(round(float(sleepy))) if pd.notna(sleepy) else None,
        })
    if not examples:
        raise ValueError("Zero mappable NHANES survey records")

    def subset(rows, upper):
        return summarize(
            [duration_class(row["hours"], upper) for row in rows],
            [row["guess"] for row in rows],
            [row["p"] for row in rows],
        )

    # Group counts below 100 suppressed; all groups are based on the same
    # external survey, so these are robustness checks, not independent sites.
    subgroups = {}
    for name, rule in (
        ("told_doctor_trouble_yes", lambda r: r["trouble"] == 1),
        ("told_doctor_trouble_no", lambda r: r["trouble"] == 2),
        ("daytime_sleepiness_rare_0_to_1", lambda r: r["sleepy"] in (0, 1)),
        ("daytime_sleepiness_frequent_3_to_4", lambda r: r["sleepy"] in (3, 4)),
    ):
        rows = [record for record in examples if rule(record)]
        if len(rows) >= 100:
            subgroups[name] = subset(rows, True)
    return {
        "dataset": "CDC NHANES 2015-2016 Sleep Disorders SLQ_I",
        "external_population_records": int(len(frame)),
        "eligible_records": len(examples),
        "exclusions": excluded,
        "primary_6_hours_in_6_to_8_band": subset(examples, True),
        "sensitivity_6_hours_in_4_to_6_band": subset(examples, False),
        "subgroups": subgroups,
        "source_codebook": SOURCE_CODEBOOK,
        "external_2015_16_survey": True,
        "survey_weighted_population_inference": False,
        "independently_measured_outcome": False,
        "medical_validation": False,
        "interpretation": (
            "External questionnaire-source schedule-mapping stress test only. "
            "CDC constructs SLD012 from SLQ300 and SLQ310: not a separately "
            "measured outcome. The current model is not a clinical assessment. "
            "Ambiguous 6-hour boundary shown as predeclared sensitivity test. "
            "Unweighted results are not nationally representative estimates."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Optional local copy of official SLQ_I.XPT")
    parser.add_argument("--model", default="models/sleep_duration_multiclass_v2.json")
    args = parser.parse_args()
    if args.file:
        raw = Path(args.file).read_bytes()
    else:
        with urlopen(SOURCE, timeout=45) as response:
            raw = response.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("NHANES source checksum differs from the audited official file; review before evaluation")
    frame = pd.read_sas(io.BytesIO(raw), format="xport")
    report = evaluate(frame, SleepDurationAssessmentV2(args.model))
    report["official_file_url"] = SOURCE
    report["file_sha256"] = digest
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
