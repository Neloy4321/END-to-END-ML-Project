"""P1 leakage-conscious screening of self-reported survey outcomes.

This is exploratory model *eligibility* research, NOT clinical validation.
No raw responses or model artifacts are emitted. No candidate is auto-deployed.

Run:
    python -m scripts.screen_p1_targets
    git show HEAD:notebooks/final_sleep_dataDev.csv | python -m scripts.screen_p1_targets --data -
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


# Behaviours/background that do not contain any evaluated symptom question,
# medical-condition question, coping or consequences. Collected cross-sectionally:
# this is *association screening*, never prospective or causal prediction.
FEATURES = (
    "Your Age",
    "What is your gender?",
    "What is your occupation?",
    "What time do you usually go to bed?",
    "What time do you usually wake up on working days?",
    "What time do you usually go to bed on weekends?",
    "What time do you usually wake up on weekends?",
    "How would you rate the comfort of your sleeping environment",
)
TARGETS = {
    "frequent_concentration_difficulty": (
        "How often do you find it hard to concentrate due to lack of sleep?",
        "frequency",
    ),
    "frequent_sleep_onset_difficulty": (
        "Do you have difficulty falling asleep?", "frequency",
    ),
    "frequent_sleep_breathing_difficulty": (
        "Do you experience breathing difficulties while sleeping", "frequency",
    ),
    "fatigue_self_report": (
        "Do you experience any of the following side effects from late sleeping?",
        "contains:Fatigue",
    ),
    "work_or_study_impact_self_report": (
        "Do you experience any of the following side effects from late sleeping?",
        "contains:Decreased academic or work performance",
    ),
}
MIN_CLASS_PER_SPLIT = 10


def assert_safe_schema(frame):
    required = set(FEATURES) | {column for column, _ in TARGETS.values()}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Missing questionnaire columns: " + ", ".join(missing))
    forbidden = ("medical", "concentrat", "difficulty falling asleep",
                 "breathing difficulties", "side effects", "coping", "sleep duration")
    for column in FEATURES:
        if any(part in column.lower() for part in forbidden):
            raise ValueError("Potential symptom/outcome leakage via " + column)
    if set(FEATURES) & {column for column, _ in TARGETS.values()}:
        raise ValueError("A target question is also a predictor")


def label_for(frame, column, label_type):
    raw = frame[column]
    valid = raw.notna() & raw.astype(str).str.strip().ne("")
    if label_type == "frequency":
        allowed = {"Never", "Rarely", "Sometimes", "Often", "Always"}
        valid = valid & raw.isin(allowed)
        labels = raw.isin({"Often", "Always"}).astype(int)
    elif label_type.startswith("contains:"):
        answer = label_type.split(":", 1)[1]
        labels = raw.fillna("").astype(str).str.contains(
            answer, regex=False, case=False
        ).astype(int)
    else:
        raise ValueError("Unknown label type")
    return valid.to_numpy(), labels.to_numpy()


def grouped_indices(frame):
    """Split *all* rows before selecting labels or fitting model selection."""
    groups = pd.util.hash_pandas_object(frame[list(FEATURES)], index=False)
    records = np.arange(len(frame))
    train, remaining = next(GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=20251024
    ).split(records, groups=groups))
    relative_val, relative_test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=20251025
    ).split(remaining, groups=groups.iloc[remaining]))
    validation = remaining[relative_val]
    test = remaining[relative_test]
    parts = (train, validation, test)
    group_sets = [set(groups.iloc[indices].tolist()) for indices in parts]
    if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("Answer-profile group leaked between splits")
    return parts


def model_candidates():
    return {
        "prior_only": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=400, random_state=42),
        ),
        "random_forest": make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            RandomForestClassifier(
                n_estimators=70, max_depth=10, min_samples_leaf=20,
                random_state=42, n_jobs=2,
            ),
        ),
    }


def positive_probability(estimator, feature_frame):
    prediction = estimator.predict_proba(feature_frame)
    return prediction[:, list(estimator.classes_).index(1)]


def performance(y, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    predicted = (probability >= 0.5).astype(int)
    return {
        "n": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": round(float(np.mean(y)), 5),
        "accuracy": round(float(accuracy_score(y, predicted)), 5),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predicted)), 5),
        "macro_f1": round(float(f1_score(y, predicted, average="macro")), 5),
        "roc_auc": round(float(roc_auc_score(y, probability)), 5),
        "log_loss": round(float(log_loss(y, probability, labels=[0, 1])), 5),
        "brier": round(float(brier_score_loss(y, probability)), 5),
    }


def auc_interval(y, probability, seed=42, repetitions=250):
    rng = np.random.default_rng(seed)
    y, probability = np.asarray(y), np.asarray(probability)
    scores = []
    for _ in range(repetitions):
        drawn = rng.integers(0, len(y), len(y))
        if np.unique(y[drawn]).size == 2:
            scores.append(roc_auc_score(y[drawn], probability[drawn]))
    if len(scores) < 20:
        return None
    return [round(float(value), 5) for value in np.percentile(scores, [2.5, 97.5])]


def run(frame):
    assert_safe_schema(frame)
    # Full-answer duplicates cannot be identified as repeated people; removing
    # them is conservative. This is still not a participant-independent split.
    duplicates = int(frame.duplicated().sum())
    unique = frame.drop_duplicates().reset_index(drop=True)
    inputs = unique[list(FEATURES)].fillna("<missing>").astype(str)
    split_indices = grouped_indices(inputs)
    results = {
        "task": "exploratory_self_reported_symptom_screen_not_diagnosis",
        "source_rows": int(len(frame)),
        "exact_response_rows_excluded": duplicates,
        "retained_rows": int(len(unique)),
        "data_source": "owner_collected_mid_2025_cross_sectional_survey",
        "features": list(FEATURES),
        "split": (
            "70/15/15 approximate seeded GroupShuffleSplit of identical "
            "eight-answer predictor profiles; no respondent IDs or site labels."
        ),
        "prior_screen_warning": (
            "These outcomes were previously explored on the same source dataset: "
            "this is retrospective, not a fresh independent confirmatory test."
        ),
        "medical_or_clinical_validation": False,
        "externally_validated": False,
        "eligible_for_app_model_replacement": False,
        "targets": {},
    }
    for name, (column, label_type) in TARGETS.items():
        valid, y = label_for(unique, column, label_type)
        parts = [indices[valid[indices]] for indices in split_indices]
        counts = [
            {str(k): int(np.sum(y[indices] == k)) for k in (0, 1)}
            for indices in parts
        ]
        result = {
            "target_question": column,
            "positive_definition": label_type if label_type != "frequency"
                                  else "Often or Always",
            "valid_n": int(np.sum(valid)),
            "class_counts_train_validation_test": counts,
            "label_source": "self_report_not_independent_clinical_measurement",
            "screen_pass": False,
        }
        if any(min(int(np.sum(y[indices] == value)) for value in (0, 1))
               < MIN_CLASS_PER_SPLIT for indices in parts):
            result["reason"] = "Insufficient support in at least one group split"
            results["targets"][name] = result
            continue

        training, validation, test = parts
        fitted = {}
        validation_metrics = {}
        for model_name, model in model_candidates().items():
            model.fit(inputs.iloc[training], y[training])
            fitted[model_name] = model
            validation_metrics[model_name] = performance(
                y[validation], positive_probability(model, inputs.iloc[validation])
            )
        # Entire candidate and algorithm selection uses validation only.
        best_name = min(validation_metrics, key=lambda key: (
            validation_metrics[key]["log_loss"],
            0 if key == "prior_only" else 1,
        ))
        # Only the selected model and prespecified prior baseline see final test.
        baseline_scores = positive_probability(fitted["prior_only"], inputs.iloc[test])
        selected_scores = positive_probability(fitted[best_name], inputs.iloc[test])
        baseline = performance(y[test], baseline_scores)
        selected = performance(y[test], selected_scores)
        ci = auc_interval(y[test], selected_scores)
        result.update({
            "validation_metrics": validation_metrics,
            "selected_on_validation_only": best_name,
            "test_prior_baseline": baseline,
            "test_selected": selected,
            "test_auc_bootstrap_95ci": ci,
            "test_log_loss_gain_over_prior": round(
                baseline["log_loss"] - selected["log_loss"], 5
            ),
        })
        # Exploratory engineering thresholds are *not* medical acceptance gates.
        # No outcome can be deployed as clinical or causal prediction on this data.
        result["screen_pass"] = bool(
            best_name != "prior_only"
            and ci is not None and ci[0] >= 0.60
            and selected["balanced_accuracy"] >= 0.60
            and result["test_log_loss_gain_over_prior"] >= 0.02
        )
        result["reason"] = (
            "Exploratory association merits future independent prospective study"
            if result["screen_pass"] else
            "No sufficiently strong evidence for a new prediction card"
        )
        results["targets"][name] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="notebooks/final_sleep_dataDev.csv")
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    dataset = pd.read_csv(sys.stdin if args.data == "-" else args.data)
    report = json.dumps(run(dataset), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(report)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
        print("Aggregate-only P1 report saved.", file=sys.stderr)


if __name__ == "__main__":
    main()
