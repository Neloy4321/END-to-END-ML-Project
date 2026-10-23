"""Train an honest, non-diagnostic survey-response predictor on Neloy's CSV.

The target is the *self-reported* response category for nightly sleep duration:
"Less than 4 hours" or "4-6 hours" versus the other two categories.
This is NOT a verified medical or measured-sleep outcome.

python scripts/train_short_sleep.py --data notebooks/final_sleep_dataDev.csv
python scripts/train_short_sleep.py --data - --output -  # stream CSV/model JSON
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES = (
    "What time do you usually go to bed?",
    "What time do you usually wake up on working days?",
    "Do you have difficulty falling asleep?",
    "Do you experience breathing difficulties while sleeping",
    "Do you experience restless legs or involuntary movements during sleep?",
    "How would you rate the comfort of your sleeping environment",
)
TARGET = "How many hours of sleep do you get on average per night?"
POSITIVE = ("Less than 4 hours", "4-6 hours")
KNOWN_RESPONSES = set(POSITIVE) | {"6-8 hours", "More than 8 hours"}


def score_metrics(y_true, probabilities):
    predictions = probabilities >= 0.5
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 6),
        "short_sleep_recall": round(float(recall_score(y_true, predictions)), 6),
        "confusion_matrix_true_0_1": confusion_matrix(y_true, predictions).tolist(),
    }


def export_weights(estimator):
    encoder, logistic = estimator.steps[0][1], estimator.steps[1][1]
    coefficients = logistic.coef_[0]
    assert logistic.classes_.tolist() == [0, 1]
    result = {}
    offset = 0
    for name, categories in zip(FEATURES, encoder.categories_):
        result[name] = {
            str(category): float(weight)
            for category, weight in zip(
                categories, coefficients[offset: offset + len(categories)]
            )
        }
        offset += len(categories)
    assert offset == len(coefficients)
    return result, float(logistic.intercept_[0])


def train(data):
    needed = list(FEATURES) + [TARGET]
    missing = [field for field in needed if field not in data.columns]
    if missing:
        raise ValueError("Required survey columns absent: " + ", ".join(missing))
    if data[TARGET].isna().any():
        raise ValueError("Target contains missing responses; audit before training")
    unexpected = set(data[TARGET].astype(str).unique()) - KNOWN_RESPONSES
    if unexpected:
        raise ValueError("Unrecognized target responses: " + repr(sorted(unexpected)))

    # No sleep-duration answer, medical-answer-derived label, or downstream
    # side effects appear in the features, and no post-split preprocessing occurs.
    assert TARGET not in FEATURES
    assert all("medical" not in field.lower() for field in FEATURES)
    X = data[list(FEATURES)].fillna("<missing>").astype(str)
    y = data[TARGET].isin(POSITIVE).astype(int)
    groups = pd.util.hash_pandas_object(X, index=False)
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    ).split(X, y, groups=groups))
    assert not set(groups.iloc[train_idx]).intersection(groups.iloc[test_idx])

    model = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=300, random_state=42),
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    prob = model.predict_proba(X.iloc[test_idx])[:, 1]
    weights, intercept = export_weights(model)

    # Independent pure-Python serving must match sklearn's own probabilities.
    serving_scores = np.full(len(test_idx), intercept, dtype=float)
    for feature in FEATURES:
        serving_scores += np.array([
            weights[feature].get(value, 0.0) for value in X.iloc[test_idx][feature]
        ])
    serving_prob = 1.0 / (1.0 + np.exp(-serving_scores))
    assert np.allclose(serving_prob, prob, atol=1e-10)

    schedule = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=300, random_state=42),
    )
    schedule.fit(X.iloc[train_idx][list(FEATURES[:2])], y.iloc[train_idx])
    baseline_prob = schedule.predict_proba(
        X.iloc[test_idx][list(FEATURES[:2])]
    )[:, 1]

    return {
        "schema_version": 1,
        "task": "predict_self_reported_short_sleep_response",
        "disclaimer": (
            "Questionnaire-answer prediction, not measured sleep duration, "
            "medical diagnosis, disorder screening, or validated health-risk score."
        ),
        "target_column": TARGET,
        "positive_response_values": list(POSITIVE),
        "feature_columns": list(FEATURES),
        "intercept": intercept,
        "weights": weights,
        "evaluation": {
            "validation": "one internal GroupShuffleSplit by exact predictor profile",
            "seed": 42,
            "total_records": int(len(data)),
            "train_records": int(len(train_idx)),
            "test_records": int(len(test_idx)),
            "unique_predictor_profiles": int(groups.nunique()),
            "observed_positive_fraction": round(float(y.mean()), 6),
            "model": score_metrics(y.iloc[test_idx], prob),
            "bed_wake_only_baseline": score_metrics(
                y.iloc[test_idx], baseline_prob
            ),
            "limitations": (
                "Exact-profile exclusion is not respondent, time, or site "
                "independence. Sleep schedule overwhelmingly predicts the "
                "survey duration response. No clinical validation."
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="notebooks/final_sleep_dataDev.csv")
    parser.add_argument("--output", default="models/short_sleep_logistic.json")
    args = parser.parse_args()
    data = pd.read_csv(sys.stdin if args.data == "-" else args.data)
    payload = json.dumps(train(data), indent=2, ensure_ascii=False) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload, encoding="utf-8")
        print("Saved " + str(target), file=sys.stderr)


if __name__ == "__main__":
    main()
