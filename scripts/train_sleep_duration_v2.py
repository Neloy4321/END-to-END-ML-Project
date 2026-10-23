"""Reproducible, non-diagnostic four-category *survey answer* model.

Train: python scripts/train_sleep_duration_v2.py --data notebooks/final_sleep_dataDev.csv
Stream: git show HEAD:notebooks/final_sleep_dataDev.csv | python scripts/train_sleep_duration_v2.py --data - --output -
Requires pandas==2.2.3 numpy==2.3.5 scikit-learn==1.8.0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
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
RESPONSES = ("Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours")
TEMPERATURES = (0.75, 0.85, 1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0)


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def ranked_log_loss(labels, scores):
    ordered = sorted(RESPONSES)
    return log_loss(labels, scores[:, [RESPONSES.index(value) for value in ordered]], labels=ordered)


def metrics(labels, scores):
    prediction = np.asarray(RESPONSES)[scores.argmax(axis=1)]
    confidence = scores.max(axis=1)
    correct = (prediction == np.asarray(labels)).astype(float)
    ece = 0.0
    for left, right in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        members = (confidence >= left) & (confidence < right if right < 1 else confidence <= right)
        if members.any():
            ece += members.mean() * abs(confidence[members].mean() - correct[members].mean())
    return {
        "accuracy": round(float(accuracy_score(labels, prediction)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, prediction)), 6),
        "macro_f1": round(float(f1_score(labels, prediction, average="macro")), 6),
        "log_loss": round(float(ranked_log_loss(labels, scores)), 6),
        "top_label_ece_10_bins": round(float(ece), 6),
        "confusion_matrix_response_order": confusion_matrix(labels, prediction, labels=RESPONSES).tolist(),
    }


def estimator(columns, frame, labels, train_idx):
    pipe = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=500, random_state=42),
    )
    pipe.fit(frame.iloc[train_idx][list(columns)], labels.iloc[train_idx])
    return pipe


def probabilities(pipe, columns, frame, indices):
    uncalibrated = pipe.predict_proba(frame.iloc[indices][list(columns)])
    permutation = [list(pipe.steps[1][1].classes_).index(category) for category in RESPONSES]
    return uncalibrated[:, permutation]


def coefficients(pipe, columns):
    enc, lr = pipe.steps[0][1], pipe.steps[1][1]
    order = [list(lr.classes_).index(value) for value in RESPONSES]
    weights = {column: {} for column in columns}
    offset = 0
    for column, choices in zip(columns, enc.categories_):
        for i, choice in enumerate(choices):
            weights[column][str(choice)] = [
                float(lr.coef_[cls, offset + i]) for cls in order
            ]
        offset += len(choices)
    assert offset == lr.coef_.shape[1]
    return [float(lr.intercept_[cls]) for cls in order], weights


def train(data):
    required = list(FEATURES) + [TARGET]
    if any(column not in data for column in required):
        raise ValueError("Dataset is missing required survey columns")
    if data[required].isna().any().any():
        raise ValueError("Missing survey/target answers; review before training")
    if not set(data[TARGET].unique()).issubset(set(RESPONSES)):
        raise ValueError("Unrecognized sleep-duration response")
    assert TARGET not in FEATURES
    assert all("medical" not in feature.lower() for feature in FEATURES)
    frame = data[list(FEATURES)].astype(str)
    labels = data[TARGET].astype(str)
    groups = pd.util.hash_pandas_object(frame, index=False)
    train_indices, unused_indices = next(GroupShuffleSplit(
        n_splits=1, test_size=0.30, random_state=42
    ).split(frame, labels, groups=groups))
    validation_local, test_local = next(GroupShuffleSplit(
        n_splits=1, test_size=0.50, random_state=73
    ).split(frame.iloc[unused_indices], labels.iloc[unused_indices],
            groups=groups.iloc[unused_indices]))
    validation_indices = unused_indices[validation_local]
    test_indices = unused_indices[test_local]
    assert not (set(groups.iloc[train_indices]) & set(groups.iloc[validation_indices]))
    assert not (set(groups.iloc[train_indices]) & set(groups.iloc[test_indices]))
    assert not (set(groups.iloc[validation_indices]) & set(groups.iloc[test_indices]))

    models = {
        "bed_wake_only": (FEATURES[:2], estimator(FEATURES[:2], frame, labels, train_indices)),
        "all_six": (FEATURES, estimator(FEATURES, frame, labels, train_indices)),
    }
    validation = {
        name: metrics(labels.iloc[validation_indices],
                      probabilities(pipe, columns, frame, validation_indices))
        for name, (columns, pipe) in models.items()
    }
    # Select ONLY by validation log-loss; the test partition stays untouched.
    selected = min(models, key=lambda name: (validation[name]["log_loss"], len(models[name][0])))
    columns, model = models[selected]
    validation_scores = probabilities(model, columns, frame, validation_indices)
    best_temperature = min(TEMPERATURES, key=lambda temperature: ranked_log_loss(
        labels.iloc[validation_indices],
        softmax(np.log(np.clip(validation_scores, 1e-15, 1)) / temperature),
    ))
    # sklearn predict_proba is exactly softmax(decision_function) for multi-class LR.
    raw_test = probabilities(model, columns, frame, test_indices)
    test_scores = softmax(np.log(np.clip(raw_test, 1e-15, 1)) / best_temperature)
    intercepts, weights = coefficients(model, columns)

    # Proof that JSON weights + pure NumPy softmax reproduce the deployed scorer.
    reconstructed_logits = np.tile(intercepts, (len(test_indices), 1))
    for column in columns:
        reconstructed_logits += np.array([
            weights[column].get(value, [0.0] * len(RESPONSES))
            for value in frame.iloc[test_indices][column]
        ])
    assert np.allclose(softmax(reconstructed_logits), raw_test, atol=1e-9)
    assert np.allclose(softmax(reconstructed_logits / best_temperature), test_scores, atol=1e-9)

    other_name = "all_six" if selected == "bed_wake_only" else "bed_wake_only"
    other_columns, other_model = models[other_name]
    return {
        "schema_version": 2,
        "task": "predict_self_reported_sleep_duration_category",
        "disclaimer": "Model-estimated survey answer only. Not an objective sleep measurement, disease diagnosis, validated risk assessment, or treatment recommendation.",
        "target_column": TARGET,
        "response_classes": list(RESPONSES),
        "feature_columns": list(columns),
        "all_form_columns": list(FEATURES),
        "category_options": {col: sorted(frame[col].unique().tolist()) for col in FEATURES},
        "intercepts": intercepts,
        "weights": weights,
        "temperature": float(best_temperature),
        "evaluation": {
            "protocol": "70/15/15 seeded split by exact 6-answer profile; preprocessing learned only on train; model and temperature chosen on validation; final metrics on untouched test",
            "seed_train": 42,
            "seed_validation_test": 73,
            "total_records": int(len(frame)),
            "train_records": int(len(train_indices)),
            "validation_records": int(len(validation_indices)),
            "test_records": int(len(test_indices)),
            "unique_profiles": int(groups.nunique()),
            "class_counts": {label: int((labels == label).sum()) for label in RESPONSES},
            "validation_candidates": validation,
            "model_selected_on_validation": selected,
            "temperature_selected_on_validation": float(best_temperature),
            "held_out_test": metrics(labels.iloc[test_indices], test_scores),
            "other_model_held_out_test": {
                "name": other_name,
                "metrics": metrics(labels.iloc[test_indices], probabilities(
                    other_model, other_columns, frame, test_indices)),
            },
            "limitations": "Survey self-report; no verified individual/time/site separation, external dataset, true measured sleep duration, or clinical validation. Repeated/similar profiles may exist across partitions; schedule-response association is not causal. Temperature fitted internally and must not be treated as clinical calibration.",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="notebooks/final_sleep_dataDev.csv")
    parser.add_argument("--output", default="models/sleep_duration_multiclass_v2.json")
    args = parser.parse_args()
    data = pd.read_csv(sys.stdin if args.data == "-" else args.data)
    result = json.dumps(train(data), ensure_ascii=False, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(result)
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(result, encoding="utf-8")
        print("Model saved to " + str(out), file=sys.stderr)


if __name__ == "__main__":
    main()
