"""Reproducible feasibility screen for non-diagnostic sleep survey outcomes.

This is an exploratory *target suitability* audit, not a clinical validation,
model acceptance certificate, or replacement for an independent holdout.
Run: python scripts/audit_sleep_targets.py --data notebooks/final_sleep_dataDev.csv
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


CONCENTRATION = "How often do you find it hard to concentrate due to lack of sleep?"
SLEEP_ONSET = "Do you have difficulty falling asleep?"
BREATHING = "Do you experience breathing difficulties while sleeping"
SLEEP_HOURS = "How many hours of sleep do you get on average per night?"
SIDE_EFFECTS = "Do you experience any of the following side effects from late sleeping?"

# Avoid the medical answer (old label source), the downstream side effects,
# coping responses, and ambiguous weight/height units as predictors.
SURVEY_FEATURES = [
    "Your Age",
    "What is your gender?",
    "What is your occupation?",
    "What time do you usually go to bed?",
    "What time do you usually wake up on working days?",
    "What time do you usually go to bed on weekends?",
    "What time do you usually wake up on weekends?",
    "How long does it take you to fall asleep after going to bed?",
    SLEEP_HOURS,
    "What are the main reasons you sleep late?",
    SLEEP_ONSET,
    BREATHING,
    "Do you experience restless legs or involuntary movements during sleep?",
    "How would you rate the comfort of your sleeping environment",
]
CORE_CONCENTRATION_FEATURES = [
    feature for feature in SURVEY_FEATURES
    if feature not in (
        "What time do you usually go to bed on weekends?",
        "What time do you usually wake up on weekends?",
        "How long does it take you to fall asleep after going to bed?",
    )
]

CANDIDATES = [
    ("frequent_concentration", CONCENTRATION, ("Often", "Always"), None),
    ("frequent_sleep_onset_difficulty", SLEEP_ONSET, ("Often", "Always"), None),
    ("frequent_breathing_difficulty", BREATHING, ("Often", "Always"), None),
    ("short_sleep", SLEEP_HOURS, ("Less than 4 hours", "4-6 hours"), None),
    ("fatigue_side_effect", SIDE_EFFECTS, None, "Fatigue"),
    ("performance_side_effect", SIDE_EFFECTS, None, "Decreased academic or work performance"),
]


def model_candidates():
    return [
        ("majority", DummyClassifier(strategy="most_frequent")),
        ("logistic", make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(max_iter=300),
        )),
        ("random_forest", make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            RandomForestClassifier(
                n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
            ),
        )),
    ]


def evaluate(X, y, train_idx, test_idx, split_name, candidate_name):
    rows = []
    for model_name, model in model_candidates():
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        rows.append({
            "candidate": candidate_name,
            "split": split_name,
            "model": model_name,
            "n": int(len(y)),
            "positive_prevalence": round(float(y.mean()), 4),
            "accuracy": round(float(accuracy_score(y.iloc[test_idx], pred)), 4),
            "balanced_accuracy": round(
                float(balanced_accuracy_score(y.iloc[test_idx], pred)), 4
            ),
            "macro_f1": round(float(f1_score(
                y.iloc[test_idx], pred, average="macro"
            )), 4),
            "roc_auc": round(float(roc_auc_score(
                y.iloc[test_idx], proba
            )), 4),
        })
    return rows


def run(data):
    results = {
        "shape": list(data.shape),
        "exact_duplicate_rows": int(data.duplicated().sum()),
        "warning": (
            "Self-reported proxies only. No verified clinical sleep-health "
            "outcome, subject IDs, or independent-site test set supplied."
        ),
        "metrics": [],
    }
    for name, target, positive_values, positive_substring in CANDIDATES:
        df = data.loc[data[target].notna()].copy()
        if positive_substring is not None:
            y = df[target].astype(str).str.contains(
                positive_substring, regex=False
            ).astype(int)
        else:
            y = df[target].isin(positive_values).astype(int)
        features = [col for col in SURVEY_FEATURES if col != target]
        assert target not in features, "Direct target leakage"
        assert not any("medical conditions" in col.lower() for col in features)
        assert SIDE_EFFECTS not in features, "Outcome-proxy leakage"
        X = df[features].fillna("<missing>").astype(str)

        train, test = train_test_split(
            np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
        )
        results["metrics"].extend(
            evaluate(X, y, train, test, "stratified_random", name)
        )
        if name == "frequent_concentration":
            # Additional stress test to keep identical predictor profiles apart.
            X_core = df[CORE_CONCENTRATION_FEATURES].fillna("<missing>").astype(str)
            groups = pd.util.hash_pandas_object(X_core, index=False)
            group_train, group_test = next(
                GroupShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=42
                ).split(X_core, y, groups=groups)
            )
            results["core_feature_duplicate_profiles"] = int(groups.duplicated().sum())
            results["metrics"].extend(
                evaluate(
                    X_core, y, group_train, group_test,
                    "core_feature_group_holdout", name
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", default="notebooks/final_sleep_dataDev.csv",
        help="Path to the checked-in 20-question survey CSV"
    )
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    print(json.dumps(run(data), indent=2))


if __name__ == "__main__":
    main()
