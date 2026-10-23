"""Lightweight, pickle-free inference for a self-reported survey answer.

This module depends only on the Python standard library. The JSON weights
are exported and verified against scikit-learn by scripts/train_short_sleep.py.
"""
import json
import math
from pathlib import Path


FORM_FIELDS = (
    ("bed_time", "What time do you usually go to bed?", "Usual bedtime"),
    (
        "wake_time",
        "What time do you usually wake up on working days?",
        "Usual wake-up time on working days",
    ),
    (
        "difficulty_sleep",
        "Do you have difficulty falling asleep?",
        "Difficulty falling asleep",
    ),
    (
        "breathing_problem",
        "Do you experience breathing difficulties while sleeping",
        "Breathing difficulties during sleep (self-reported)",
    ),
    (
        "restless_legs",
        "Do you experience restless legs or involuntary movements during sleep?",
        "Restless legs or movements during sleep",
    ),
    (
        "sleep_environment",
        "How would you rate the comfort of your sleeping environment",
        "Sleeping environment comfort (1–5)",
    ),
)


class InvalidAssessmentInput(ValueError):
    pass


class BehavioralSleepAssessment:
    def __init__(self, model_file):
        model = json.loads(Path(model_file).read_text(encoding="utf-8"))
        expected = [column for _, column, _ in FORM_FIELDS]
        if model.get("schema_version") != 1 or model.get("feature_columns") != expected:
            raise ValueError("Model feature/schema contract mismatch")
        if model.get("task") != "predict_self_reported_short_sleep_response":
            raise ValueError("Unexpected model task")
        if set(model.get("weights", {})) != set(expected):
            raise ValueError("Model is missing weights for a feature")
        if not isinstance(model.get("intercept"), (float, int)):
            raise ValueError("Model has no numeric intercept")
        for column in expected:
            weights = model["weights"][column]
            if not isinstance(weights, dict) or not weights:
                raise ValueError("Missing category weights: " + column)
            if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in weights.values()):
                raise ValueError("Invalid model coefficient")
        self.model = model

    def questions(self, selected=None):
        selected = selected or {}
        return [
            {
                "key": key,
                "column": column,
                "label": label,
                "options": list(self.model["weights"][column]),
                "selected": selected.get(key, ""),
            }
            for key, column, label in FORM_FIELDS
        ]

    def predict(self, submitted):
        score = float(self.model["intercept"])
        chosen = {}
        for key, column, _ in FORM_FIELDS:
            raw = submitted.get(key, "")
            value = raw.strip() if isinstance(raw, str) else ""
            categories = self.model["weights"][column]
            if value not in categories:
                raise InvalidAssessmentInput("Choose a valid option for " + key)
            chosen[key] = value
            score += float(categories[value])

        # Stable sigmoid even if the trained coefficients become large.
        if score >= 0:
            probability = 1.0 / (1.0 + math.exp(-score))
        else:
            e = math.exp(score)
            probability = e / (1.0 + e)

        return {
            "short_sleep_response_likely": probability >= 0.5,
            "short_sleep_response_score_percent": round(probability * 100, 1),
            "chosen": chosen,
            "noted_frequent_symptoms": [
                label
                for key, _, label in FORM_FIELDS[2:5]
                if chosen[key] in {"Often", "Always"}
            ],
        }
