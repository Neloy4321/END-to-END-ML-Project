"""Standard-library, schema-checked inference for four survey response categories.

No raw respondent records, pickle files, sklearn, or training data at serving time.
The multinomial weights and temperature are exported by train_sleep_duration_v2.py.
"""
import json
import math
from pathlib import Path

from .behavioral_assessment import FORM_FIELDS, InvalidAssessmentInput

ORDER = {
    "bed_time": ("Before 10 PM", "10 PM - 12 AM", "12 AM - 2 AM", "After 2 AM"),
    "wake_time": ("Before 6 AM", "6 AM - 8 AM", "8 AM - 10 AM", "After 10 AM"),
}
FREQUENCY = ("Never", "Rarely", "Sometimes", "Often", "Always")
CLASSES = ("Less than 4 hours", "4-6 hours", "6-8 hours", "More than 8 hours")


class SleepDurationAssessmentV2:
    def __init__(self, model_file):
        model = json.loads(Path(model_file).read_text(encoding="utf-8"))
        self.model = model
        columns = [column for _, column, _ in FORM_FIELDS]
        if model.get("schema_version") != 2 or model.get("task") != "predict_self_reported_sleep_duration_category":
            raise ValueError("Unsupported duration model version/task")
        if model.get("target_column") in model.get("feature_columns", ()):
            raise ValueError("Target leakage detected in features")
        if model.get("all_form_columns") != columns:
            raise ValueError("Form columns changed since training")
        if model.get("response_classes") != list(CLASSES):
            raise ValueError("Response categories changed")
        features = model.get("feature_columns")
        if not features or any(col not in columns for col in features) or len(features) != len(set(features)):
            raise ValueError("Invalid model feature list")
        if set(model.get("weights", {})) != set(features):
            raise ValueError("Coefficient columns do not match features")
        if set(model.get("category_options", {})) != set(columns):
            raise ValueError("Missing category options")
        if len(model.get("intercepts", [])) != len(CLASSES):
            raise ValueError("Invalid intercept vector")
        self.temperature = model.get("temperature")
        if not isinstance(self.temperature, (int, float)) or not (0 < self.temperature < 10):
            raise ValueError("Invalid temperature")
        for values in [model["intercepts"]] + [
            coefficients
            for column in features for coefficients in model["weights"][column].values()
        ]:
            if len(values) != len(CLASSES) or not all(
                isinstance(v, (int, float)) and math.isfinite(v) for v in values
            ):
                raise ValueError("Invalid coefficient vector")
        for col in columns:
            opts = model["category_options"][col]
            if not isinstance(opts, list) or not opts or len(opts) != len(set(opts)):
                raise ValueError("Invalid survey answer options")

    def questions(self, selected=None):
        selected = selected or {}
        questions = []
        for key, column, label in FORM_FIELDS:
            available = self.model["category_options"][column]
            preferred = ORDER.get(key, FREQUENCY if set(available) == set(FREQUENCY) else tuple(available))
            options = [value for value in preferred if value in available]
            questions.append({
                "key": key, "column": column, "label": label,
                "options": options,
                "selected": selected.get(key, ""),
                "used_by_model": column in self.model["feature_columns"],
            })
        return questions

    def validate(self, values):
        if not isinstance(values, dict):
            raise InvalidAssessmentInput("Survey answers must be an object.")
        clean = {}
        for key, column, _ in FORM_FIELDS:
            raw = values.get(key)
            answer = raw.strip() if isinstance(raw, str) else ""
            if answer not in self.model["category_options"][column]:
                raise InvalidAssessmentInput("Choose a valid option for " + key)
            clean[key] = answer
        return clean

    def _raw_scores(self, chosen):
        logits = [float(value) for value in self.model["intercepts"]]
        for key, column, _ in FORM_FIELDS:
            if column in self.model["feature_columns"]:
                coeff = self.model["weights"][column].get(chosen[key])
                if coeff:
                    logits = [total + value for total, value in zip(logits, coeff)]
        logits = [logit / self.temperature for logit in logits]
        maximum = max(logits)
        exponential = [math.exp(logit - maximum) for logit in logits]
        denominator = sum(exponential)
        return [value / denominator for value in exponential]

    def predict(self, answers):
        chosen = self.validate(answers)
        raw = self._raw_scores(chosen)
        rank = sorted(range(len(CLASSES)), key=lambda idx: (-raw[idx], idx))
        primary, alternate = rank[:2]
        contributors = []
        for key, column, label in FORM_FIELDS:
            if column in self.model["feature_columns"]:
                pair = self.model["weights"][column][chosen[key]]
                difference = (pair[primary] - pair[alternate]) / self.temperature
                contributors.append({
                    "label": label, "answer": chosen[key],
                    "logit_margin_vs_runner_up": round(difference, 3),
                    "favours_top": difference > 0,
                })
        contributors.sort(key=lambda item: -abs(item["logit_margin_vs_runner_up"]))
        return {
            "chosen": chosen,
            "predicted_category": CLASSES[primary],
            "alternative_category": CLASSES[alternate],
            "predicted_percent": round(raw[primary] * 100, 1),
            "uncertain": raw[primary] < 0.60,
            "category_probabilities": [
                {"label": label, "probability": probability, "percent": round(probability * 100, 1)}
                for label, probability in zip(CLASSES, raw)
            ],
            "short_response_percent": round(sum(raw[:2]) * 100, 1),
            "longer_response_percent": round(sum(raw[2:]) * 100, 1),
            "contributors": contributors,
            "reported_frequent_experiences": [
                label for key, _, label in FORM_FIELDS[2:5]
                if chosen[key] in {"Often", "Always"}
            ],
            "self_reported_environment": chosen["sleep_environment"],
            "model_inputs": [
                label for _, column, label in FORM_FIELDS if column in self.model["feature_columns"]
            ],
        }

    def compare(self, submitted, new_bed_time, new_wake_time):
        base = self.predict(submitted)
        scenario = dict(base["chosen"])
        scenario["bed_time"] = new_bed_time
        scenario["wake_time"] = new_wake_time
        alternative = self.predict(scenario)
        return {
            "original": base,
            "scenario": alternative,
            "short_response_change_pp": round(
                alternative["short_response_percent"] - base["short_response_percent"], 1
            ),
        }
