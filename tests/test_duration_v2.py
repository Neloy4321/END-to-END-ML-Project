"""Four-class contract, reproducibility, inference, and HTTP regression tests."""
import json
import math
from pathlib import Path
import unittest

from sleep_project.behavioral_assessment import FORM_FIELDS, InvalidAssessmentInput
from sleep_project.duration_v2 import CLASSES, SleepDurationAssessmentV2

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sleep_duration_multiclass_v2.json"


class DurationV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assessor = SleepDurationAssessmentV2(MODEL_PATH)

    def sample(self):
        return {item["key"]: item["options"][0] for item in self.assessor.questions()}

    def test_schema_and_no_target_or_medical_leakage(self):
        m = self.assessor.model
        self.assertEqual(m["schema_version"], 2)
        self.assertEqual(m["response_classes"], list(CLASSES))
        self.assertEqual(len(m["all_form_columns"]), 6)
        self.assertEqual(len(set(m["all_form_columns"])), 6)
        self.assertNotIn(m["target_column"], m["feature_columns"])
        self.assertFalse(any("medical" in column.lower() for column in m["feature_columns"]))
        self.assertEqual(m["feature_columns"], m["all_form_columns"][:2])
        self.assertEqual(sum(int(v) for v in m["evaluation"]["class_counts"].values()), 52610)

    def test_form_uses_exact_survey_category_values(self):
        for item in self.assessor.questions():
            self.assertCountEqual(item["options"], self.assessor.model["category_options"][item["column"]])
            self.assertEqual(item["used_by_model"], item["key"] in {"bed_time", "wake_time"})

    def test_probabilities_and_descriptive_context(self):
        answers = self.sample()
        answers["difficulty_sleep"] = "Often"
        answers["breathing_problem"] = "Always"
        answers["restless_legs"] = "Never"
        result = self.assessor.predict(answers)
        probabilities = [item["probability"] for item in result["category_probabilities"]]
        self.assertEqual(len(probabilities), 4)
        self.assertAlmostEqual(sum(probabilities), 1, places=12)
        self.assertEqual(len(result["reported_frequent_experiences"]), 2)
        self.assertAlmostEqual(
            result["short_response_percent"] + result["longer_response_percent"],
            100, delta=0.1
        )
        self.assertEqual(result["predicted_category"], CLASSES[probabilities.index(max(probabilities))])
        self.assertEqual(len(result["contributors"]), 2)

    def test_scores_match_exported_logits(self):
        answers = self.sample()
        result = self.assessor.predict(answers)
        model = self.assessor.model
        logits = list(model["intercepts"])
        for key, col, _ in FORM_FIELDS:
            if col in model["feature_columns"]:
                addition = model["weights"][col][answers[key]]
                logits = [a + b for a, b in zip(logits, addition)]
        e = [math.exp((v - max(logits)) / model["temperature"]) for v in logits]
        expected = [v / sum(e) for v in e]
        for row, prob in zip(result["category_probabilities"], expected):
            self.assertAlmostEqual(row["probability"], prob, places=12)

    def test_validation_rejects_missing_non_string_or_unseen_option(self):
        answers = self.sample()
        for variant in (
            {**answers, "bed_time": "11PM"},
            {**answers, "wake_time": None},
            {key: val for key, val in answers.items() if key != "breathing_problem"},
        ):
            with self.assertRaises(InvalidAssessmentInput):
                self.assessor.predict(variant)
        with self.assertRaises(InvalidAssessmentInput):
            self.assessor.predict(None)

    def test_comparison_keeps_other_answers_unchanged_and_no_mutation(self):
        answers = self.sample()
        before = dict(answers)
        comparison = self.assessor.compare(answers, "After 2 AM", "Before 6 AM")
        self.assertEqual(answers, before)
        self.assertEqual(comparison["original"]["chosen"], answers)
        self.assertEqual(comparison["scenario"]["chosen"]["bed_time"], "After 2 AM")
        self.assertEqual(comparison["scenario"]["chosen"]["wake_time"], "Before 6 AM")
        self.assertNotEqual(
            comparison["original"]["category_probabilities"],
            comparison["scenario"]["category_probabilities"],
        )
        for key in ("difficulty_sleep", "breathing_problem", "restless_legs", "sleep_environment"):
            self.assertEqual(comparison["scenario"]["chosen"][key], answers[key])
        with self.assertRaises(InvalidAssessmentInput):
            self.assessor.compare(answers, "Not available", "Before 6 AM")

    def test_retrain_exactly_reproduces_checked_in_model(self):
        import pandas as pd
        from scripts.train_sleep_duration_v2 import train

        source = ROOT / "notebooks" / "final_sleep_dataDev.csv"
        fresh = train(pd.read_csv(source))
        saved = self.assessor.model
        self.assertEqual(fresh["evaluation"], saved["evaluation"])
        self.assertEqual(fresh["feature_columns"], saved["feature_columns"])
        self.assertEqual(fresh["temperature"], saved["temperature"])
        self.assertEqual(fresh["category_options"], saved["category_options"])
        for a, b in zip(fresh["intercepts"], saved["intercepts"]):
            self.assertAlmostEqual(a, b, places=8)
        for column in fresh["feature_columns"]:
            for category in fresh["weights"][column]:
                for a, b in zip(fresh["weights"][column][category],
                                saved["weights"][column][category]):
                    self.assertAlmostEqual(a, b, places=8)


class HttpV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app import app
        cls.client = TestClient(app)
        cls.assessor = SleepDurationAssessmentV2(MODEL_PATH)

    def sample(self):
        return {item["key"]: item["options"][0] for item in self.assessor.questions()}

    def test_html_and_report_and_safety(self):
        with self.client as client:
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Sleep Survey Explorer", response.text)
            self.assertIn("no-store", response.headers["cache-control"])
            result = client.post("/assess", data=self.sample())
            self.assertEqual(result.status_code, 200)
            self.assertIn("Model-estimated questionnaire response", result.text)
            self.assertIn("What if you selected a different schedule?", result.text)
            self.assertIn("Print / Save results as PDF", result.text)

    def test_compare_api_validation_and_legacy_retirement(self):
        with self.client as client:
            body = {
                "answers": self.sample(),
                "bed_time": "After 2 AM",
                "wake_time": "Before 6 AM",
            }
            result = client.post("/api/compare", json=body)
            self.assertEqual(result.status_code, 200)
            self.assertEqual(len(result.json()["scenario"]["category_probabilities"]), 4)
            body["bed_time"] = "unseen"
            self.assertEqual(client.post("/api/compare", json=body).status_code, 422)
            self.assertEqual(client.post("/api/compare", content="{", headers={"Content-Type": "application/json"}).status_code, 400)
            self.assertEqual(client.post("/api/compare", content="x" * 4097).status_code, 413)
            self.assertEqual(client.post("/assess", data={"bed_time": "bogus"}).status_code, 422)
            self.assertEqual(client.post("/predict").status_code, 410)
            health = client.get("/healthz").json()
            self.assertEqual(health["version"], 2)
            self.assertEqual(health["status"], "ok")


if __name__ == "__main__":
    unittest.main()
