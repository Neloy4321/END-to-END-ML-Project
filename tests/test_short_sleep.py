"""Contract, validation, and leakage regression tests for the new assessment."""
import json
from pathlib import Path
import unittest

from sleep_project.behavioral_assessment import (
    BehavioralSleepAssessment,
    FORM_FIELDS,
    InvalidAssessmentInput,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "models" / "short_sleep_logistic.json"


class BehavioralAssessmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assessor = BehavioralSleepAssessment(ARTIFACT)

    def normal_answers(self):
        return {q["key"]: q["options"][0] for q in self.assessor.questions()}

    def test_model_contract_and_outcome_leakage(self):
        model = self.assessor.model
        columns = model["feature_columns"]
        self.assertEqual(len(columns), 6)
        self.assertEqual(len(columns), len(set(columns)))
        self.assertNotIn(model["target_column"], columns)
        self.assertFalse(any("medical" in c.lower() for c in columns))
        self.assertFalse(any("side effects" in c.lower() for c in columns))
        self.assertEqual(columns, [column for _, column, _ in FORM_FIELDS])

    def test_all_form_options_from_fitted_model(self):
        for question in self.assessor.questions():
            self.assertTrue(question["options"])
            self.assertEqual(
                question["options"], list(self.assessor.model["weights"][question["column"]])
            )

    def test_valid_predict_and_frequent_answers_are_observed(self):
        answers = self.normal_answers()
        answers["difficulty_sleep"] = "Often"
        answers["breathing_problem"] = "Always"
        answers["restless_legs"] = "Never"
        result = self.assessor.predict(answers)
        self.assertTrue(0 <= result["short_sleep_response_score_percent"] <= 100)
        self.assertEqual(len(result["noted_frequent_symptoms"]), 2)
        self.assertEqual(
            result["short_sleep_response_likely"],
            result["short_sleep_response_score_percent"] >= 50,
        )

    def test_unknown_category_and_missing_field_are_rejected(self):
        with self.assertRaises(InvalidAssessmentInput):
            self.assessor.predict({**self.normal_answers(), "bed_time": "invalid"})
        answers = self.normal_answers()
        del answers["wake_time"]
        with self.assertRaises(InvalidAssessmentInput):
            self.assessor.predict(answers)

    def test_duration_is_not_submitted_as_a_predictor(self):
        result = self.assessor.predict({
            **self.normal_answers(),
            "sleep_hours": "Less than 4 hours",
            "medical_condition": "Asthma",
        })
        self.assertNotIn("sleep_hours", result["chosen"])
        self.assertNotIn("medical_condition", result["chosen"])

    def test_intuitive_schedule_direction(self):
        early = self.normal_answers()
        early["bed_time"] = "Before 10 PM"
        early["wake_time"] = "After 10 AM"
        late = dict(early)
        late["bed_time"] = "After 2 AM"
        late["wake_time"] = "Before 6 AM"
        self.assertLess(
            self.assessor.predict(early)["short_sleep_response_score_percent"],
            self.assessor.predict(late)["short_sleep_response_score_percent"],
        )

    def test_committed_model_is_reproducible_with_collected_csv(self):
        # More expensive than the contract checks (~3 seconds), but protects
        # model provenance and catches silent preprocessing/schema drift.
        import pandas as pd
        from scripts.train_short_sleep import train
        source = ROOT / "notebooks" / "final_sleep_dataDev.csv"
        fresh = train(pd.read_csv(source))
        saved = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        self.assertEqual(fresh["evaluation"], saved["evaluation"])
        self.assertEqual(fresh["feature_columns"], saved["feature_columns"])
        self.assertAlmostEqual(fresh["intercept"], saved["intercept"], places=8)
        for col in fresh["feature_columns"]:
            for label, coefficient in fresh["weights"][col].items():
                self.assertAlmostEqual(
                    coefficient, saved["weights"][col][label], places=8
                )


if __name__ == "__main__":
    unittest.main()
