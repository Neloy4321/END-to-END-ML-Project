"""V3 survey-insight evidence and frozen-V2 regression tests."""
import json
from pathlib import Path
import unittest

from sleep_project.behavioral_assessment import InvalidAssessmentInput
from sleep_project.duration_v2 import SleepDurationAssessmentV2
from sleep_project.survey_insights_v3 import SurveyInsightsV3

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "models" / "sleep_duration_multiclass_v2.json"
COHORTS = ROOT / "models" / "survey_cohorts_v3.json"


class SurveyInsightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.insights = SurveyInsightsV3(COHORTS)
        cls.assessor = SleepDurationAssessmentV2(V2)

    def sample_six(self):
        return {q["key"]: q["options"][0] for q in self.assessor.questions()}

    def test_cohort_artifact_schema_privacy_and_sums(self):
        model = self.insights.data
        self.assertEqual(model["source_record_count"], 52610)
        self.assertGreaterEqual(model["minimum_cohort_size"], 100)
        self.assertEqual(sum(model["overall"]["counts"].values()), 52610)
        for key in ("by_age", "by_gender", "by_occupation", "by_medical_self_report"):
            for cohort in model[key].values():
                self.assertGreaterEqual(cohort["n"], 100)
                self.assertEqual(sum(cohort["counts"].values()), cohort["n"])
                self.assertAlmostEqual(sum(cohort["percent"].values()), 100, delta=0.2)
        self.assertIn("Asthma", model["by_medical_self_report"])
        self.assertNotIn("Heart Disease", model["by_medical_self_report"])

    def test_eleven_optional_questions_and_valid_categories(self):
        sections = self.insights.sections()
        singles = [q for item in sections for q in item["single"]]
        multiples = [q for item in sections for q in item["multi"]]
        self.assertEqual(len(singles), 8)
        self.assertEqual(len(multiples), 3)
        self.assertEqual(len(singles) + len(multiples), 11)
        blank = self.insights.parse({})
        self.assertFalse(self.insights.report(self.sample_six(), blank)["supplied_optional_answers"])
        self.assertEqual(blank["medical_self_report"], "")

    def test_medical_comparison_is_descriptive_and_not_in_prediction(self):
        answers = self.sample_six()
        original = self.assessor.predict(answers)
        extras = self.insights.parse({
            "age": "18-30", "weekend_bed": "After 2 AM",
            "weekend_wake": "After 10 AM", "medical_self_report": "Asthma",
            "late_reasons": ["Social Media/Internet", "Study & Exam Purpose"],
            "side_effects": ["Fatigue"], "coping": ["Exercise"],
        })
        report = self.insights.report(answers, extras)
        self.assertTrue(report["supplied_optional_answers"])
        self.assertEqual(report["bed_shift"], "3 time band(s) later")
        self.assertEqual(len(report["cohorts"]), 2)
        self.assertEqual(len(report["selected_behaviours"]), 4)
        self.assertIn("Asthma", [cohort["answer"] for cohort in report["cohorts"]])
        self.assertEqual(self.assessor.predict(answers), original)
        self.assertNotIn("medical_self_report", original["chosen"])

    def test_invalid_optional_answers_are_rejected(self):
        for payload in (
            {"medical_self_report": "Unverified disease"},
            {"age": "110 years"},
            {"side_effects": ["Fatigue", "Fatigue"]},
            {"coping": ["unsupported side effect"]},
            {"late_reasons": "Social Media/Internet"},
        ):
            with self.assertRaises(InvalidAssessmentInput):
                self.insights.parse(payload)

    def test_saved_cohorts_reproducible_from_same_collected_csv(self):
        import pandas as pd
        from scripts.build_survey_cohorts_v3 import build
        fresh = build(pd.read_csv(ROOT / "notebooks" / "final_sleep_dataDev.csv"))
        self.assertEqual(fresh, self.insights.data)

    def test_frozen_v2_accuracy_and_model_contract(self):
        model = self.assessor.model
        self.assertEqual(model["schema_version"], 2)
        self.assertEqual(model["feature_columns"], model["all_form_columns"][:2])
        self.assertEqual(model["evaluation"]["test_records"], 9207)
        self.assertAlmostEqual(model["evaluation"]["held_out_test"]["accuracy"], 0.860975, places=6)
        self.assertAlmostEqual(model["evaluation"]["held_out_test"]["macro_f1"], 0.856607, places=6)


class SurveyHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        from app import app
        cls.client = TestClient(app)
        cls.assessor = SleepDurationAssessmentV2(V2)

    def core_six(self):
        return {q["key"]: q["options"][0] for q in self.assessor.questions()}

    def test_richer_form_and_descriptive_result(self):
        with self.client as client:
            home = client.get("/")
            self.assertEqual(home.status_code, 200)
            self.assertIn("eleven optional insight questions", home.text.lower())
            self.assertIn("medical_self_report", home.text)
            answers = self.core_six()
            answers.update({
                "age": "18-30", "medical_self_report": "Asthma",
                "weekend_bed": "After 2 AM", "weekend_wake": "After 10 AM",
                "late_reasons": "Social Media/Internet",
                "side_effects": "Fatigue",
                "coping": "Exercise",
            })
            result = client.post("/assess", data=answers)
            self.assertEqual(result.status_code, 200)
            self.assertIn("Your estimated response distribution", result.text)
            self.assertIn("Reported medical condition: Asthma", result.text)
            self.assertIn("Survey cohort comparisons", result.text)
            self.assertIn("model-estimated", result.text.lower())
            self.assertEqual(result.headers["cache-control"], "no-store")
            health = client.get("/healthz").json()
            self.assertEqual(health["version"], 2)
            self.assertEqual(health["dashboard_version"], 3)

    def test_empty_extras_preserve_old_form_and_unknown_rejected(self):
        with self.client as client:
            old = client.post("/assess", data=self.core_six())
            self.assertEqual(old.status_code, 200)
            self.assertIn("Want richer insights", old.text)
            invalid = self.core_six()
            invalid["medical_self_report"] = "A made-up condition"
            self.assertEqual(client.post("/assess", data=invalid).status_code, 422)
            other = self.core_six()
            other["late_reasons"] = "not-in-dataset"
            self.assertEqual(client.post("/assess", data=other).status_code, 422)
            self.assertEqual(client.post("/predict").status_code, 410)


if __name__ == "__main__":
    unittest.main()
