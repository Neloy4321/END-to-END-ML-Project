"""P0 source quality, P2 external mapping, P3 runtime privacy tests."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.audit_dataset_quality import audit
from scripts.evaluate_external_nhanes import (
    SOURCE_SHA256, bed_band, clock_minutes, duration_class, evaluate,
    summarize, wake_band,
)
from sleep_project.duration_v2 import CLASSES, SleepDurationAssessmentV2
from sleep_project.operations import RequestCounters, route_group

ROOT = Path(__file__).resolve().parents[1]


class DataEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = SleepDurationAssessmentV2(ROOT / "models/sleep_duration_multiclass_v2.json")

    def test_duplicate_not_assumed_same_participant_and_conflicting_target(self):
        columns = self.model.model["all_form_columns"]
        first = {col: self.model.model["category_options"][col][0] for col in columns}
        target = self.model.model["target_column"]
        first[target] = CLASSES[0]
        second = dict(first, **{target: CLASSES[1]})
        frame = pd.DataFrame([first, first.copy(), second])
        result = audit(frame, self.model)
        self.assertEqual(result["exact_complete_row_duplicates"], 1)
        self.assertEqual(result["six_answer_distinct_profiles"], 1)
        self.assertEqual(result["six_answer_profiles_with_disagreeing_duration_answers"], 1)
        self.assertTrue(result["no_verified_participant_identifier"])
        self.assertTrue(all(v == 0 for v in result["invalid_core_answer_counts"].values()))

    def test_official_external_hours_not_relabelled_by_predictions(self):
        self.assertEqual(len(SOURCE_SHA256), 64)
        self.assertEqual(clock_minutes(b"23:30"), 1410)
        self.assertIsNone(clock_minutes(b"99999"))
        self.assertEqual(bed_band(23 * 60), "10 PM - 12 AM")
        self.assertEqual(bed_band(2 * 60), "After 2 AM")
        self.assertIsNone(bed_band(12 * 60))
        self.assertEqual(wake_band(6 * 60), "6 AM - 8 AM")
        self.assertIsNone(wake_band(16 * 60))
        self.assertEqual(duration_class(3.5), CLASSES[0])
        self.assertEqual(duration_class(4.0), CLASSES[1])
        self.assertEqual(duration_class(6.0), CLASSES[2])
        self.assertEqual(duration_class(6.0, False), CLASSES[1])
        self.assertEqual(duration_class(8.0), CLASSES[2])
        self.assertEqual(duration_class(8.5), CLASSES[3])
        self.assertIsNone(duration_class(float("nan")))

    def test_external_metrics_denominators_and_no_test_set_tuning(self):
        frame = pd.DataFrame({
            "SLQ300": [b"23:00", b"02:00", b"12:00"],
            "SLQ310": [b"07:00", b"06:00", b"06:00"],
            "SLD012": [8.0, 4.0, 18.0],
            "SLQ050": [2.0, 1.0, 2.0],
            "SLQ120": [2.0, 4.0, 1.0],
        })
        result = evaluate(frame, self.model)
        self.assertEqual(result["external_population_records"], 3)
        self.assertEqual(result["eligible_records"], 2)
        self.assertEqual(result["exclusions"]["invalid_or_unmapped_clock"], 1)
        self.assertFalse(result["independently_measured_outcome"])
        self.assertFalse(result["medical_validation"])
        self.assertEqual(sum(result["primary_6_hours_in_6_to_8_band"]["support_by_class"].values()), 2)

    def test_ece_and_brier_perfect_prediction(self):
        p = [[1., 0., 0., 0.], [0., 0., 1., 0.]]
        summary = summarize([CLASSES[0], CLASSES[2]], [CLASSES[0], CLASSES[2]], p)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["ece_10_bins"], 0)
        self.assertEqual(summary["multiclass_brier"], 0)


class OpsTests(unittest.TestCase):
    def test_untrusted_path_never_leaks_to_metrics(self):
        c = RequestCounters()
        c.observe("/assess?medical_self_report=Asthma", 200, 0.3)
        c.observe("/random/John%20Doe", 404, 0.1)
        self.assertEqual(route_group("/random/John"), "other")
        out = c.exposition()
        self.assertNotIn("Asthma", out)
        self.assertNotIn("John", out)
        self.assertIn('route="other",status="4xx"', out)

    def test_health_and_protected_metrics_http(self):
        from fastapi.testclient import TestClient
        from app import app

        with patch.dict(os.environ, {"SLEEP_METRICS_TOKEN": "test-only-token"}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/readyz").json()["status"], "ready")
                self.assertEqual(client.get("/metrics").status_code, 401)
                good = client.get("/metrics", headers={"Authorization": "Bearer test-only-token"})
                self.assertEqual(good.status_code, 200)
                self.assertIn("sleep_http_requests_total", good.text)
                self.assertNotIn("Authorization", good.text)
                self.assertEqual(good.headers["cache-control"], "no-store")
                self.assertEqual(good.headers["x-frame-options"], "DENY")
                self.assertTrue(good.headers["x-request-id"])
                excessive = client.post("/assess", content=b"x" * 16385)
                self.assertEqual(excessive.status_code, 413)
        with patch.dict(os.environ, {"SLEEP_METRICS_TOKEN": ""}):
            with TestClient(app) as client:
                self.assertEqual(client.get("/metrics").status_code, 404)


if __name__ == "__main__":
    unittest.main()
