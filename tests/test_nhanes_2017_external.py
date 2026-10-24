"""No-network 2017-18 NHANES regression tests with synthetic public-style fields."""
import unittest

import pandas as pd

from scripts.evaluate_external_nhanes_2017_18 import (
    CODEBOOK, SOURCE, SOURCE_SHA256, score,
)
from sleep_project.duration_v2 import SleepDurationAssessmentV2


class Nhanes2017Tests(unittest.TestCase):
    def test_source_identity_is_cycle_specific(self):
        self.assertIn("/2017/", SOURCE)
        self.assertIn("SLQ_J.XPT", SOURCE)
        self.assertIn("/2017/", CODEBOOK)
        self.assertEqual(len(SOURCE_SHA256), 64)

    def test_cycle_metadata_and_unmeasured_outcome(self):
        model = SleepDurationAssessmentV2("models/sleep_duration_multiclass_v2.json")
        records = pd.DataFrame({
            "SLQ300": [b"23:00", b"02:00", b"12:00"],
            "SLQ310": [b"07:00", b"06:00", b"06:00"],
            "SLD012": [8.0, 4.0, 18.0],
            "SLQ050": [2.0, 1.0, 2.0],
            "SLQ120": [2.0, 4.0, 1.0],
        })
        output = score(records, model)
        self.assertEqual(output["external_population_records"], 3)
        self.assertEqual(output["eligible_records"], 2)
        self.assertTrue(output["external_2017_18_survey"])
        self.assertFalse(output["external_2015_16_survey"])
        self.assertFalse(output["independently_measured_outcome"])
        self.assertFalse(output["medical_validation"])
        self.assertFalse(output["survey_weighted_population_inference"])
        self.assertIn("falling asleep", output["important_questionnaire_mismatch"])
        self.assertEqual(
            output["source_codebook"], CODEBOOK
        )


if __name__ == "__main__":
    unittest.main()
