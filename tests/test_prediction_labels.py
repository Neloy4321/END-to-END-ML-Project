"""Regression test: UI labels must match the training-time LabelEncoder."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import LabelEncoder

from sleep_project.pipeline.prediction_pipeline import SleepClassifier


class PredictionLabelMappingTests(unittest.TestCase):
    def test_decodes_using_persisted_target_encoder(self):
        encoder = LabelEncoder().fit([
            "No Condition",
            "Sleep/Respiratory Disorders",
            "Health Issues",
            "Mental Health Issues",
            "Others",
        ])
        classifier = SleepClassifier.__new__(SleepClassifier)
        classifier._last_loaded_model = SimpleNamespace(
            preprocessing_object={"target_encoder": encoder}
        )

        with patch.object(
            SleepClassifier,
            "predict",
            return_value=np.array([0, 1, 2, 3, 4]),
        ):
            labels = classifier.predict_labels(dataframe=None)

        self.assertEqual(
            labels.tolist(),
            [
                "Health Issues",
                "Mental Health Issues",
                "No Condition",
                "Others",
                "Sleep/Respiratory Disorders",
            ],
        )


if __name__ == "__main__":
    unittest.main()
