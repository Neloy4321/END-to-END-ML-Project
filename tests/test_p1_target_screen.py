"""P1 target gate invariants on synthetic answers; no respondent records."""
import unittest

import numpy as np
import pandas as pd

from scripts.screen_p1_targets import (
    FEATURES, TARGETS, assert_safe_schema, grouped_indices, label_for, run,
)


class TargetGateTests(unittest.TestCase):
    @staticmethod
    def survey(n=420):
        rng = np.random.default_rng(17)
        df = pd.DataFrame({
            col: rng.choice(
                [f"answer-a-{j}", f"answer-b-{j}", f"answer-c-{j}"], size=n
            )
            for j, col in enumerate(FEATURES)
        })
        for column, label_type in set(TARGETS.values()):
            if label_type == "frequency":
                # These two fields have the same label scale but independent draws.
                if column not in df:
                    df[column] = rng.choice(
                        ["Never", "Rarely", "Sometimes", "Often", "Always"],
                        size=n, p=[0.2, 0.2, 0.2, 0.2, 0.2],
                    )
            else:
                # Shared side-effect question; both substrings can coexist.
                df[column] = [
                    ("Fatigue;Decreased academic or work performance" if j % 3 == 0
                     else "Fatigue" if j % 3 == 1 else "No effects")
                    for j in range(n)
                ]
        return df

    def test_group_profiles_do_not_cross_splits(self):
        df = self.survey()
        duplicated = pd.concat([df, df.iloc[:10]], ignore_index=True)
        train, validation, test = grouped_indices(duplicated)
        sets = [
            set(pd.util.hash_pandas_object(duplicated.iloc[index][list(FEATURES)],
                                           index=False))
            for index in (train, validation, test)
        ]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])

    def test_validated_label_and_missing_are_separate(self):
        x = pd.DataFrame({"question": ["Always", "Never", "", None, "unknown", "Often"]})
        valid, labels = label_for(x, "question", "frequency")
        self.assertEqual(valid.tolist(), [True, True, False, False, False, True])
        self.assertEqual(labels.tolist(), [1, 0, 0, 0, 0, 1])

    def test_screen_never_promotes_to_clinical_model(self):
        df = self.survey()
        assert_safe_schema(df)
        result = run(df)
        self.assertFalse(result["medical_or_clinical_validation"])
        self.assertFalse(result["externally_validated"])
        self.assertFalse(result["eligible_for_app_model_replacement"])
        self.assertEqual(set(result["targets"]), set(TARGETS))
        self.assertEqual(result["exact_response_rows_excluded"], 0)
        self.assertTrue(all(
            "self_report" in value["label_source"]
            for value in result["targets"].values()
        ))

    def test_missing_or_leaked_target_is_rejected(self):
        df = self.survey()
        self.assertNotIn(
            "How often do you find it hard to concentrate due to lack of sleep?",
            FEATURES,
        )
        col = next(iter(TARGETS.values()))[0]
        with self.assertRaisesRegex(ValueError, "Missing questionnaire columns"):
            assert_safe_schema(df.drop(columns=col))


if __name__ == "__main__":
    unittest.main()
