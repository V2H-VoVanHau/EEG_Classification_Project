import unittest

import pandas as pd

from eegoddball.config import DATA_VERSION, get_folds_hash
from eegoddball.evaluate import (
    calculate_metrics,
    check_same_test_sets,
)


class TestEvaluate(unittest.TestCase):
    def setUp(self):
        rows = []

        # Hai mô hình cùng dự đoán trên bốn mẫu test
        for model in ["TEST_LDA", "TEST_EEGNet"]:
            for trial_id, label, score in [
                (0, 0, 0.1),
                (1, 0, 0.4),
                (2, 1, 0.6),
                (3, 1, 0.9),
            ]:
                rows.append({
                    "scheme": "within_subject",
                    "subject": "01",
                    "fold": "fold_0",
                    "model": model,
                    "seed": 42,
                    "trial_id": trial_id,
                    "y_true": label,
                    "y_score": score,
                    "data_version": DATA_VERSION,
                    "folds_hash": get_folds_hash(),
                })

        self.predictions = pd.DataFrame(rows)

    def test_same_test_sets_pass(self):
        check_same_test_sets(self.predictions.copy())

    def test_perfect_predictions(self):
        metrics, roc_data = calculate_metrics(
            self.predictions.copy()
        )

        self.assertEqual(len(metrics), 2)
        self.assertFalse(roc_data.empty)

        for row in metrics.itertuples():
            self.assertAlmostEqual(row.balanced_accuracy, 1.0)
            self.assertAlmostEqual(row.roc_auc, 1.0)
            self.assertEqual(
                (row.tn, row.fp, row.fn, row.tp),
                (2, 0, 0, 2),
            )

    def test_reversed_predictions(self):
        data = self.predictions.copy()
        data["y_score"] = 1 - data["y_score"]

        metrics, _ = calculate_metrics(data)

        for row in metrics.itertuples():
            self.assertAlmostEqual(row.balanced_accuracy, 0.0)
            self.assertAlmostEqual(row.roc_auc, 0.0)
            self.assertEqual(
                (row.tn, row.fp, row.fn, row.tp),
                (0, 2, 2, 0),
            )

    def test_constant_predictions(self):
        data = self.predictions.copy()
        data["y_score"] = 0.2

        metrics, _ = calculate_metrics(data)

        for row in metrics.itertuples():
            self.assertAlmostEqual(row.balanced_accuracy, 0.5)
            self.assertAlmostEqual(row.roc_auc, 0.5)

    def test_missing_trial_rejected(self):
        data = self.predictions.drop(index=0).copy()

        with self.assertRaises(ValueError):
            check_same_test_sets(data)

    def test_duplicate_prediction_rejected(self):
        data = pd.concat(
            [self.predictions, self.predictions.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            check_same_test_sets(data)

    def test_wrong_hash_rejected(self):
        data = self.predictions.copy()
        data["folds_hash"] = "wrong_hash"

        with self.assertRaises(ValueError):
            check_same_test_sets(data)

    def test_wrong_version_rejected(self):
        data = self.predictions.copy()
        data["data_version"] = "wrong_version"

        with self.assertRaises(ValueError):
            check_same_test_sets(data)

    def test_different_true_labels_rejected(self):
        data = self.predictions.copy()
        data.loc[0, "y_true"] = 1

        with self.assertRaises(ValueError):
            check_same_test_sets(data)

    def test_invalid_score_rejected(self):
        for score in [-0.1, 1.1, float("nan"), float("inf")]:
            with self.subTest(score=score):
                data = self.predictions.copy()
                data.loc[0, "y_score"] = score

                with self.assertRaises(ValueError):
                    check_same_test_sets(data)


if __name__ == "__main__":
    unittest.main()