import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from eegoddball.config import RESULTS_DIR
from eegoddball.evaluate import (
    load_predictions,
    check_same_test_sets,
    check_against_reference,
    compare_models,
)


class TestMonteCarlo(unittest.TestCase):
    def test_monte_carlo_matches_known_probability(self):
        rows = []

        # Ba chênh lệch đều bằng 0.125.
        # Xác suất đổi dấu hai phía chính xác là 2/8 = 0.25.
        for subject in ["01", "02", "03"]:
            for model, score in [("A", 0.625), ("B", 0.500)]:
                rows.append({
                    "scheme": "within_subject",
                    "subject": subject,
                    "model": model,
                    "balanced_accuracy": score,
                    "roc_auc": score,
                })

        data = pd.DataFrame(rows)

        # Ép chạy nhánh Monte Carlo ngay trên ví dụ nhỏ
        # có đáp án biết trước; không sửa config thật.
        with (
            patch(
                "eegoddball.evaluate.PERMUTATION_EXACT_MAX_N",
                0,
            ),
            patch(
                "eegoddball.evaluate.N_PERMUTATIONS",
                10000,
            ),
            patch(
                "eegoddball.evaluate.N_BOOTSTRAP",
                100,
            ),
        ):
            first = compare_models(data, "A", "B")
            second = compare_models(data, "A", "B")

        # Cùng seed phải tái lập được kết quả
        pd.testing.assert_frame_equal(first, second)

        for row in first.itertuples():
            self.assertEqual(
                row.permutation_method,
                "monte_carlo_sign_flip",
            )
            self.assertGreaterEqual(row.permutation_p_raw, 0)
            self.assertLessEqual(row.permutation_p_raw, 1)

            # Monte Carlo gần đáp án chính xác,
            # không yêu cầu bằng tuyệt đối.
            self.assertAlmostEqual(
                row.permutation_p_raw,
                0.25,
                delta=0.03,
            )


class TestReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dummy_dir = Path(RESULTS_DIR) / "dummy"

        frames = [
            load_predictions(
                dummy_dir / "predictions_tv3.csv"
            ),
            load_predictions(
                dummy_dir / "predictions_tv4.csv"
            ),
        ]

        cls.original = pd.concat(
            frames,
            ignore_index=True,
        )

    def test_original_matches_reference(self):
        check_against_reference(self.original.copy())

    def test_both_models_missing_same_trial_rejected(self):
        data = self.original.copy()

        # Chọn một khóa test và xóa ở TẤT CẢ mô hình/seed
        first = data.iloc[0]

        mask = (
            (data["scheme"] == first["scheme"])
            & (data["subject"] == first["subject"])
            & (data["fold"] == first["fold"])
            & (data["trial_id"] == first["trial_id"])
        )

        data = data.loc[~mask].copy()

        # Hai mô hình vẫn có cùng tập test nên kiểm tra này qua
        check_same_test_sets(data)

        # Nhưng nguồn tham chiếu phải phát hiện thiếu trial
        with self.assertRaisesRegex(ValueError, "lệch folds/epochs"):
            check_against_reference(data)

    def test_both_models_wrong_same_label_rejected(self):
        data = self.original.copy()
        first = data.iloc[0]

        mask = (
            (data["scheme"] == first["scheme"])
            & (data["subject"] == first["subject"])
            & (data["fold"] == first["fold"])
            & (data["trial_id"] == first["trial_id"])
        )

        # Đảo cùng một nhãn ở tất cả mô hình/seed
        data.loc[mask, "y_true"] = (
            1 - data.loc[mask, "y_true"]
        )

        check_same_test_sets(data)

        with self.assertRaisesRegex(ValueError, "sai nhãn thật"):
            check_against_reference(data)


if __name__ == "__main__":
    unittest.main()