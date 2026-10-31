import unittest

import pandas as pd

from eegoddball.evaluate import compare_models


def make_scores(scores_a, scores_b):
    """Tạo bảng điểm theo người để thử hàm thống kê."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Hai danh sách phải cùng số người.")

    rows = []

    for index, (a, b) in enumerate(zip(scores_a, scores_b)):
        subject = f"{index + 1:02d}"

        for model, score in [("A", a), ("B", b)]:
            rows.append({
                "scheme": "within_subject",
                "subject": subject,
                "model": model,
                "balanced_accuracy": score,
                "roc_auc": score,
            })

    return pd.DataFrame(rows)


class TestStatistics(unittest.TestCase):
    def test_equal_models(self):
        data = make_scores(
            [0.6, 0.7, 0.8],
            [0.6, 0.7, 0.8],
        )

        result = compare_models(data, "A", "B")

        for row in result.itertuples():
            self.assertAlmostEqual(
                row.mean_difference_a_minus_b, 0.0
            )
            self.assertAlmostEqual(row.bootstrap_ci95_low, 0.0)
            self.assertAlmostEqual(row.bootstrap_ci95_high, 0.0)
            self.assertAlmostEqual(row.wilcoxon_p_raw, 1.0)
            self.assertAlmostEqual(row.permutation_p_raw, 1.0)

    def test_known_constant_difference(self):
        # Mỗi người có A - B = 0.125.
        # Mọi mẫu bootstrap đều có trung bình 0.125.
        data = make_scores(
            [0.625, 0.750, 0.875],
            [0.500, 0.625, 0.750],
        )

        result = compare_models(data, "A", "B")

        for row in result.itertuples():
            self.assertEqual(row.n_subjects, 3)
            self.assertAlmostEqual(
                row.mean_difference_a_minus_b, 0.125
            )
            self.assertAlmostEqual(
                row.bootstrap_ci95_low, 0.125
            )
            self.assertAlmostEqual(
                row.bootstrap_ci95_high, 0.125
            )

            # Trong 8 tổ hợp đổi dấu, hai tổ hợp
            # có trị tuyệt đối trung bình >= quan sát.
            self.assertAlmostEqual(row.permutation_p_raw, 0.25)

    def test_swapping_models(self):
        data = make_scores(
            [0.8, 0.7, 0.9],
            [0.5, 0.6, 0.7],
        )

        ab = compare_models(data, "A", "B")
        ba = compare_models(data, "B", "A")

        for metric in ["balanced_accuracy", "roc_auc"]:
            first = ab.loc[ab["metric"] == metric].iloc[0]
            second = ba.loc[ba["metric"] == metric].iloc[0]

            self.assertAlmostEqual(
                first["mean_difference_a_minus_b"],
                -second["mean_difference_a_minus_b"],
            )
            self.assertAlmostEqual(
                first["bootstrap_ci95_low"],
                -second["bootstrap_ci95_high"],
            )
            self.assertAlmostEqual(
                first["bootstrap_ci95_high"],
                -second["bootstrap_ci95_low"],
            )
            self.assertAlmostEqual(
                first["wilcoxon_p_raw"],
                second["wilcoxon_p_raw"],
            )
            self.assertAlmostEqual(
                first["permutation_p_raw"],
                second["permutation_p_raw"],
            )

    def test_row_order_does_not_change_pairing(self):
        data = make_scores(
            [0.8, 0.7, 0.9],
            [0.5, 0.6, 0.7],
        )

        original = compare_models(data, "A", "B")

        shuffled = data.sample(
            frac=1,
            random_state=123,
        ).reset_index(drop=True)

        reordered = compare_models(shuffled, "A", "B")

        pd.testing.assert_frame_equal(original, reordered)

    def test_missing_subject_score_rejected(self):
        data = make_scores(
            [0.8, 0.7, 0.9],
            [0.5, 0.6, 0.7],
        )

        # Xóa riêng điểm của B ở người 02
        data = data.loc[
            ~(
                (data["subject"] == "02")
                & (data["model"] == "B")
            )
        ]

        with self.assertRaises(ValueError):
            compare_models(data, "A", "B")

    def test_duplicate_subject_score_rejected(self):
        data = make_scores(
            [0.8, 0.7, 0.9],
            [0.5, 0.6, 0.7],
        )

        data = pd.concat(
            [data, data.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            compare_models(data, "A", "B")

    def test_same_model_rejected(self):
        data = make_scores([0.8, 0.7], [0.5, 0.6])

        with self.assertRaises(ValueError):
            compare_models(data, "A", "A")

    def test_schemes_are_separate(self):
        within = make_scores(
            [0.625, 0.750, 0.875],
            [0.500, 0.625, 0.750],
        )

        cross = within.copy()
        cross["scheme"] = "cross_subject"

        # Đảo điểm A/B riêng trong cross
        cross["model"] = cross["model"].map(
            {"A": "B", "B": "A"}
        )

        data = pd.concat(
            [within, cross],
            ignore_index=True,
        )

        result = compare_models(data, "A", "B")

        self.assertEqual(len(result), 4)

        for row in result.itertuples():
            expected = (
                0.125
                if row.scheme == "within_subject"
                else -0.125
            )

            self.assertAlmostEqual(
                row.mean_difference_a_minus_b,
                expected,
            )
            self.assertEqual(row.n_subjects, 3)


if __name__ == "__main__":
    unittest.main()