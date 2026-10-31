
import argparse

import numpy as np
import pandas as pd
import json
from collections import Counter

from eegoddball.config import DATA_VERSION, get_folds_hash

from pathlib import Path

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from eegoddball.config import (
    DATA_VERSION,
    DERIVATIVES_DIR,
    FOLDS_PATH,
    get_folds_hash,
    SEED,
    N_BOOTSTRAP,
    N_PERMUTATIONS,
    PERMUTATION_EXACT_MAX_N,
)
from itertools import product
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


REQUIRED_COLUMNS = [
    "scheme",
    "subject",
    "fold",
    "model",
    "seed",
    "trial_id",
    "y_true",
    "y_score",
    "data_version",
    "folds_hash",
]

# Định danh một lần chạy mô hình
RUN_COLUMNS = ["model", "seed"]

# Định danh một mẫu test
TEST_COLUMNS = ["scheme", "subject", "fold", "trial_id"]


def load_predictions(path):
    """Đọc CSV và giữ subject='01' thay vì chuyển thành số 1."""
    predictions = pd.read_csv(
        path,
        dtype={
            "scheme": "string",
            "subject": "string",
            "fold": "string",
            "model": "string",
            "data_version": "string",
            "folds_hash": "string",
        },
    )

    validate_predictions(predictions, source=str(path))
    return predictions


def validate_predictions(predictions, source="predictions"):
    """Kiểm tra cấu trúc và giá trị trong file dự đoán."""
    missing = set(REQUIRED_COLUMNS) - set(predictions.columns)

    if missing:
        raise ValueError(
            f"{source}: thiếu cột {sorted(missing)}"
        )

    if predictions.empty:
        raise ValueError(f"{source}: file không có dự đoán.")

    if predictions[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(
            f"{source}: có giá trị bị thiếu trong các cột bắt buộc."
        )

    text_columns = [
        "scheme", "subject", "fold", "model",
        "data_version", "folds_hash",
    ]

    for column in text_columns:
        values = predictions[column].astype(str)

        if values.str.strip().eq("").any():
            raise ValueError(
                f"{source}: cột {column} có chuỗi rỗng."
            )

        if values.ne(values.str.strip()).any():
            raise ValueError(
                f"{source}: cột {column} có khoảng trắng thừa."
            )

    # Chốt tên hai kịch bản dùng chung
    allowed_schemes = {"within_subject", "cross_subject"}

    if not predictions["scheme"].isin(allowed_schemes).all():
        raise ValueError(
            f"{source}: scheme phải là within_subject "
            "hoặc cross_subject."
        )

    # trial_id và seed phải là số nguyên không âm
    for column in ["trial_id", "seed"]:
        values = pd.to_numeric(
            predictions[column],
            errors="raise",
        )

        if (
            not np.isfinite(values.to_numpy(dtype=float)).all()
            or (values < 0).any()
            or (values % 1 != 0).any()
        ):
            raise ValueError(
                f"{source}: {column} phải là số nguyên không âm."
            )

        predictions[column] = values.astype("int64")

    # Nhãn nhị phân khi đánh giá:
    # standard=0, target=1
    labels = pd.to_numeric(
        predictions["y_true"],
        errors="raise",
    )

    if not labels.isin([0, 1]).all():
        raise ValueError(
            f"{source}: y_true phải là 0 hoặc 1."
        )

    predictions["y_true"] = labels.astype(int)

    # y_score là xác suất của lớp target
    scores = pd.to_numeric(
        predictions["y_score"],
        errors="raise",
    )

    if (
        not np.isfinite(scores.to_numpy(dtype=float)).all()
        or not scores.between(0, 1).all()
    ):
        raise ValueError(
            f"{source}: y_score phải hữu hạn và nằm trong [0, 1]."
        )

    predictions["y_score"] = scores.astype(float)

    key_columns = RUN_COLUMNS + TEST_COLUMNS

    if predictions.duplicated(key_columns).any():
        raise ValueError(
            f"{source}: có dự đoán trùng "
            "(model, seed, scheme, subject, fold, trial_id)."
        )


def check_same_test_sets(predictions):
    """Kiểm tra phiên bản, hash, tập test và nhãn giữa các run."""
    validate_predictions(predictions)

    # Đối chiếu cấu hình thực tế, không chỉ so CSV với nhau
    if not predictions["data_version"].eq(DATA_VERSION).all():
        raise ValueError(
            f"Sai data_version: yêu cầu {DATA_VERSION!r}."
        )

    expected_hash = get_folds_hash()

    if not predictions["folds_hash"].eq(expected_hash).all():
        raise ValueError(
            "folds_hash không khớp splits/folds.json hiện tại."
        )

    reference = None
    reference_run = None
    number_of_runs = 0

    for run, group in predictions.groupby(
        RUN_COLUMNS,
        sort=True,
    ):
        number_of_runs += 1

        # Mỗi khóa test gắn với một nhãn thật
        current = {
            tuple(row[:-1]): int(row[-1])
            for row in group[
                TEST_COLUMNS + ["y_true"]
            ].itertuples(index=False, name=None)
        }

        if reference is None:
            reference = current
            reference_run = run
            continue

        missing = set(reference) - set(current)
        extra = set(current) - set(reference)

        if missing or extra:
            raise ValueError(
                f"Run {run} lệch tập test so với {reference_run}: "
                f"thiếu {len(missing)}, thừa {len(extra)} mẫu."
            )

        if current != reference:
            raise ValueError(
                f"Run {run} có y_true khác {reference_run} "
                "trên cùng mẫu test."
            )

    if number_of_runs < 2:
        raise ValueError(
            "Cần ít nhất hai run (model, seed) để đối chiếu."
        )

    print("PASS: cấu trúc và giá trị dự đoán hợp lệ.")
    print(f"PASS: data_version = {DATA_VERSION}")
    print(f"PASS: folds_hash = {expected_hash}")
    print(
        f"PASS: {number_of_runs} run dùng cùng "
        f"{len(reference)} mẫu test và cùng y_true."
    )

def calculate_metrics(predictions, threshold=0.5):
    """Tính điểm và chuẩn bị dữ liệu ROC cho từng nhóm test."""
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Ngưỡng phân lớp phải nằm trong [0, 1].")

    validate_predictions(predictions)

    group_columns = [
        "scheme", "subject", "fold", "model", "seed",
    ]

    metric_rows = []
    roc_rows = []

    for key, group in predictions.groupby(
        group_columns, sort=True
    ):
        identifiers = dict(zip(group_columns, key))

        y_true = group["y_true"].to_numpy(dtype=int)
        y_score = group["y_score"].to_numpy(dtype=float)

        # Không âm thầm tính điểm khi nhóm test thiếu một lớp
        if set(np.unique(y_true)) != {0, 1}:
            raise ValueError(
                f"{identifiers}: tập test phải có cả "
                "standard và target để tính BAcc/ROC-AUC."
            )

        # Quy tắc: xác suất target >= threshold -> target
        y_pred = (y_score >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        metric_rows.append({
            **identifiers,
            "n_test": len(group),
            "n_standard": int(np.sum(y_true == 0)),
            "n_target": int(np.sum(y_true == 1)),
            "threshold": threshold,
            "balanced_accuracy": balanced_accuracy_score(
                y_true, y_pred
            ),
            "roc_auc": roc_auc_score(y_true, y_score),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "data_version": group["data_version"].iloc[0],
            "folds_hash": group["folds_hash"].iloc[0],
        })

        # Dữ liệu dùng để vẽ đường ROC
        fpr, tpr, _ = roc_curve(
            y_true,
            y_score,
            pos_label=1,
        )

        for x, y in zip(fpr, tpr):
            roc_rows.append({
                **identifiers,
                "fpr": float(x),
                "tpr": float(y),
            })

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(roc_rows),
    )
    
def check_against_reference(predictions):
    """Đối chiếu dự đoán với folds và epochs của phiên bản hiện tại."""
    import mne

    validate_predictions(predictions)

    folds = json.loads(
        Path(FOLDS_PATH).read_text(encoding="utf-8")
    )

    within = folds["within_subject"]
    cross = folds["cross_subject"]

    if not within or not cross:
        raise ValueError("folds.json thiếu within hoặc cross.")

    subjects = set(within)
    expected = {}

    def add_expected(scheme, subject, fold_name, trial_ids, labels):
        for trial_id in trial_ids:
            key = (
                scheme,
                subject,
                fold_name,
                int(trial_id),
            )

            if key in expected:
                raise ValueError(
                    f"Trùng mẫu trong bản tham chiếu: {key}"
                )

            expected[key] = int(labels.loc[trial_id])

    # Cross-subject: mỗi người phải thuộc test đúng một lần
    cross_subjects = [
        subject
        for fold in cross.values()
        for subject in fold["test_subjects"]
    ]

    if (
        set(cross_subjects) != subjects
        or any(
            count != 1
            for count in Counter(cross_subjects).values()
        )
        or any(
            not fold["test_subjects"]
            for fold in cross.values()
        )
    ):
        raise ValueError(
            "Cross folds phải phủ đúng các đối tượng, "
            "mỗi người thuộc test đúng một lần."
        )

    for sub in sorted(subjects):
        path = (
            Path(DERIVATIVES_DIR)
            / DATA_VERSION
            / f"sub-{sub}_task-oddball-epo.fif"
        )

        epochs = mne.read_epochs(
            path,
            preload=False,
            verbose=False,
        )

        metadata = epochs.metadata

        if (
            metadata is None
            or not {"trial_id", "condition"}.issubset(
                metadata.columns
            )
        ):
            raise ValueError(
                f"sub-{sub}: thiếu trial_id hoặc condition."
            )

        ids = pd.to_numeric(
            metadata["trial_id"],
            errors="raise",
        )

        if (
            not np.isfinite(ids.to_numpy(dtype=float)).all()
            or (ids < 0).any()
            or (ids % 1 != 0).any()
            or ids.duplicated().any()
        ):
            raise ValueError(
                f"sub-{sub}: trial_id không hợp lệ."
            )

        # Đối chiếu condition với mã sự kiện trong FIF
        mapped_codes = metadata["condition"].map({
            "standard": 1,
            "target": 2,
            "distractor": 3,
        })

        if (
            mapped_codes.isna().any()
            or not np.array_equal(
                mapped_codes.to_numpy(),
                epochs.events[:, 2],
            )
        ):
            raise ValueError(
                f"sub-{sub}: condition không khớp events."
            )

        binary = metadata.loc[
            metadata["condition"].isin(["standard", "target"])
        ].copy()

        if binary.empty:
            raise ValueError(
                f"sub-{sub}: không còn standard/target."
            )

        binary["trial_id"] = pd.to_numeric(
            binary["trial_id"]
        ).astype("int64")

        labels = binary.set_index("trial_id")[
            "condition"
        ].map({"standard": 0, "target": 1})

        available_ids = set(labels.index)

        # Dùng mã gốc trong folds.
        # Trial bị loại khỏi v1 không được đánh số lại.
        assigned_ids = []

        for fold_name, fold in within[sub].items():
            original_ids = fold["test_trial_ids"]

            if any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or i < 0
                for i in original_ids
            ):
                raise ValueError(
                    f"{sub}/{fold_name}: ID trong folds không hợp lệ."
                )

            assigned_ids.extend(original_ids)

            retained_ids = sorted(
                set(original_ids) & available_ids
            )

            add_expected(
                "within_subject",
                sub,
                fold_name,
                retained_ids,
                labels,
            )

        # Kiểm tra folds không trùng ID và không bỏ sót
        # các trial binary còn hiện diện trong epochs.
        if (
            len(assigned_ids) != len(set(assigned_ids))
            or not available_ids.issubset(set(assigned_ids))
        ):
            raise ValueError(
                f"sub-{sub}: within folds trùng hoặc thiếu trial."
            )

        for fold_name, fold in cross.items():
            if sub in fold["test_subjects"]:
                add_expected(
                    "cross_subject",
                    sub,
                    fold_name,
                    sorted(available_ids),
                    labels,
                )

    # Kiểm tra TỪNG mô hình/seed với nguồn tham chiếu
    for run, group in predictions.groupby(
        RUN_COLUMNS, sort=True
    ):
        actual = {
            tuple(row[:-1]): int(row[-1])
            for row in group[
                TEST_COLUMNS + ["y_true"]
            ].itertuples(index=False, name=None)
        }

        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)

        if missing or extra:
            raise ValueError(
                f"Run {run} lệch folds/epochs: "
                f"thiếu {len(missing)}, thừa {len(extra)} mẫu. "
                f"Ví dụ thiếu: {sorted(missing)[:3]}; "
                f"ví dụ thừa: {sorted(extra)[:3]}"
            )

        wrong_labels = [
            key
            for key in expected
            if actual[key] != expected[key]
        ]

        if wrong_labels:
            raise ValueError(
                f"Run {run}: sai nhãn thật so với epochs, "
                f"ví dụ {wrong_labels[:3]}"
            )

    print(
        f"PASS: từng run khớp {len(expected)} mẫu test "
        f"và nhãn thật từ folds/epochs {DATA_VERSION}."
    )
    
def summarize_by_subject(metrics):
    """Trung bình fold trước, trung bình seed sau."""
    score_columns = ["balanced_accuracy", "roc_auc"]

    if metrics.empty:
        raise ValueError("Bảng điểm rỗng.")

    if not np.isfinite(
        metrics[score_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError("Bảng điểm có giá trị không hữu hạn.")

    # Không được đếm một fold nhiều lần
    fold_key = [
        "scheme", "subject", "model", "seed", "fold",
    ]

    if metrics.duplicated(fold_key).any():
        raise ValueError("Bảng điểm bị trùng fold.")

    # Mỗi mô hình phải chạy cùng danh sách seed
    # cho mọi người trong cùng kịch bản.
    for (scheme, model), group in metrics.groupby(
        ["scheme", "model"]
    ):
        seed_sets = group.groupby("subject")["seed"].agg(
            lambda values: frozenset(values)
        )

        if len(set(seed_sets.tolist())) != 1:
            raise ValueError(
                f"{scheme}/{model}: danh sách seed "
                "không đồng nhất giữa các người."
            )

    # Trong một người/kịch bản, các model và seed
    # phải được tính trên cùng danh sách fold.
    for (scheme, subject), group in metrics.groupby(
        ["scheme", "subject"]
    ):
        fold_sets = group.groupby(
            ["model", "seed"]
        )["fold"].agg(lambda values: frozenset(values))

        if len(set(fold_sets.tolist())) != 1:
            raise ValueError(
                f"{scheme}/{subject}: các run lệch fold."
            )

    # Bước 1: trung bình fold trong mỗi seed
    by_seed = (
        metrics.groupby(
            ["scheme", "subject", "model", "seed"],
            as_index=False,
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            n_folds=("fold", "nunique"),
        )
    )

    # Bước 2: trung bình các seed trong mỗi người
    by_subject = (
        by_seed.groupby(
            ["scheme", "subject", "model"],
            as_index=False,
        )
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            n_seeds=("seed", "nunique"),
            n_folds=("n_folds", "first"),
        )
    )

    # Giữ thông tin nguồn trong các bảng tổng hợp
    for column in [
        "data_version", "folds_hash", "threshold",
    ]:
        if metrics[column].nunique(dropna=False) != 1:
            raise ValueError(
                f"Bảng điểm chứa nhiều giá trị {column}."
            )

        value = metrics[column].iloc[0]
        by_seed[column] = value
        by_subject[column] = value

    return by_seed, by_subject

def compare_models(by_subject, model_a, model_b):
    """So sánh ghép cặp theo người, riêng từng kịch bản."""
    if model_a == model_b:
        raise ValueError("Phải chọn hai mô hình khác nhau.")

    if N_BOOTSTRAP < 1 or N_PERMUTATIONS < 1:
        raise ValueError("Số lần lấy mẫu phải lớn hơn 0.")

    results = []

    for scheme, group in by_subject.groupby("scheme"):
        selected = group.loc[
            group["model"].isin([model_a, model_b])
        ].copy()

        if selected.duplicated(["subject", "model"]).any():
            raise ValueError(
                f"{scheme}: mỗi người/mô hình phải có một dòng."
            )

        for metric in ["balanced_accuracy", "roc_auc"]:
            paired = selected.pivot(
                index="subject",
                columns="model",
                values=metric,
            ).sort_index()

            if (
                model_a not in paired.columns
                or model_b not in paired.columns
            ):
                raise ValueError(
                    f"{scheme}: thiếu {model_a} hoặc {model_b}."
                )

            # Không âm thầm bỏ người bị thiếu điểm
            if paired[[model_a, model_b]].isna().any().any():
                raise ValueError(
                    f"{scheme}: hai mô hình không đủ cùng người."
                )

            a = paired[model_a].to_numpy(dtype=float)
            b = paired[model_b].to_numpy(dtype=float)

            if not np.isfinite(np.column_stack([a, b])).all():
                raise ValueError("Điểm so sánh không hữu hạn.")

            n = len(a)

            if n < 2:
                raise ValueError(
                    "Cần ít nhất hai người để chạy thử thống kê."
                )

            difference = a - b

            # Wilcoxon: làm tròn sai số số thực trước xếp hạng
            ranked_difference = np.round(difference, 12)

            if np.all(ranked_difference == 0):
                statistic = 0.0
                wilcoxon_p = 1.0
            else:
                test = wilcoxon(
                    ranked_difference,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic = float(test.statistic)
                wilcoxon_p = float(test.pvalue)

            # Bootstrap ghép cặp:
            # lấy mẫu lại các chênh lệch theo người
            boot_rng = np.random.default_rng(SEED)
            bootstrap_means = np.empty(N_BOOTSTRAP)

            for i in range(N_BOOTSTRAP):
                sample = boot_rng.choice(
                    difference,
                    size=n,
                    replace=True,
                )
                bootstrap_means[i] = sample.mean()

            ci_low, ci_high = np.quantile(
                bootstrap_means,
                [0.025, 0.975],
            )

            # Hoán vị ghép cặp: đổi dấu chênh lệch từng người.
            # Đây là kiểm định chênh lệch hai mô hình,
            # không phải hoán vị nhãn rồi huấn luyện lại.
            observed = abs(float(difference.mean()))
            tolerance = 1e-12

            if n <= PERMUTATION_EXACT_MAX_N:
                extreme = 0
                total = 0

                for signs in product([-1, 1], repeat=n):
                    permuted = abs(
                        float(
                            np.mean(
                                difference * np.asarray(signs)
                            )
                        )
                    )

                    extreme += int(
                        permuted >= observed - tolerance
                    )
                    total += 1

                permutation_p = extreme / total
                permutation_method = "exact_sign_flip"

            else:
                perm_rng = np.random.default_rng(SEED)
                extreme = 0

                for _ in range(N_PERMUTATIONS):
                    signs = perm_rng.choice([-1, 1], size=n)
                    permuted = abs(
                        float(np.mean(difference * signs))
                    )

                    extreme += int(
                        permuted >= observed - tolerance
                    )

                permutation_p = (
                    (extreme + 1) / (N_PERMUTATIONS + 1)
                )
                permutation_method = "monte_carlo_sign_flip"

            results.append({
                "scheme": scheme,
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "n_subjects": n,
                "mean_a": float(a.mean()),
                "mean_b": float(b.mean()),
                "mean_difference_a_minus_b": float(
                    difference.mean()
                ),
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_raw": wilcoxon_p,
                "permutation_p_raw": float(permutation_p),
                "permutation_method": permutation_method,
                "n_bootstrap": N_BOOTSTRAP,
                "random_seed": SEED,
            })

    if not results:
        raise ValueError("Không có dữ liệu để so sánh.")

    return pd.DataFrame(results)

def save_evaluation_plots(metrics, roc_data, output_dir):
    """Vẽ riêng từng nhóm test; không gộp điểm giữa các fold."""
    figure_dir = Path(output_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    group_columns = [
        "scheme", "subject", "fold", "model", "seed",
    ]

    manifest = []

    for index, row in enumerate(
        metrics.to_dict("records"),
        start=1,
    ):
        mask = np.ones(len(roc_data), dtype=bool)

        for column in group_columns:
            mask &= (
                roc_data[column].to_numpy() == row[column]
            )

        curve = roc_data.loc[mask]

        if curve.empty:
            raise ValueError(
                f"Không có dữ liệu ROC cho nhóm {row}"
            )

        matrix = np.array([
            [row["tn"], row["fp"]],
            [row["fn"], row["tp"]],
        ], dtype=int)

        fig, axes = plt.subplots(
            1, 2,
            figsize=(10, 4.5),
        )

        try:
            # ROC giữ nguyên thứ tự điểm do roc_curve trả về
            axes[0].plot(
                curve["fpr"],
                curve["tpr"],
                label=f"AUC = {row['roc_auc']:.3f}",
                linewidth=2,
            )

            axes[0].plot(
                [0, 1], [0, 1],
                linestyle="--",
                color="gray",
                label="Chance",
            )

            axes[0].set(
                xlabel="False positive rate",
                ylabel="True positive rate",
                title="ROC curve",
                xlim=(0, 1),
                ylim=(0, 1.02),
            )
            axes[0].legend(loc="lower right")
            axes[0].grid(alpha=0.2)

            # Hàng: nhãn thật; cột: nhãn dự đoán
            display = ConfusionMatrixDisplay(
                confusion_matrix=matrix,
                display_labels=["Standard", "Target"],
            )

            display.plot(
                ax=axes[1],
                cmap="Blues",
                colorbar=False,
                values_format="d",
            )

            axes[1].set_title(
                f"Confusion matrix | "
                f"BAcc = {row['balanced_accuracy']:.3f}\n"
                f"Threshold = {row['threshold']:g}"
            )

            title = (
                f"{row['scheme']} | sub-{row['subject']} | "
                f"{row['fold']} | {row['model']} | "
                f"seed={row['seed']}"
            )

            fig.suptitle(title, fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.91))

            filename = f"evaluation_{index:03d}.png"
            figure_path = figure_dir / filename

            fig.savefig(
                figure_path,
                dpi=300,
                bbox_inches="tight",
            )

            manifest.append({
                **{
                    column: row[column]
                    for column in group_columns
                },
                "figure": filename,
            })

        finally:
            plt.close(fig)

    # Bảng tra ảnh nào thuộc mô hình/người/fold nào
    pd.DataFrame(manifest).to_csv(
        figure_dir / "figure_manifest.csv",
        index=False,
    )

    print(
        f"\nĐã lưu {len(manifest)} ảnh ROC và ma trận "
        f"nhầm lẫn tại: {figure_dir}"
    )

def main():
    parser = argparse.ArgumentParser(
        description="Kiểm tra và chấm các CSV dự đoán."
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="Đường dẫn các file CSV dự đoán.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Thư mục lưu bảng điểm và dữ liệu ROC.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Ngưỡng phân lớp target, mặc định 0.5.",
    )
    
    parser.add_argument(
        "--model-a",
        required=True,
        help="Tên mô hình A trong CSV.",
    )

    parser.add_argument(
        "--model-b",
        required=True,
        help="Tên mô hình B trong CSV.",
    )
    
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Lưu ảnh ROC và ma trận nhầm lẫn.",
    )

    args = parser.parse_args()

    frames = [
        load_predictions(path)
        for path in args.files
    ]

    predictions = pd.concat(
        frames,
        ignore_index=True,
    )

    # Chỉ tính điểm sau khi vượt qua kiểm tra đầu vào
    check_same_test_sets(predictions)
    
    check_against_reference(predictions)

    metrics, roc_data = calculate_metrics(
        predictions,
        threshold=args.threshold,
    )
    
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "metrics_by_fold.csv"
    roc_path = output_dir / "roc_by_fold.csv"

    metrics.to_csv(metrics_path, index=False)
    roc_data.to_csv(roc_path, index=False)
    
    if args.plot:
        save_evaluation_plots(
            metrics,
            roc_data,
            output_dir,
        )
    
    by_seed, by_subject = summarize_by_subject(metrics)

    seed_path = output_dir / "metrics_by_subject_seed.csv"
    subject_path = output_dir / "metrics_by_subject.csv"

    by_seed.to_csv(seed_path, index=False)
    by_subject.to_csv(subject_path, index=False)
    
    comparison = compare_models(
    by_subject,
    model_a=args.model_a,
    model_b=args.model_b,
    )

    comparison_path = output_dir / "model_comparison.csv"

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(f"\nĐã lưu thống kê: {comparison_path}")
    print(comparison.to_string(index=False))

    print(f"\nĐã lưu điểm từng người/seed: {seed_path}")
    print(f"Đã lưu điểm từng người: {subject_path}")

    print("\nBẢNG ĐIỂM TỔNG HỢP THEO NGƯỜI:")
    print(
        by_subject[
            [
                "scheme",
                "subject",
                "model",
                "balanced_accuracy",
                "roc_auc",
                "n_seeds",
                "n_folds",
            ]
        ].to_string(index=False)
    )

    print(f"\nĐã lưu bảng điểm: {metrics_path}")
    print(f"Đã lưu dữ liệu ROC: {roc_path}")
    print(f"Số nhóm được chấm: {len(metrics)}")

    print(
        metrics[
            [
                "scheme", "subject", "fold", "model",
                "seed", "balanced_accuracy", "roc_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()