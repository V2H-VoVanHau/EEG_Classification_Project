import argparse
import hashlib
import json
from pathlib import Path

import mne
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from eegoddball.config import (
    DERIVATIVES_DIR,
    SAMPLE_SUBJECTS,
    EVENT_ID,
    EXPECTED_TRIALS,
    N_SPLITS,
    SEED,
    FOLDS_PATH,
)


def create_folds(overwrite=False):
    output_path = Path(FOLDS_PATH)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} đã tồn tại. "
            "Nếu cần thay bản folds cũ sai, chạy với --overwrite."
        )

    subjects = list(SAMPLE_SUBJECTS)

    if len(subjects) < 2 or len(set(subjects)) != len(subjects):
        raise ValueError("Cần ít nhất hai đối tượng khác nhau.")

    folds_data = {
        "within_subject": {},
        "cross_subject": {},
        "metadata": {
            "seed": SEED,
            "classes": ["standard", "target"],
            "subjects": subjects,
            "within_n_splits": N_SPLITS,
            "cross_n_splits": min(N_SPLITS, len(subjects)),
        },
    }

    for sub in subjects:
        fif_path = (
            Path(DERIVATIVES_DIR)
            / "v0"
            / f"sub-{sub}_task-oddball-epo.fif"
        )

        if not fif_path.is_file():
            raise FileNotFoundError(f"Chưa có {fif_path}")

        epochs = mne.read_epochs(
            fif_path,
            preload=False,
            verbose=False,
        )

        if epochs.event_id != EVENT_ID:
            raise ValueError(
                f"sub-{sub}: nhãn không đúng. Hãy tạo lại v0."
            )

        metadata = epochs.metadata

        required = {"trial_id", "condition", "event_code"}

        if metadata is None or not required.issubset(metadata.columns):
            raise ValueError(
                f"sub-{sub}: thiếu metadata. Hãy tạo lại v0."
            )

        # v0 phải chứa đủ kích thích gốc, đúng thứ tự
        if (
            len(epochs) != EXPECTED_TRIALS
            or not np.array_equal(
                metadata["trial_id"].to_numpy(),
                np.arange(EXPECTED_TRIALS),
            )
        ):
            raise ValueError(
                f"sub-{sub}: v0 phải có đủ trial_id "
                f"0..{EXPECTED_TRIALS - 1}, không trùng và đúng thứ tự."
            )

        expected_codes = metadata["condition"].map(EVENT_ID)

        if (
            expected_codes.isna().any()
            or not np.array_equal(
                expected_codes.to_numpy(),
                metadata["event_code"].to_numpy(),
            )
            or not np.array_equal(
                metadata["event_code"].to_numpy(),
                epochs.events[:, 2],
            )
        ):
            raise ValueError(
                f"sub-{sub}: nhãn trong metadata và events không khớp."
            )

        # Bài toán nhị phân: chỉ chia standard và target.
        # Distractor vẫn giữ trong FIF để phân tích khác.
        binary = metadata.loc[
            metadata["condition"].isin(["standard", "target"])
        ].copy()

        trial_ids = binary["trial_id"].to_numpy(dtype=int)
        labels = binary["event_code"].to_numpy(dtype=int)

        class_counts = binary["condition"].value_counts()

        if (
            set(class_counts.index) != {"standard", "target"}
            or class_counts.min() < N_SPLITS
        ):
            raise ValueError(
                f"sub-{sub}: không đủ mẫu mỗi lớp để chia "
                f"{N_SPLITS} fold."
            )

        splitter = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=SEED,
        )

        sub_folds = {}

        print(f"\nsub-{sub}: {class_counts.to_dict()}")

        # Vòng chia fold nằm BÊN TRONG vòng từng người
        for fold_idx, (_, test_indices) in enumerate(
            splitter.split(trial_ids, labels)
        ):
            test_ids = trial_ids[test_indices]
            test_labels = labels[test_indices]

            sub_folds[f"fold_{fold_idx}"] = {
                "test_trial_ids": test_ids.tolist()
            }

            n_standard = int(
                np.sum(test_labels == EVENT_ID["standard"])
            )
            n_target = int(
                np.sum(test_labels == EVENT_ID["target"])
            )

            print(
                f"  fold_{fold_idx}: "
                f"standard={n_standard}, target={n_target}, "
                f"tổng={len(test_ids)}"
            )

        folds_data["within_subject"][sub] = sub_folds

    # Pilot có 3 người: cross-subject gồm 3 fold.
    subjects_array = np.asarray(subjects)

    cross_splitter = KFold(
        n_splits=min(N_SPLITS, len(subjects)),
        shuffle=True,
        random_state=SEED,
    )

    for fold_idx, (_, test_indices) in enumerate(
        cross_splitter.split(subjects_array)
    ):
        folds_data["cross_subject"][f"fold_{fold_idx}"] = {
            "test_subjects": subjects_array[test_indices].tolist()
        }

    # Chỉ ghi sau khi tất cả đối tượng đã kiểm tra thành công
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(folds_data, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)

    folds_hash = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()

    print(f"\nĐã lưu: {output_path}")
    print(f"SHA-256: {folds_hash}")
    print(
        f"Within-subject: {N_SPLITS} fold/người; "
        f"cross-subject: {min(N_SPLITS, len(subjects))} fold."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Thay thế folds.json hiện có.",
    )
    args = parser.parse_args()

    create_folds(overwrite=args.overwrite)