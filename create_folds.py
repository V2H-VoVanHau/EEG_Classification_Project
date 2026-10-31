import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from mne_bids import BIDSPath
from sklearn.model_selection import KFold, StratifiedKFold

from create_v0_data import read_stimulus_metadata
from eegoddball.config import (
    DATA_DIR,
    SUBJECTS,
    EVENT_ID,
    N_SPLITS,
    SEED,
    FOLDS_PATH,
    get_expected_trials,
)


def create_folds(overwrite=False):
    output_path = Path(FOLDS_PATH)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} đã tồn tại. "
            "Nếu cần thay bản folds cũ sai, chạy với --overwrite."
        )

    subjects = list(SUBJECTS)

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
        events_path = (
            BIDSPath(
                subject=sub,
                task="oddball",
                datatype="eeg",
                root=DATA_DIR,
            )
            .update(suffix="events", extension=".tsv")
            .fpath
        )

        # Lấy trial_id trực tiếp từ events.tsv, cùng hàm đánh số với v0 và v1:
        # chia trên toàn bộ kích thích gốc, không phụ thuộc epoch nào bị loại
        metadata = read_stimulus_metadata(
            events_path,
            get_expected_trials(sub),
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
    # Luôn ghi LF: folds_hash băm từng byte nên phải giống nhau trên mọi máy
    temporary_path.write_text(
        json.dumps(folds_data, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
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
