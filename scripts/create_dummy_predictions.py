"""Tạo dự đoán ngẫu nhiên để chạy thử máy chấm.

Dùng: python scripts/create_dummy_predictions.py
"""
import json
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from eegoddball.config import (
    DATA_VERSION,
    EPOCHS_DIR,
    FOLDS_PATH,
    RESULTS_DIR,
    SEED,
    get_folds_hash,
)


def main():
    folds = json.loads(
        Path(FOLDS_PATH).read_text(encoding="utf-8")
    )
    folds_hash = get_folds_hash()

    # Lập danh sách mẫu test từ folds và metadata thực tế
    test_rows = []

    for sub in folds["metadata"]["subjects"]:
        path = Path(EPOCHS_DIR) / f"sub-{sub}_task-oddball-epo.fif"

        epochs = mne.read_epochs(
            path, preload=False, verbose=False
        )
        metadata = epochs.metadata

        if metadata is None:
            raise ValueError(f"sub-{sub}: thiếu metadata.")

        binary = metadata.loc[
            metadata["condition"].isin(["standard", "target"])
        ].copy()

        def append_rows(scheme, fold_name, selected):
            for trial in selected.itertuples(index=False):
                test_rows.append({
                    "scheme": scheme,
                    "subject": sub,
                    "fold": fold_name,
                    "trial_id": int(trial.trial_id),
                    "y_true": int(trial.condition == "target"),
                })

        for fold_name, fold in (
            folds["within_subject"][sub].items()
        ):
            selected = binary.loc[
                binary["trial_id"].isin(
                    fold["test_trial_ids"]
                )
            ]
            append_rows(
                "within_subject", fold_name, selected
            )

        for fold_name, fold in folds["cross_subject"].items():
            if sub in fold["test_subjects"]:
                append_rows(
                    "cross_subject", fold_name, binary
                )

    base = pd.DataFrame(test_rows)

    if base.empty:
        raise ValueError("Không tìm thấy mẫu test.")

    output_dir = Path(RESULTS_DIR) / "dummy"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    for filename, model in [
        ("predictions_tv3.csv", "DUMMY_LDA"),
        ("predictions_tv4.csv", "DUMMY_EEGNet"),
    ]:
        predictions = base.copy()
        predictions["model"] = model
        predictions["seed"] = SEED

        # Xác suất ngẫu nhiên, không phải kết quả mô hình thật
        predictions["y_score"] = rng.uniform(
            0.0, 1.0, size=len(predictions)
        )

        predictions["data_version"] = DATA_VERSION
        predictions["folds_hash"] = folds_hash

        columns = [
            "scheme", "subject", "fold", "model", "seed",
            "trial_id", "y_true", "y_score",
            "data_version", "folds_hash",
        ]

        output_path = output_dir / filename
        predictions[columns].to_csv(
            output_path, index=False
        )

        print(f"Đã tạo: {output_path}")
        print(f"Số dòng: {len(predictions)}")


if __name__ == "__main__":
    main()