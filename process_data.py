import json
import numpy as np
import mne
from src.eegoddball.config import EPOCHS_DIR, FOLDS_PATH
import os

def load_data(sub_id, fold_name, flatten = True):
    # 1. Đường dẫn file: phiên bản dữ liệu (v0/v1) do DATA_VERSION trong config quyết định
    fif_path = os.path.join(EPOCHS_DIR, f"sub-{sub_id}_task-oddball-epo.fif")

    # 2. Đọc file folds.json
    with open(FOLDS_PATH, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # 3. Đọc dữ liệu EEG đã qua tiền xử lý
    # Bài toán chỉ phân loại target vs standard; distractor vẫn nằm trong file để vẽ ERP
    epochs = mne.read_epochs(fif_path, preload=True)[["standard", "target"]]
    X = epochs.get_data()      # Mảng dữ liệu: (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]   # Nhãn: standard=1, target=2 (n_trials,)

    # 4. Chia theo trial_id, KHÔNG theo vị trí: v1 đã loại epoch nên vị trí không còn khớp mã gốc
    trial_ids = epochs.metadata["trial_id"].to_numpy()
    test_ids = splits_data["within_subject"][sub_id][fold_name]["test_trial_ids"]
    is_test = np.isin(trial_ids, test_ids)

    # 5. Trích xuất dữ liệu Train / Test cho mô hình
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]

    print(f"Sub-{sub_id} [{fold_name}] - Tập Train: {X_train.shape}, Tập Test: {X_test.shape}")

    # 6. Trải phẳng dữ liệu
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test, test_indices
