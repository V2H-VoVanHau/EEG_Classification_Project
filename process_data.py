import json
import numpy as np
import mne
from src.eegoddball.config import SPLIT_DIR, DERIVATIVES_DIR
import os

def load_data(sub_id, fold_name, flatten = True):
    # 1. Đường dẫn file
    # json_path = r"d:\Đại Học\Kỳ 1 năm Ba\Tính toán khoa học thần kinh và ứng dụng\EEG_Classification_Project\splits\fold.json"
    # fif_path = r"d:\Đại Học\Kỳ 1 năm Ba\Tính toán khoa học thần kinh và ứng dụng\EEG_Classification_Project\derivatives\v0\sub-03_task-oddball-epo.fif"

    json_path = os.path.join(SPLIT_DIR, 'fold.json')
    v0_dir = os.path.join(DERIVATIVES_DIR, 'v0')
    fif_path = os.path.join(v0_dir, f"sub-03_task-oddball-epo.fif")

    # 2. Đọc file fold.json
    with open(json_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    # 3. Đọc dữ liệu EEG đã qua tiền xử lý
    epochs = mne.read_epochs(fif_path, preload=True)
    X = epochs.get_data()      # Mảng dữ liệu: (n_trials, n_channels, n_times)
    y = epochs.events[:, -1]   # Nhãn (labels): (n_trials,)


    # Lấy danh sách test indices (xử lý cả trường hợp khóa bị gõ nhầm 'test_trail_ids' hoặc 'test_trial_ids')
    fold_info = splits_data["within_subject"][sub_id][fold_name]
    test_indices = fold_info.get("test_trail_ids") or fold_info.get("test_trial_ids")

    # 4. Suy ra danh sách train indices (các index còn lại không thuộc test)
    total_trials = len(epochs)
    all_indices = np.arange(total_trials)
    train_indices = np.setdiff1d(all_indices, test_indices)

    # 5. Trích xuất dữ liệu Train / Test cho mô hình
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    print(f"Sub-{sub_id} [{fold_name}] - Tập Train: {X_train.shape}, Tập Test: {X_test.shape}")

    # 6. Trải phẳng dữ liệu
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

    return X_train, y_train, X_test, y_test
