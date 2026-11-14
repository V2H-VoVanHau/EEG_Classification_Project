"""Bài Lab chung tuần 1 dành cho TV5:
1. Đọc dữ liệu EEG Oddball (sub-01) từ BIDS/BrainVision.
2. Tiền xử lý nhanh: lọc 0.1 - 40 Hz, giảm tần số lấy mẫu về 250 Hz.
3. Trích xuất sự kiện: mapping nhãn chuẩn từ dataset (S 5 = standard, S 6 = target, S 7 = distractor).
4. Cắt Epochs: cửa sổ [-0.2, 0.8] s, baseline correction [-0.2, 0] s.
5. Vẽ sóng ERP tại điện cực Pz (target vs. standard), làm nổi bật thành phần P300 (~300-500 ms) sử dụng figstyle.py.
6. Chạy thử mô hình học máy LDA (Linear Discriminant Analysis) đánh giá hiệu năng phân loại qua 5-Fold Stratified CV.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne_bids import BIDSPath, read_raw_bids
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from eegoddball.config import DATA_DIR, FIGURES_DIR, SFREQ_NEW, L_FREQ, H_FREQ, TMIN, TMAX, SEED
    from eegoddball.figstyle import apply_style, save_figure, COLORS, LABELS
except ImportError:
    from src.eegoddball.config import DATA_DIR, FIGURES_DIR, SFREQ_NEW, L_FREQ, H_FREQ, TMIN, TMAX, SEED
    from src.eegoddball.figstyle import apply_style, save_figure, COLORS, LABELS

def run_lab():
    print("=" * 60)
    print("BÀI LAB CHUNG TUẦN 1: EEG ODDBALL & PHÂN LOẠI P300")
    print("=" * 60)
    
    # 1. Đọc dữ liệu thô của sub-01
    print("\n[Bước 1/6] Đọc dữ liệu BrainVision của sub-01...")
    vhdr_path = os.path.join(DATA_DIR, "sub-01", "eeg", "sub-01_task-oddball_eeg.vhdr")
    if os.path.exists(vhdr_path):
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
    else:
        bids_path = BIDSPath(subject="01", task="oddball", datatype="eeg", root=DATA_DIR)
        raw = read_raw_bids(bids_path=bids_path, verbose=False)
        raw.load_data()
    print(f"  -> Đã đọc dữ liệu thành công:")
    print(f"     - Số kênh: {len(raw.ch_names)}")
    print(f"     - Tần số lấy mẫu gốc: {raw.info['sfreq']} Hz")
    print(f"     - Độ dài bản ghi: {raw.times[-1]:.1f} giây")

    # 2. Tiền xử lý nhanh: lọc dải tần & resample
    print("\n[Bước 2/6] Tiền xử lý: Lọc dải [0.1 - 40 Hz] & resample 250 Hz...")
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, phase="zero", verbose=False)
    raw.resample(sfreq=SFREQ_NEW, verbose=False)
    print(f"  -> Tần số lấy mẫu sau resample: {raw.info['sfreq']} Hz")

    # 3. Trích xuất sự kiện
    print("\n[Bước 3/6] Trích xuất sự kiện kích thích...")
    events, event_id_dict = mne.events_from_annotations(raw, verbose=False)
    print("  -> Các annotation tìm thấy:", event_id_dict)

    # Ánh xạ theo đặc tả dataset ds004621:
    # S 5: Standard, S 6: Target, S 7: Distractor
    event_id_mapped = {}
    for ann_name, code in event_id_dict.items():
        if "S  5" in ann_name or ann_name == "Stimulus/S  5":
            event_id_mapped["standard"] = code
        elif "S  6" in ann_name or ann_name == "Stimulus/S  6":
            event_id_mapped["target"] = code
        elif "S  7" in ann_name or ann_name == "Stimulus/S  7":
            event_id_mapped["distractor"] = code
            
    print(f"  -> Ánh xạ điều kiện sử dụng:", event_id_mapped)

    # 4. Cắt Epochs
    print(f"\n[Bước 4/6] Cắt Epochs ({TMIN}s đến {TMAX}s, baseline {TMIN}s đến 0s)...")
    epochs = mne.Epochs(
        raw, events, event_id=event_id_mapped,
        tmin=TMIN, tmax=TMAX, baseline=(TMIN, 0),
        preload=True, verbose=False
    )
    print(f"  -> Tổng số epochs: {len(epochs)}")
    print(f"     - Standard: {len(epochs['standard'])}")
    print(f"     - Target: {len(epochs['target'])}")
    if "distractor" in epochs.event_id:
        print(f"     - Distractor: {len(epochs['distractor'])}")

    # 5. Vẽ sóng ERP tại điện cực Pz
    print("\n[Bước 5/6] Tính trung bình ERP và vẽ sóng tại điện cực Pz...")
    apply_style()
    
    # Tính evoked response (ERP)
    evoked_std = epochs["standard"].average()
    evoked_tgt = epochs["target"].average()
    
    times = evoked_std.times
    ch_idx = evoked_std.ch_names.index("Pz") if "Pz" in evoked_std.ch_names else 0
    pz_name = evoked_std.ch_names[ch_idx]
    
    # Đổi đơn vị sang microvolts (µV)
    std_data_uv = evoked_std.data[ch_idx] * 1e6
    tgt_data_uv = evoked_tgt.data[ch_idx] * 1e6
    diff_data_uv = tgt_data_uv - std_data_uv

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, std_data_uv, label="Standard (S5)", color=COLORS["standard"], linewidth=1.8)
    ax.plot(times, tgt_data_uv, label="Target (S6)", color=COLORS["target"], linewidth=2.2)
    ax.plot(times, diff_data_uv, label="Difference (Target - Standard)", color=COLORS["difference"], linestyle="--", linewidth=1.5)
    
    # Đánh dấu vùng P300 (300 - 500 ms)
    ax.axvspan(0.3, 0.5, color=COLORS["target"], alpha=0.12, label="P300 Window (300-500 ms)")
    ax.axvline(0, color=COLORS["baseline"], linestyle="-", linewidth=1.0, alpha=0.7)
    ax.axhline(0, color=COLORS["baseline"], linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_title(f"ERP Waveform at Electrode {pz_name} (Subject 01)", pad=10)
    ax.set_xlabel(LABELS["time"])
    ax.set_ylabel(LABELS["amplitude"])
    ax.set_xlim(TMIN, TMAX)
    ax.legend(loc="upper left")

    save_path = save_figure(fig, "lab_chung_erp_pz", folder=FIGURES_DIR, save_pdf=True, close=False)
    print(f"  -> Đã lưu hình ERP tại: {save_path}")

    # 6. Chạy thử mô hình phân loại LDA (Linear Discriminant Analysis)
    print("\n[Bước 6/6] Thử nghiệm mô hình phân loại nhị phân Target vs. Standard bằng LDA...")
    # Lấy epochs của 2 lớp: target (1) và standard (0)
    binary_epochs = epochs["standard", "target"]
    y = np.where(binary_epochs.events[:, -1] == event_id_mapped["target"], 1, 0)
    
    # Trích xuất dữ liệu: làm phẳng ma trận (kênh x thời gian) thành vector đặc trưng
    X = binary_epochs.get_data(copy=True) # (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = X.shape
    X_flat = X.reshape(n_epochs, n_channels * n_times)
    
    print(f"  -> Kích thước ma trận đặc trưng: {X_flat.shape}")
    print(f"     - Số mẫu Target: {np.sum(y == 1)}, Standard: {np.sum(y == 0)}")

    # Pipeline StandardScaler + LDA
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # Đánh giá bằng ROC-AUC và Balanced Accuracy
    auc_scores = cross_val_score(clf, X_flat, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_flat, y, cv=cv, scoring="balanced_accuracy")

    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ 5-FOLD STRATIFIED CV TRÊN SUB-01 (LDA):")
    print("=" * 60)
    print(f"  * ROC AUC:           {auc_scores.mean():.4f} ± {auc_scores.std():.4f} (Các fold: {np.round(auc_scores, 3)})")
    print(f"  * Balanced Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f} (Các fold: {np.round(acc_scores, 3)})")
    print("=" * 60)
    print("HOÀN TẤT BÀI LAB CHUNG!")

if __name__ == "__main__":
    run_lab()
