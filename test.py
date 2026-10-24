from extract_features import *
from process_data import load_data

X_train, y_train, X_test, y_test = load_data('03', 'fold_3', flatten=False)

# sfreq = 250  # Tần số lấy mẫu 250Hz

# # 1. Trích xuất đặc trưng không phụ thuộc vào nhãn
# feat_amp_tr = extract_window_amplitude(X_train, n_windows=5)
# feat_bp_tr  = extract_bandpower(X_train, sfreq=sfreq)
# feat_ent_tr = extract_spectral_entropy(X_train, sfreq=sfreq)
# feat_dwt_tr = extract_wavelet_features(X_train, wavelet='db4', level=4)

# feat_amp_te = extract_window_amplitude(X_test, n_windows=5)
# feat_bp_te  = extract_bandpower(X_test, sfreq=sfreq)
# feat_ent_te = extract_spectral_entropy(X_test, sfreq=sfreq)
# feat_dwt_te = extract_wavelet_features(X_test, wavelet='db4', level=4)

# # 2. Trích xuất xDAWN (Fit trên Train)
# feat_xdawn_tr, feat_xdawn_te = extract_xdawn_features(X_train, y_train, X_test, n_components=2)

# # 3. Nối (Concatenate) toàn bộ đặc trưng lại thành 1 Ma trận 2D
# X_train_combined = np.hstack([feat_amp_tr, feat_bp_tr, feat_ent_tr, feat_dwt_tr, feat_xdawn_tr])
# X_test_combined  = np.hstack([feat_amp_te, feat_bp_te, feat_ent_te, feat_dwt_te, feat_xdawn_te])

# print("Kích thước ma trận Train mới:", X_train_combined.shape)
# print("Kích thước ma trận Test mới:", X_test_combined.shape)


import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from mne.preprocessing import Xdawn

# Cấu hình giao diện biểu đồ
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def _check_and_scale_uV(X):
    """Quy đổi đơn vị sang uV nếu dữ liệu ở đơn vị Volt (max < 0.01)."""
    if np.max(np.abs(X)) < 1e-2:
        return X * 1e6
    return X

def _get_target_nontarget_masks(y_train):
    """
    Tự động nhận diện nhãn Target và Non-Target trong y_train.
    Trong bài toán Oddball, số lượng thử nghiệm Target luôn ÍT HƠN Non-Target.
    """
    unique_labels, counts = np.unique(y_train, return_counts=True)
    if len(unique_labels) < 2:
        raise ValueError(f"y_train chỉ chứa 1 loại nhãn: {unique_labels}. Cần ít nhất 2 nhãn phân biệt!")
    
    target_label = unique_labels[np.argmin(counts)]      # Nhãn xuất hiện ít hơn
    nontarget_label = unique_labels[np.argmax(counts)]   # Nhãn xuất hiện nhiều hơn
    
    target_mask = (y_train == target_label)
    nontarget_mask = (y_train == nontarget_label)
    
    print(f"[Info] Nhận diện nhãn -> Target ({target_label}): {np.sum(target_mask)} trials | Non-Target ({nontarget_label}): {np.sum(nontarget_mask)} trials")
    return target_mask, nontarget_mask, target_label, nontarget_label

def plot_erp_comparison(X_train, y_train, sfreq=250, tmin=-0.2, save_dir="plots"):
    """1. Trực quan hóa sóng ERP trung bình (Target vs Non-Target)."""
    os.makedirs(save_dir, exist_ok=True)
    X_uV = _check_and_scale_uV(X_train)
    
    n_trials, n_channels, n_times = X_uV.shape
    times = (np.arange(n_times) / sfreq + tmin) * 1000  # Đổi sang ms

    target_mask, nontarget_mask, t_lbl, nt_lbl = _get_target_nontarget_masks(y_train)

    X_target = X_uV[target_mask]
    X_nontarget = X_uV[nontarget_mask]

    # Tính trung bình ERP
    mean_target = X_target.mean(axis=(0, 1))
    mean_nontarget = X_nontarget.mean(axis=(0, 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, mean_target, label=f'Target (Label {t_lbl})', color='crimson', linewidth=2)
    ax.plot(times, mean_nontarget, label=f'Non-Target (Label {nt_lbl})', color='royalblue', linewidth=1.5, linestyle='--')

    ax.axvspan(250, 450, color='gold', alpha=0.2, label='Vùng sóng P300 (~300ms)')
    ax.axvline(0, color='black', linestyle=':', alpha=0.7)
    ax.axhline(0, color='black', linestyle=':', alpha=0.7)

    ax.set_xlim(times[0], times[-1])
    ax.set_title("So sánh đáp ứng ERP Trung bình trên Tập Train", fontsize=14, fontweight='bold')
    ax.set_xlabel("Thời gian (ms)", fontsize=12)
    ax.set_ylabel("Biên độ (µV)", fontsize=12)
    ax.legend(loc='upper right')

    save_path = os.path.join(save_dir, "01_erp_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã lưu biểu đồ ERP tại: {save_path}")

def plot_psd_comparison(X_train, sfreq=250, save_dir="plots"):
    """2. Trực quan hóa Mật độ Phổ Công suất (PSD)."""
    os.makedirs(save_dir, exist_ok=True)
    X_uV = _check_and_scale_uV(X_train)

    nperseg = min(X_uV.shape[-1], int(sfreq))
    freqs, psd = welch(X_uV, fs=sfreq, axis=-1, nperseg=nperseg)

    idx = (freqs >= 1) & (freqs <= 45)
    freqs = freqs[idx]
    psd = psd[:, :, idx]

    mean_psd = psd.mean(axis=(0, 1))
    std_psd = psd.std(axis=(0, 1)) / np.sqrt(psd.shape[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(freqs, mean_psd, color='darkgreen', linewidth=2, label='PSD Trung bình')
    ax.fill_between(freqs, mean_psd - std_psd, mean_psd + std_psd, color='darkgreen', alpha=0.2)

    bands = [('Delta', 1, 4, 'lightblue'), ('Theta', 4, 8, 'lightgreen'), 
             ('Alpha', 8, 12, 'orange'), ('Beta', 12, 30, 'lightpink')]
    
    for name, fmin, fmax, col in bands:
        ax.axvspan(fmin, fmax, color=col, alpha=0.15, label=f'Dải {name} ({fmin}-{fmax}Hz)')

    ax.set_xlim(1, 45)
    ax.set_title("Mật độ Phổ Công suất (PSD) trên Tập Train", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tần số (Hz)", fontsize=12)
    ax.set_ylabel("Công suất (µV²/Hz)", fontsize=12)
    ax.legend(loc='upper right')

    save_path = os.path.join(save_dir, "02_psd_spectrum.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã lưu biểu đồ PSD tại: {save_path}")

def plot_xdawn_comparison(X_train, y_train, sfreq=250, tmin=-0.2, save_dir="plots"):
    """3. Trực quan hóa thành phần sau lọc không gian xDAWN."""
    os.makedirs(save_dir, exist_ok=True)
    n_trials, n_channels, n_times = X_train.shape
    times = (np.arange(n_times) / sfreq + tmin) * 1000

    target_mask, nontarget_mask, t_lbl, nt_lbl = _get_target_nontarget_masks(y_train)

    info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types='eeg')
    events = np.column_stack([np.arange(n_trials), np.zeros(n_trials, dtype=int), y_train.astype(int)])
    epochs = mne.EpochsArray(X_train, info, events=events, verbose=False)

    xdawn = Xdawn(n_components=2)
    xdawn.fit(epochs)
    
    X_filt = xdawn.transform(epochs)
    X_filt_uV = _check_and_scale_uV(X_filt)

    comp_0_target = X_filt_uV[target_mask, 0, :].mean(axis=0)
    comp_0_nontarget = X_filt_uV[nontarget_mask, 0, :].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, comp_0_target, label=f'Component 0 (Target - Label {t_lbl})', color='crimson', linewidth=2)
    ax.plot(times, comp_0_nontarget, label=f'Component 0 (Non-Target - Label {nt_lbl})', color='royalblue', linewidth=1.5, linestyle='--')

    ax.set_xlim(times[0], times[-1])
    ax.set_title("Tín hiệu Lọc Không Gian xDAWN (Component 0) trên Tập Train", fontsize=14, fontweight='bold')
    ax.set_xlabel("Thời gian (ms)", fontsize=12)
    ax.set_ylabel("Biên độ Lọc (µV)", fontsize=12)
    ax.legend(loc='upper right')

    save_path = os.path.join(save_dir, "03_xdawn_component.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[✓] Đã lưu biểu đồ xDAWN tại: {save_path}")

def visualize_train_data(X_train, y_train, sfreq=250, tmin=-0.2, save_dir="plots"):
    """Hàm tổng hợp."""
    print("--- Đang khởi tạo trực quan hóa tập dữ liệu Train ---")
    plot_erp_comparison(X_train, y_train, sfreq=sfreq, tmin=tmin, save_dir=save_dir)
    plot_psd_comparison(X_train, sfreq=sfreq, save_dir=save_dir)
    plot_xdawn_comparison(X_train, y_train, sfreq=sfreq, tmin=tmin, save_dir=save_dir)
    print(f"--- Hoàn tất! Tất cả biểu đồ đã lưu tại thư mục '{save_dir}/' ---")

# --- Ví dụ cách chạy trực tiếp ---
if __name__ == "__main__":
    # Giả lập dữ liệu X_train (601 trials, 127 channels, 251 timepoints)
    np.random.seed(42)
    mock_X_train = X_train  # Đơn vị Volt
    mock_y_train = y_train  # 0: Standard, 1: Oddball

    # Gọi hàm vẽ
    visualize_train_data(mock_X_train, mock_y_train, sfreq=250, tmin=-0.2)

    # print(mock_y_train[:10])