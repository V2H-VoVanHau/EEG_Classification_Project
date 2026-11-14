import numpy as np
import mne
from scipy.signal import welch
from scipy.integrate import simpson
import pywt
from mne.preprocessing import Xdawn

def extract_window_amplitude(X, n_windows=5):
    """
    Trích xuất đặc trưng biên độ theo từng cửa sổ thời gian.
    X: (n_trials, n_channels, n_times)
    Returns: Array (n_trials, n_channels * n_windows * 4)
    """
    n_trials, n_channels, n_times = X.shape
    win_len = n_times // n_windows
    features = []

    for w in range(n_windows):
        start = w * win_len
        end = (w + 1) * win_len if w < n_windows - 1 else n_times
        win_data = X[:, :, start:end]

        mean = np.mean(win_data, axis=-1)
        std = np.std(win_data, axis=-1)
        ptp = np.ptp(win_data, axis=-1)  # Biến thiên đỉnh-đối-đỉnh
        rms = np.sqrt(np.mean(win_data**2, axis=-1))  # Giá trị hiệu dụng

        features.extend([mean, std, ptp, rms])

    # Gộp đặc trưng: (n_trials, n_channels, n_windows * 4) -> (n_trials, -1)
    feats = np.stack(features, axis=-1)
    return feats.reshape(n_trials, -1)


def extract_bandpower(X, sfreq=250, bands=None):
    """
    Tính công suất năng lượng theo dải tần sóng EEG.
    X: (n_trials, n_channels, n_times)
    Returns: Array (n_trials, n_channels * n_bands)
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 45)
        }

    n_trials = X.shape[0]
    nperseg = min(X.shape[-1], int(sfreq))
    freqs, psd = welch(X, fs=sfreq, axis=-1, nperseg=nperseg)

    bp_list = []
    for _, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        # Tích phân Simpson tính công suất dải tần
        bp = simpson(psd[..., idx], x=freqs[idx], axis=-1)
        bp_list.append(bp)

    bp_feats = np.stack(bp_list, axis=-1)  # (n_trials, n_channels, n_bands)
    return bp_feats.reshape(n_trials, -1)

def extract_spectral_entropy(X, sfreq=250):
    """
    Tính Entropy Phổ (Spectral Entropy) cho từng kênh EEG.
    X: (n_trials, n_channels, n_times)
    Returns: Array (n_trials, n_channels)
    """
    n_trials = X.shape[0]
    nperseg = min(X.shape[-1], int(sfreq))
    freqs, psd = welch(X, fs=sfreq, axis=-1, nperseg=nperseg)

    # Chuẩn hóa PSD thành phân bố xác suất
    psd_norm = psd / np.sum(psd, axis=-1, keepdims=True) + 1e-12
    
    # Tính Shannon Entropy trên phổ năng lượng
    spec_entropy = -np.sum(psd_norm * np.log2(psd_norm), axis=-1) / np.log2(psd.shape[-1])
    return spec_entropy.reshape(n_trials, -1)



def extract_wavelet_features(X, wavelet='db4', level=4):
    """
    Phân rã DWT và trích xuất Năng lượng, Std, Mean của các hệ số.
    X: (n_trials, n_channels, n_times)
    Returns: Array (n_trials, n_channels * (level + 1) * 3)
    """
    n_trials, n_channels, _ = X.shape
    all_feats = []

    for i in range(n_trials):
        trial_feats = []
        for c in range(n_channels):
            # Phân rã wavelet
            coeffs = pywt.wavedec(X[i, c, :], wavelet=wavelet, level=level)
            ch_feats = []
            for coef in coeffs:
                energy = np.sum(coef**2)
                std = np.std(coef)
                mean = np.mean(coef)
                ch_feats.extend([energy, std, mean])
            trial_feats.extend(ch_feats)
        all_feats.append(trial_feats)

    return np.array(all_feats)


def fit_xdawn(X_train, y_train, sfreq=250, n_components=2):
    """
    Chỉ dùng để khởi tạo và huấn luyện (fit) mô hình xDAWN trên tập Train.
    """
    n_trials_tr, n_channels, _ = X_train.shape
    info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types='eeg')
    
    events_tr = np.column_stack([
        np.arange(n_trials_tr), 
        np.zeros(n_trials_tr, dtype=int), 
        y_train.astype(int)
    ])
    
    epochs_tr = mne.EpochsArray(X_train, info, events=events_tr, verbose=False)
    
    xdawn_model = Xdawn(n_components=n_components)
    xdawn_model.fit(epochs_tr)
    
    return xdawn_model

def apply_xdawn(xdawn_model, X, sfreq=250):
    """
    Dùng mô hình xDAWN đã fit để transform dữ liệu mới (Train, Val hoặc Test).
    """
    n_trials, n_channels, _ = X.shape
    info = mne.create_info(ch_names=n_channels, sfreq=sfreq, ch_types='eeg')
    
    # Với quá trình transform, nhãn (event) không có ý nghĩa nên ta dùng np.ones giả lập
    events = np.column_stack([
        np.arange(n_trials), 
        np.zeros(n_trials, dtype=int), 
        np.ones(n_trials, dtype=int)
    ])
    
    epochs = mne.EpochsArray(X, info, events=events, verbose=False)
    
    # MNE transform sẽ trả về NumPy array 3D: (n_trials, n_components_total, n_times)
    X_filt = xdawn_model.transform(epochs) 
    
    return X_filt