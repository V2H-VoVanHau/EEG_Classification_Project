"""Giải thích mô hình (Explainability / RQ3) và Chuẩn hóa Hình 1, Hình 9, Hình 10 & Occlusion (TV5).

Bao gồm:
1. Hình 1: Sơ đồ tổng quan Pipeline 6 bước & 2 nhánh mô hình (ML vs. DL).
2. Giải mã theo thời gian (Time-resolved decoding) bằng `mne.decoding.SlidingEstimator` -> Hình 9.
3. Activation Pattern theo Haufe et al. (2014) bằng `mne.decoding.LinearModel` & `get_coef(..., 'patterns_', inverse_transform=True)`.
4. Tương thích trực tiếp với gói bàn giao `EEGNet` của TV4 (`derivatives/models/<stem>.pt`, `_norm.npz`, `_test.npz`, `_meta.json`).
5. Integrated Gradients (Captum), đóng gói thành `mne.EvokedArray` để vẽ Topomap đối chiếu ERP (cùng `vlim`), và Sanity Check (Adebayo et al., 2018) trên mạng trọng số ngẫu nhiên vs. mạng đã huấn luyện -> Hình 10.
6. Che khuất không-thời gian (Spatiotemporal Occlusion) trên EEGNet để đo mức sụt giảm ΔAUC khi che các cửa sổ thời gian và cụm kênh -> `figures/fig_eegnet_occlusion.png`/`.pdf` & `results/occlusion_eegnet_<ver>.csv`.

Dùng:
    python scripts/05_explain.py --data-version v0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from mne.decoding import LinearModel, SlidingEstimator, cross_val_multiscore, get_coef
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eegoddball import figstyle
from eegoddball.config import (
    DERIVATIVES_DIR,
    EVENT_ID,
    FIGURES_DIR,
    RESULTS_DIR,
    SAMPLE_SUBJECTS,
    SEED,
    TMAX,
    TMIN,
)


# ==============================================================================
# 1. KIẾN TRÚC EEGNET (Lawhern et al., 2018) KHỚP 100% HỢP ĐỒNG BÀN GIAO CỦA TV4
# ==============================================================================
class EEGNet(nn.Module):
    """EEGNet-8,2 tương thích trực tiếp với `state_dict` trong `derivatives/models/*.pt` của TV4.

    Đầu vào: `(batch, n_ch, n_times)` hoặc `(batch, 1, n_ch, n_times)`.
    Đầu ra: logits 2 lớp `(batch, 2)` (cột 1 = target) khi `n_classes=2`,
            hoặc `(batch,)` khi `n_classes=1`.
    """

    def __init__(
        self,
        n_ch: int = None,
        n_times: int = 103,
        n_classes: int = 2,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        k1: int = 64,
        drop: float = 0.5,
        n_channels: int = None,
    ):
        super().__init__()
        if n_ch is None:
            n_ch = n_channels if n_channels is not None else 60
        self.n_ch = n_ch
        self.n_times = n_times
        self.n_classes = n_classes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        with torch.no_grad():
            n_feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()
        self.classify = nn.Linear(n_feat, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.block2(self.block1(x))
        logits = self.classify(out.flatten(1))
        if self.n_classes == 1:
            return logits.squeeze(-1)
        return logits


def load_tv4_bundle(stem: str, models_dir: Path = None, device: str = "cpu"):
    """Nạp gói bàn giao của TV4 (`{stem}.pt`, `_norm.npz`, `_test.npz`, `_meta.json`)."""
    if models_dir is None:
        models_dir = Path(DERIVATIVES_DIR) / "models"
    else:
        models_dir = Path(models_dir)

    meta_path = models_dir / f"{stem}_meta.json"
    pt_path = models_dir / f"{stem}.pt"
    norm_path = models_dir / f"{stem}_norm.npz"
    test_path = models_dir / f"{stem}_test.npz"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    arch = meta.get("arch", dict(F1=8, D=2, F2=16, k1=64, drop=0.5))
    model = EEGNet(meta["n_ch"], meta["n_times"], n_classes=2, **arch)
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.to(device).eval()

    nrm = dict(np.load(norm_path)) if norm_path.exists() else None
    test = dict(np.load(test_path)) if test_path.exists() else None
    return model, nrm, test, meta


def predict_target_proba(model: nn.Module, X_tensor: torch.Tensor) -> np.ndarray:
    """Tính xác suất lớp target (lớp 1) từ đầu ra của EEGNet."""
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        if logits.dim() == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = torch.softmax(logits, dim=1)[:, 1]
    return probs.detach().cpu().numpy()


# ==============================================================================
# 2. VẼ HÌNH 1: SƠ ĐỒ PIPELINE TỔNG QUAN DỰ ÁN (6 BƯỚC & 2 NHÁNH MÔ HÌNH)
# ==============================================================================
def plot_pipeline_diagram():
    """Vẽ Hình 1: Sơ đồ tổng quan 6 bước và 2 nhánh ML vs DL của dự án."""
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(15.0, 5.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    def draw_box(x, y, w, h, title, subtitle, facecolor="#f8f9fa", edgecolor="#333333", title_color="#111111"):
        box = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=1.2",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2, y + h * 0.74, title,
            ha="center", va="center", fontsize=9.2, fontweight="bold", color=title_color,
        )
        ax.text(
            x + w / 2, y + h * 0.34, subtitle,
            ha="center", va="center", fontsize=8.0, color="#222222", linespacing=1.35,
        )

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.55,head_width=0.32",
                lw=1.5, color="#444444", shrinkA=0, shrinkB=0,
            ),
        )

    # Cột 1: Bước 1 (Dữ liệu gốc & Chia tập)
    draw_box(
        1.5, 11.5, 20.0, 15.0,
        "1. Raw BIDS & Splits (TV1)",
        "OpenNeuro ds004621 (N = 42)\nS5 (Std), S6 (Tgt), S7 (Dist)\nsplits/folds.json (trial_id & sub)",
        facecolor="#eef4fa", edgecolor="#1f77b4", title_color="#1f77b4",
    )

    # Cột 2: Bước 2 (Tiền xử lý & Cắt Epoch)
    draw_box(
        26.5, 11.5, 20.5, 15.0,
        "2. Preprocessing & ERP (TV2)",
        "0.1–40 Hz, Spline/Bridge, 1 Hz ICA\nAvg Ref, 250 Hz, ±150 µV, Correct\nderivatives/v1 & Figures 2–5",
        facecolor="#eefaf2", edgecolor="#2ca02c", title_color="#1e7e34",
    )

    # Cột 3 - Nhánh trên: Bước 3A (Học máy cổ điển - TV3)
    draw_box(
        52.5, 21.5, 20.5, 13.5,
        "3A. Classical ML (TV3)",
        "Time/Freq/Wavelet/xDAWN Features\nLDA, LogReg, SVM, RF, xDAWN\nFigure 6 & predictions_tv3.csv",
        facecolor="#fff8ec", edgecolor="#ff7f0e", title_color="#b85600",
    )

    # Cột 3 - Nhánh dưới: Bước 3B (Học sâu - TV4)
    draw_box(
        52.5, 3.0, 20.5, 13.5,
        "3B. Deep Learning (TV4)",
        "End-to-End EEGNet (128 Hz, 0–0.8 s)\nClass Weights & Early Stopping\n.pt Bundles & predictions_tv4.csv",
        facecolor="#fdf0f0", edgecolor="#d62728", title_color="#a71d2a",
    )

    # Cột 4 - Nhánh trên: Bước 4 (Đánh giá thống kê RQ1 & RQ2 - TV1)
    draw_box(
        78.5, 21.5, 20.5, 13.5,
        "4. Unified Evaluation (TV1)",
        "RQ1 (Within) & RQ2 (Cross-subject)\nROC-AUC, Bal. Acc, Wilcoxon, CI95%\nFigures 7–8 & results/summary.csv",
        facecolor="#eef4fa", edgecolor="#1f77b4", title_color="#1f77b4",
    )

    # Cột 4 - Nhánh dưới: Bước 5–6 (Giải thích XAI & Confound RQ3 - TV5)
    draw_box(
        78.5, 3.0, 20.5, 13.5,
        "5–6. Explainability & RQ3 (TV5)",
        "Sliding AUC, Haufe, IG & Occlusion\nMotor / Visual / Rarity Confounds\nFigures 9–10 & predictions_tv5.csv",
        facecolor="#f5eefb", edgecolor="#9467bd", title_color="#6f42c1",
    )

    # Mũi tên luồng dữ liệu (bắt đầu và kết thúc cách mép hộp 0.4 đơn vị)
    draw_arrow(21.9, 19.0, 26.1, 19.0)
    draw_arrow(47.4, 21.5, 52.1, 28.0)
    draw_arrow(47.4, 16.5, 52.1, 10.0)
    draw_arrow(73.4, 29.0, 78.1, 29.0)
    draw_arrow(73.4, 12.5, 78.1, 25.5)
    draw_arrow(73.4, 25.5, 78.1, 12.5)
    draw_arrow(73.4, 9.0, 78.1, 9.0)

    ax.text(
        50.0, 38.0,
        "Figure 1: End-to-End Pipeline for Single-Trial EEG Oddball Classification & Neurophysiological Explainability",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color="#111111",
    )

    return figstyle.save(fig, "fig1_pipeline", folder=FIGURES_DIR, save_pdf=True, close=True)


# ==============================================================================
# 3. ĐỌC DỮ LIỆU EPOCHS
# ==============================================================================
def get_available_subjects(subjects, data_version="v0"):
    """Lọc danh sách đối tượng có sẵn file epoch trong derivatives/<data_version>."""
    ver_dir = Path(DERIVATIVES_DIR) / data_version
    loaded_subs = []
    for sub in subjects:
        fif_path = ver_dir / f"sub-{sub}_task-oddball-epo.fif"
        if not fif_path.exists():
            print(f"[Cảnh báo] Chưa tìm thấy {fif_path}, bỏ qua sub-{sub}.")
            continue
        loaded_subs.append(sub)
    if not loaded_subs:
        raise FileNotFoundError(f"Không tìm thấy bất kỳ file epoch nào trong {ver_dir}")
    return loaded_subs


def read_one_subject_epochs(sub, data_version="v0"):
    """Đọc Epochs của 1 đối tượng để tiết kiệm RAM khi chạy 42 người."""
    fif_path = Path(DERIVATIVES_DIR) / data_version / f"sub-{sub}_task-oddball-epo.fif"
    return mne.read_epochs(fif_path, preload=True, verbose=False)


def load_subject_epochs(subjects, data_version="v0"):
    """Trả về danh sách đối tượng hợp lệ (giữ tương thích API)."""
    loaded_subs = get_available_subjects(subjects, data_version=data_version)
    return loaded_subs, loaded_subs


def _prepare_valid_topomap_evoked(data: np.ndarray, info: mne.Info, tmin: float, comment: str = "") -> mne.EvokedArray:
    """Đóng gói mảng (n_channels, n_times) thành mne.EvokedArray, ưu tiên montage sẵn có của v1 (giữ O9/O10)."""
    evoked = mne.EvokedArray(data.copy(), info.copy(), tmin=tmin, comment=comment, verbose=False)

    def _valid_chs(ev):
        return [
            ch["ch_name"]
            for ch in ev.info["chs"]
            if ch["loc"][:3] is not None and np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3] != 0)
        ]

    valid_chs = _valid_chs(evoked)
    if len(valid_chs) < max(3, len(evoked.ch_names) // 2):
        # Fallback cho v0 (v0 chưa gắn sẵn tọa độ điện cực)
        montage = mne.channels.make_standard_montage("standard_1005")
        evoked.set_montage(montage, on_missing="ignore", verbose=False)
        valid_chs = _valid_chs(evoked)

    if valid_chs:
        evoked.pick(valid_chs)
    return evoked


# ==============================================================================
# 4. TIME-RESOLVED DECODING (TUẦN 1 & HÌNH 9)
# ==============================================================================
def run_time_resolved_decoding(epochs_list, loaded_subs, data_version="v0"):
    """Chạy SlidingEstimator (LogReg + StandardScaler) ở từng thời điểm và vẽ Hình 9."""
    print(f"\n[1/4] Đang chạy Time-Resolved Decoding trên {len(loaded_subs)} đối tượng ({loaded_subs[0]}..{loaded_subs[-1]})...")
    all_subject_aucs = []
    evoked_std_list = []
    evoked_tgt_list = []

    first_ep = read_one_subject_epochs(loaded_subs[0], data_version=data_version)
    times = first_ep.times
    ch_names = first_ep.ch_names
    del first_ep

    decim_step = 2 if len(loaded_subs) > 10 else 1

    for sub in loaded_subs:
        epochs = (
            read_one_subject_epochs(sub, data_version=data_version)
            if isinstance(epochs_list[0], str)
            else epochs_list[loaded_subs.index(sub)]
        )
        binary_ep = epochs["standard", "target"]
        X_full = binary_ep.get_data(copy=True)
        X = X_full[:, :, ::decim_step]
        t_decim = times[::decim_step]
        y = (binary_ep.events[:, 2] == EVENT_ID["target"]).astype(int)

        base_clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=300, class_weight="balanced", random_state=SEED),
        )
        time_decod = SlidingEstimator(base_clf, n_jobs=1, scoring="roc_auc", verbose=False)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = cross_val_multiscore(time_decod, X, y, cv=cv, n_jobs=1)
        mean_auc_decim = scores.mean(axis=0)
        mean_auc = np.interp(times, t_decim, mean_auc_decim) if decim_step > 1 else mean_auc_decim
        all_subject_aucs.append(mean_auc)

        evoked_std_list.append(epochs["standard"].average())
        evoked_tgt_list.append(epochs["target"].average())

        peak_idx = int(np.argmax(mean_auc))
        print(f"  * sub-{sub}: Peak AUC = {mean_auc[peak_idx]:.4f} tại t = {times[peak_idx]*1000:.0f} ms")
        del epochs, binary_ep, X_full, X

    aucs_arr = np.array(all_subject_aucs)  # (n_subs, n_times)
    grand_auc_mean = aucs_arr.mean(axis=0)
    grand_auc_std = aucs_arr.std(axis=0) if len(loaded_subs) > 1 else np.zeros_like(grand_auc_mean)

    # Tính sóng ERP trung bình tại kênh Pz
    pz_idx = ch_names.index("Pz") if "Pz" in ch_names else 0
    pz_name = ch_names[pz_idx]

    std_waves = np.array([ev.data[pz_idx] * 1e6 for ev in evoked_std_list])
    tgt_waves = np.array([ev.data[pz_idx] * 1e6 for ev in evoked_tgt_list])
    std_mean = std_waves.mean(axis=0)
    tgt_mean = tgt_waves.mean(axis=0)
    diff_mean = tgt_mean - std_mean

    # Lưu kết quả số liệu ra CSV
    out_df = pd.DataFrame({
        "time_s": times,
        "auc_mean": grand_auc_mean,
        "auc_std": grand_auc_std,
        "erp_standard_uV": std_mean,
        "erp_target_uV": tgt_mean,
        "erp_diff_uV": diff_mean,
    })
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = Path(RESULTS_DIR) / f"time_resolved_auc_{data_version}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  -> Đã lưu bảng AUC theo thời gian tại: {csv_path}")

    # Vẽ Hình 9: Đường cong AUC theo thời gian chồng lên sóng ERP tại Pz
    figstyle.apply()
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Trục 1: Sóng ERP tại kênh Pz (µV)
    l1 = ax1.plot(times, std_mean, color=figstyle.COLORS["standard"], lw=1.6, label=f"ERP Standard ({pz_name})")
    l2 = ax1.plot(times, tgt_mean, color=figstyle.COLORS["target"], lw=2.0, label=f"ERP Target ({pz_name})")
    l3 = ax1.plot(times, diff_mean, color=figstyle.COLORS["difference"], ls="--", lw=1.6, label="ERP Difference (Tgt - Std)")
    ax1.set_xlabel(figstyle.LABELS["time"])
    ax1.set_ylabel(f"ERP {figstyle.LABELS['amplitude']} at {pz_name}", color="#333333")
    ax1.axvline(0, color=figstyle.COLORS["baseline"], ls="-", lw=1.0, alpha=0.7)
    ax1.axvspan(0.3, 0.6, color=figstyle.COLORS["target"], alpha=0.10, label="P3b Window (300–600 ms)")
    ax1.set_xlim(TMIN, TMAX)

    # Trục 2: Đường cong Time-Resolved ROC AUC
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    l4 = ax2.plot(times, grand_auc_mean, color=figstyle.COLORS["auc"], lw=2.5, label="Time-Resolved Decoding AUC")
    if len(loaded_subs) > 1:
        ax2.fill_between(
            times,
            grand_auc_mean - grand_auc_std,
            grand_auc_mean + grand_auc_std,
            color=figstyle.COLORS["auc"],
            alpha=0.18,
        )
    ax2.axhline(0.5, color="#666666", ls=":", lw=1.0)
    ax2.set_ylabel(figstyle.LABELS["auc"], color=figstyle.COLORS["auc"])
    ax2.tick_params(axis="y", labelcolor=figstyle.COLORS["auc"])
    ax2.set_ylim(0.42, 1.0)

    # Gộp legend của 2 trục
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    peak_t = times[int(np.argmax(grand_auc_mean))]
    peak_v = float(np.max(grand_auc_mean))
    ax1.set_title(
        f"Figure 9: Time-Resolved Decoding AUC vs. ERP at {pz_name} "
        f"(N={len(loaded_subs)}, Peak AUC={peak_v:.3f} at {peak_t*1000:.0f} ms)"
    )

    fig_path = figstyle.save(fig, "fig9_time_resolved_decoding", folder=FIGURES_DIR, save_pdf=True, close=True)
    return out_df, fig_path


# ==============================================================================
# 5. HAUFE ACTIVATION PATTERN (Haufe et al., 2014) (TUẦN 2)
# ==============================================================================
def run_haufe_activation_pattern(epochs_list, loaded_subs, data_version="v0"):
    """Tính Activation Pattern (Haufe et al., 2014) từ mô hình tuyến tính và vẽ chuỗi Topomap."""
    print(f"\n[2/4] Đang tính Haufe Activation Pattern (Haufe et al., 2014) trên {len(loaded_subs)} đối tượng...")
    patterns_list = []
    filters_list = []
    ref_info = None
    ref_tmin = None

    for sub in loaded_subs:
        epochs = (
            read_one_subject_epochs(sub, data_version=data_version)
            if isinstance(epochs_list[0], str)
            else epochs_list[loaded_subs.index(sub)]
        )
        if ref_info is None:
            ref_info = epochs.info.copy()
            ref_tmin = epochs.tmin
        binary_ep = epochs["standard", "target"]
        X = binary_ep.get_data(copy=True)
        y = (binary_ep.events[:, 2] == EVENT_ID["target"]).astype(int)

        clf = make_pipeline(
            StandardScaler(),
            LinearModel(LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=SEED)),
        )
        time_decod = SlidingEstimator(clf, n_jobs=1, scoring="roc_auc", verbose=False)
        time_decod.fit(X, y)

        # Lấy activation pattern (nhân hiệp phương sai dữ liệu và inverse_transform qua StandardScaler)
        pattern = get_coef(time_decod, "patterns_", inverse_transform=True)
        filt = get_coef(time_decod, "filters_", inverse_transform=False)
        patterns_list.append(pattern)
        filters_list.append(filt)
        del epochs, binary_ep, X

    mean_pattern = np.mean(patterns_list, axis=0)  # (n_channels, n_times)
    mean_filter = np.mean(filters_list, axis=0)

    info = ref_info
    tmin = ref_tmin
    evoked_pattern = _prepare_valid_topomap_evoked(mean_pattern, info, tmin, comment="Haufe Activation Pattern")
    evoked_filter = _prepare_valid_topomap_evoked(mean_filter, info, tmin, comment="Raw Classifier Weights")

    figstyle.apply()
    topo_times = [0.10, 0.20, 0.35, 0.45, 0.60]
    fig, axes = plt.subplots(2, len(topo_times), figsize=(12, 5.2))

    vlim_p = float(np.max(np.abs(evoked_pattern.data)))
    vlim_f = float(np.max(np.abs(evoked_filter.data)))

    for col_idx, t_val in enumerate(topo_times):
        t_idx = evoked_pattern.time_as_index(t_val)[0]
        pat_vec = evoked_pattern.data[:, t_idx]
        flt_vec = evoked_filter.data[:, t_idx]

        mne.viz.plot_topomap(
            pat_vec, evoked_pattern.info, axes=axes[0, col_idx], show=False,
            vlim=(-vlim_p, vlim_p), cmap="RdBu_r", contours=4,
        )
        axes[0, col_idx].set_title(f"{int(t_val*1000)} ms", fontsize=10, fontweight="bold")

        mne.viz.plot_topomap(
            flt_vec, evoked_filter.info, axes=axes[1, col_idx], show=False,
            vlim=(-vlim_f, vlim_f), cmap="RdBu_r", contours=4,
        )

    axes[0, 0].set_ylabel("Haufe Pattern\n(Physiological)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Raw Weights (w)\n(Noisy / Filter)", fontsize=10, fontweight="bold")
    fig.suptitle(
        "Haufe et al. (2014) Activation Pattern vs. Raw Linear Classifier Weights",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig_path = figstyle.save(fig, "fig_haufe_activation_pattern", folder=FIGURES_DIR, save_pdf=True, close=True)
    return fig_path


# ==============================================================================
# 6. INTEGRATED GRADIENTS & SANITY CHECK (TUẦN 2 & HÌNH 10)
# ==============================================================================
def compute_norm_zero_baseline(mean_ch: np.ndarray, std_ch: np.ndarray, baseline_mode: str = "zero_uv") -> np.ndarray:
    """Tính giá trị baseline 1D (n_ch,) trong không gian đã chuẩn hóa z-score.

    - 'zero_uv': tương ứng 0 µV vật lý = (0 - mean) / std (theo tài liệu bàn giao TV4).
    - 'train_mean': tương ứng 0.0 trong không gian chuẩn hóa (= trung bình kênh tập train).
    """
    mean_1d = np.asarray(mean_ch, dtype=np.float32).reshape(-1)
    std_1d = np.asarray(std_ch, dtype=np.float32).reshape(-1)
    if baseline_mode == "zero_uv":
        return ((0.0 - mean_1d) / np.maximum(std_1d, 1e-6)).astype(np.float32)
    return np.zeros_like(mean_1d, dtype=np.float32)


def compute_ig_map(
    model: nn.Module,
    X_target_tensor: torch.Tensor,
    max_samples: int = 256,
    baseline_1d: np.ndarray | None = None,
) -> np.ndarray:
    """Tính bản đồ Integrated Gradients trung bình trên các epoch target theo từng batch (tiết kiệm RAM)."""
    model.eval()
    ig = IntegratedGradients(model)
    if len(X_target_tensor) > max_samples:
        X_target_tensor = X_target_tensor[:max_samples]
    with torch.no_grad():
        sample_out = model(X_target_tensor[:1])
    target_cls = 1 if (sample_out.dim() == 2 and sample_out.shape[1] >= 2) else None

    if baseline_1d is not None:
        b_proto = torch.tensor(baseline_1d, dtype=X_target_tensor.dtype, device=X_target_tensor.device).view(1, -1, 1)
    else:
        b_proto = None

    batch_size = 32
    attr_sum = None
    n_total = 0
    for start in range(0, len(X_target_tensor), batch_size):
        batch = X_target_tensor[start : start + batch_size]
        baseline = b_proto.expand_as(batch) if b_proto is not None else torch.zeros_like(batch)
        attr = ig.attribute(batch, baselines=baseline, target=target_cls, n_steps=32, internal_batch_size=32)
        batch_np = attr.detach().cpu().numpy().sum(axis=0)
        attr_sum = batch_np if attr_sum is None else (attr_sum + batch_np)
        n_total += len(batch)
    return attr_sum / max(n_total, 1)


def _train_or_load_eegnet_for_explain(
    X_norm: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    models_dir: Path = None,
):
    """Nếu có sẵn model bundle của TV4 có cùng (n_ch, n_times) thì ưu tiên nạp; nếu không thì huấn luyện tại chỗ."""
    n_channels, n_times = X_norm.shape[1], X_norm.shape[2]
    if models_dir is None:
        models_dir = Path(DERIVATIVES_DIR) / "models"

    if models_dir.exists():
        for meta_file in sorted(models_dir.glob("*_meta.json")):
            stem = meta_file.name[: -len("_meta.json")]
            try:
                model, _, _, meta = load_tv4_bundle(stem, models_dir=models_dir)
                if meta.get("n_ch") == n_channels and meta.get("n_times") == n_times:
                    print(f"  -> Đã nạp trực tiếp mô hình EEGNet từ gói bàn giao TV4: {stem}")
                    return model, f"TV4 bundle ({stem})"
            except Exception as exc:
                print(f"  [Bỏ qua bundle {stem}]: {exc}")

    # Huấn luyện EEGNet chuẩn 2 lớp (CrossEntropyLoss có trọng số lớp y hệt TV4)
    torch.manual_seed(SEED)
    trained_model = EEGNet(n_ch=n_channels, n_times=n_times, n_classes=2, drop=0.25)
    X_train_t = torch.tensor(X_norm[train_idx], dtype=torch.float32)
    y_train_t = torch.tensor(y[train_idx], dtype=torch.long)

    counts = np.bincount(y[train_idx].astype(int), minlength=2)
    w = torch.tensor(len(y_train_t) / (2.0 * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(trained_model.parameters(), lr=1e-3, weight_decay=1e-4)

    trained_model.train()
    batch_size = 64
    for _ in range(18):
        idx_perm = torch.randperm(len(X_train_t))
        for start in range(0, len(X_train_t), batch_size):
            b_idx = idx_perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = trained_model(X_train_t[b_idx])
            loss = criterion(logits, y_train_t[b_idx])
            loss.backward()
            optimizer.step()

    trained_model.eval()
    return trained_model, "In-script Trained EEGNet"


# ==============================================================================
# 7. SPATIOTEMPORAL OCCLUSION TRÊN EEGNET (TUẦN 2 - KIỂM TRA NHÂN QUẢ)
# ==============================================================================
def save_occlusion_outputs(df_occ: pd.DataFrame, data_version: str = "v0") -> Path:
    """Lưu CSV và vẽ biểu đồ Occlusion."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = Path(RESULTS_DIR) / f"occlusion_eegnet_{data_version}.csv"
    df_occ.to_csv(csv_path, index=False)
    print(f"  -> Đã lưu kết quả Occlusion tại: {csv_path}")

    base_auc = float(df_occ["baseline_auc"].iloc[0])
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    y_pos = np.arange(len(df_occ))
    colors = [
        figstyle.COLORS["target"] if "P3b" in cond or "350" in cond else figstyle.COLORS["distractor"]
        for cond in df_occ["condition"]
    ]
    bars = ax.barh(y_pos, df_occ["delta_auc"], color=colors, alpha=0.88, edgecolor="#222222", height=0.62)
    ax.axvline(0.0, color="#333333", lw=1.1, ls="-")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_occ["condition"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("AUC Drop when Occluded (ΔAUC = Baseline AUC − Occluded AUC)")
    ax.set_title(
        f"Spatiotemporal Occlusion Analysis on EEGNet (Baseline Test AUC = {base_auc:.3f})",
        fontsize=11.5, fontweight="bold",
    )
    for bar, val in zip(bars, df_occ["delta_auc"]):
        ax.text(
            val + (0.004 if val >= 0 else -0.004),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8.5,
            fontweight="bold",
        )
    figstyle.save(fig, "fig_eegnet_occlusion", folder=FIGURES_DIR, save_pdf=True, close=True)
    return csv_path


def run_eegnet_occlusion(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    times: np.ndarray,
    ch_names: list[str],
    data_version: str = "v0",
    save_outputs: bool = True,
    fill_1d: np.ndarray | None = None,
) -> pd.DataFrame:
    """Che khuất có hệ thống (Spatiotemporal Occlusion) từng cửa sổ thời gian & cụm kênh trên EEGNet.

    Đo mức sụt giảm ΔAUC = AUC_baseline - AUC_occluded:
    - `fill_1d` cho phép thay thế đúng giá trị `0 µV` trong không gian chuẩn hóa ((0 - mean) / std)
      hoặc `0.0` (= trung bình kênh tập train) nhất quán với baseline của IG.
    """
    print(f"\n[4/4] Đang chạy Spatiotemporal Occlusion trên EEGNet...")
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    base_probs = predict_target_proba(model, X_test_t)
    base_auc = float(roc_auc_score(y_test, base_probs))

    parietal_chs = [c for c in ["Pz", "CPz", "P1", "P2", "P3", "P4", "POz", "CP1", "CP2"] if c in ch_names]
    motor_c3_chs = [c for c in ["C3", "C1", "C5", "FC3", "CP3", "Cz"] if c in ch_names]
    occipital_chs = [c for c in ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"] if c in ch_names]

    occlusion_specs = [
        ("Time: Pre-stimulus (-200–0 ms)", (-0.20, 0.00), None),
        ("Time: Early Visual (80–200 ms)", (0.08, 0.20), None),
        ("Time: P3b Core Window (300–600 ms)", (0.30, 0.60), None),
        ("Time: Post-350ms Window (350–800 ms)", (0.35, 0.80), None),
        ("Time: Late Window (600–800 ms)", (0.60, 0.80), None),
        ("Space: Parietal P3b Cluster (Pz/CPz/P3/P4)", None, parietal_chs),
        ("Space: Motor Button-Press Cluster (C3/FC3/CP3)", None, motor_c3_chs),
        ("Space: Occipital Visual Cluster (O1/O2/Oz)", None, occipital_chs),
        ("Joint: Parietal P3b (300–600 ms)", (0.30, 0.60), parietal_chs),
    ]

    fill_arr = (
        np.asarray(fill_1d, dtype=np.float32).reshape(1, -1, 1)
        if fill_1d is not None
        else np.zeros((1, len(ch_names), 1), dtype=np.float32)
    )

    rows = []
    for label, t_win, ch_subset in occlusion_specs:
        X_occ = X_test.copy()
        t_mask = np.ones(len(times), dtype=bool)
        if t_win is not None:
            t_mask = (times >= t_win[0]) & (times <= t_win[1])
            if not t_mask.any():
                continue

        if ch_subset is not None and len(ch_subset) > 0:
            ch_idx = [ch_names.index(c) for c in ch_subset]
            t_idx = np.flatnonzero(t_mask)
            X_occ[:, np.ix_(ch_idx, t_idx)[0], np.ix_(ch_idx, t_idx)[1]] = fill_arr[:, ch_idx, :]
        elif ch_subset is None:
            t_idx = np.flatnonzero(t_mask)
            X_occ[:, :, t_idx] = fill_arr

        occ_probs = predict_target_proba(model, torch.tensor(X_occ, dtype=torch.float32))
        occ_auc = float(roc_auc_score(y_test, occ_probs))
        delta_auc = base_auc - occ_auc
        rows.append({
            "condition": label,
            "baseline_auc": base_auc,
            "occluded_auc": occ_auc,
            "delta_auc": delta_auc,
        })
        print(f"  * {label:<45} | AUC = {occ_auc:.4f} (ΔAUC = {-delta_auc:+.4f})")

    df_occ = pd.DataFrame(rows)

    if save_outputs:
        save_occlusion_outputs(df_occ, data_version=data_version)

    return df_occ


def _extract_dl_grid(epochs: mne.Epochs, sfreq_dl: float = 128.0, tmin_dl: float = 0.0, tmax_dl: float = 0.8):
    """Chuẩn bị epoch đúng lưới thời gian học sâu của TV4 (128 Hz, 0.0–0.8 s -> 103 mẫu, µV)."""
    ep = epochs["standard", "target"].copy()
    if ep.baseline is None:
        ep.apply_baseline((-0.2, 0.0), verbose=False)
    if not np.isclose(ep.info["sfreq"], sfreq_dl):
        ep.resample(sfreq_dl, verbose=False)
    ep.crop(tmin_dl, tmax_dl)
    X_sub = (ep.get_data(picks="eeg") * 1e6).astype(np.float32)
    y_sub = (ep.events[:, 2] == EVENT_ID["target"]).astype(np.float32)
    tids = ep.metadata["trial_id"].to_numpy(dtype=np.int64) if ep.metadata is not None and "trial_id" in ep.metadata else np.arange(len(ep))
    return X_sub, y_sub, tids, ep.times.copy(), ep.info.copy()


def _run_on_tv4_bundles_if_available(
    loaded_subs: list[str],
    data_version: str,
    baseline_mode: str = "zero_uv",
    models_dir: Path | None = None,
):
    """Nếu có gói bàn giao TV4 trong derivatives/models/, chạy IG & Occlusion chuẩn trên tập test từng fold."""
    if models_dir is None:
        models_dir = Path(DERIVATIVES_DIR) / "models"
    if not models_dir.exists():
        return None

    meta_files = sorted(models_dir.glob("*_meta.json"))
    if not meta_files:
        return None

    print(f"  -> Phát hiện {len(meta_files)} gói mô hình TV4 trong {models_dir}. Đang kiểm tra tập test không rò rỉ...")
    # Nhóm theo stem đầu tiên khả dụng (hỗ trợ cả within & cross)
    ig_trained_maps, ig_random_maps = [], []
    occ_dfs = []
    first_meta = None
    ref_info = None
    ref_times = None
    X_raw_all, y_raw_all = [], []

    sub_cache = {}
    for meta_file in meta_files[:15]:  # giới hạn 15 bundle đại diện (5 fold x 3 seed) để chạy nhanh & đủ đại diện
        stem = meta_file.name[: -len("_meta.json")]
        model, nrm, test, meta = load_tv4_bundle(stem, models_dir=models_dir)
        if test is None or nrm is None:
            continue
        if first_meta is None:
            first_meta = meta

        test_subs = np.unique(test["subject"].astype(str))
        X_te_list, y_te_list = [], []
        for s in test_subs:
            if s not in sub_cache:
                ep = read_one_subject_epochs(s, data_version=data_version)
                X_s, y_s, tids_s, times_s, info_s = _extract_dl_grid(
                    ep, sfreq_dl=meta.get("sfreq", 128.0), tmin_dl=meta.get("tmin", 0.0), tmax_dl=meta.get("tmax", 0.8)
                )
                sub_cache[s] = (X_s, y_s, tids_s, times_s, info_s)
                X_raw_all.append(X_s)
                y_raw_all.append(y_s)
                if ref_info is None:
                    ref_info, ref_times = info_s, times_s

            X_s, y_s, tids_s, _, _ = sub_cache[s]
            mask_s = test["subject"].astype(str) == s
            is_test = np.isin(tids_s, test["trial_id"][mask_s])
            if is_test.any():
                X_norm_s = ((X_s[is_test] - nrm["mean"][None, :, None]) / nrm["std"][None, :, None]).astype(np.float32)
                X_te_list.append(X_norm_s)
                y_te_list.append(y_s[is_test])

        if not X_te_list:
            continue
        X_te = np.concatenate(X_te_list, axis=0)
        y_te = np.concatenate(y_te_list, axis=0)
        fill_1d = compute_norm_zero_baseline(nrm["mean"], nrm["std"], baseline_mode=baseline_mode)

        X_te_tgt = torch.tensor(X_te[y_te == 1], dtype=torch.float32)
        ig_trained_maps.append(compute_ig_map(model, X_te_tgt, baseline_1d=fill_1d))

        torch.manual_seed(SEED)
        untrained = EEGNet(meta["n_ch"], meta["n_times"], n_classes=2, **meta.get("arch", {})).eval()
        ig_random_maps.append(compute_ig_map(untrained, X_te_tgt, baseline_1d=fill_1d))

        df_occ = run_eegnet_occlusion(
            model, X_te, y_te, times=ref_times, ch_names=meta["ch_names"],
            data_version=data_version, save_outputs=False, fill_1d=fill_1d,
        )
        occ_dfs.append(df_occ)

    if not ig_trained_maps:
        return None

    X_raw = np.concatenate(X_raw_all, axis=0)
    y_raw = np.concatenate(y_raw_all, axis=0)
    t_stat_map, _ = ttest_ind(X_raw[y_raw == 1], X_raw[y_raw == 0], axis=0, equal_var=False)
    mean_ig_tr = np.mean(ig_trained_maps, axis=0)
    mean_ig_rd = np.mean(ig_random_maps, axis=0)

    df_occ_mean = pd.concat(occ_dfs).groupby("condition", sort=False).mean(numeric_only=True).reset_index()
    return dict(
        ig_trained=mean_ig_tr,
        ig_random=mean_ig_rd,
        t_stat_map=t_stat_map,
        times=ref_times,
        info=ref_info,
        df_occ=df_occ_mean,
        source=f"TV4 Handover Bundles ({len(ig_trained_maps)} folds/seeds, baseline={baseline_mode})",
    )


def run_ig_and_sanity_check(epochs_list, loaded_subs, data_version="v0", baseline_mode: str = "zero_uv"):
    """Chạy Integrated Gradients, đóng gói mne.EvokedArray Topomap, Sanity Check (Hình 10) & Occlusion."""
    print(f"\n[3/4] Đang chạy Integrated Gradients (Captum) & Sanity Check (baseline={baseline_mode})...")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tv4_res = _run_on_tv4_bundles_if_available(loaded_subs, data_version=data_version, baseline_mode=baseline_mode)
    if tv4_res is not None:
        ig_trained = tv4_res["ig_trained"]
        ig_random = tv4_res["ig_random"]
        t_stat_map = tv4_res["t_stat_map"]
        times = tv4_res["times"]
        ref_info = tv4_res["info"]
        ref_tmin = float(times[0])
        ch_names = ref_info["ch_names"]
        model_source = tv4_res["source"]
        df_occ_precomputed = tv4_res["df_occ"]
    else:
        df_occ_precomputed = None
        # Fallback: chuẩn bị đúng lưới 128 Hz, 0.0–0.8 s giống TV4 để khớp kiến trúc & tiết kiệm RAM
        max_dl_subs = 12
        dl_subs = loaded_subs[:max_dl_subs]
        X_all, y_all = [], []
        ref_info = None
        ref_times = None

        for sub in dl_subs:
            epochs = (
                read_one_subject_epochs(sub, data_version=data_version)
                if isinstance(epochs_list[0], str)
                else epochs_list[loaded_subs.index(sub)]
            )
            X_sub, y_sub, _, times_sub, info_sub = _extract_dl_grid(epochs)
            if ref_info is None:
                ref_info = info_sub
                ref_times = times_sub
            X_all.append(X_sub)
            y_all.append(y_sub)
            del epochs

        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)
        del X_all, y_all

        n_trials, n_channels, n_times = X.shape
        times = ref_times
        ref_tmin = float(times[0])
        ch_names = ref_info["ch_names"]

        # Tách tập train/test để chuẩn hóa chỉ trên train và tính IG/Occlusion trên tập test (Zero Leakage)
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_trials)
        split = int(0.8 * n_trials)
        train_idx, test_idx = perm[:split], perm[split:]

        mean_ch = X[train_idx].mean(axis=(0, 2), keepdims=True)
        std_ch = X[train_idx].std(axis=(0, 2), keepdims=True) + 1e-6
        X_norm = ((X - mean_ch) / std_ch).astype(np.float32)
        fill_1d = compute_norm_zero_baseline(mean_ch, std_ch, baseline_mode=baseline_mode)

        X_test_norm = X_norm[test_idx]
        y_test = y[test_idx]
        X_test_t = torch.tensor(X_test_norm, dtype=torch.float32)
        X_test_target = X_test_t[y_test == 1]

        # 1. Mạng EEGNet TRỌNG SỐ NGẪU NHIÊN (Sanity Check - Untrained)
        torch.manual_seed(SEED)
        random_model = EEGNet(n_ch=n_channels, n_times=n_times, n_classes=2, drop=0.5)
        ig_random = compute_ig_map(random_model, X_test_target, baseline_1d=fill_1d)

        # 2. Mạng EEGNet ĐÃ HUẤN LUYỆN
        trained_model, model_source = _train_or_load_eegnet_for_explain(X_norm, y, train_idx)
        ig_trained = compute_ig_map(trained_model, X_test_target, baseline_1d=fill_1d)

        # 3. Tính bản đồ t-statistic sinh lý học của sóng hiệu ERP trên cùng lưới 128 Hz, 0.0–0.8 s
        t_stat_map, _ = ttest_ind(X[y == 1], X[y == 0], axis=0, equal_var=False)

    # Kiểm định tương quan hạng Spearman giữa |IG| và |t-stat ERP|
    rho_random, p_random = spearmanr(np.abs(ig_random).ravel(), np.abs(t_stat_map).ravel())
    rho_trained, p_trained = spearmanr(np.abs(ig_trained).ravel(), np.abs(t_stat_map).ravel())

    print(f"  * Nguồn mô hình: {model_source}")
    print(f"  * Sanity Check (Random Untrained EEGNet): Spearman r = {rho_random:.4f} (p = {p_random:.3e})")
    print(f"  * Trained EEGNet Attribution vs. ERP t-map: Spearman r = {rho_trained:.4f} (p = {p_trained:.3e})")

    # Đóng gói IG và ERP t-map vào mne.EvokedArray để chuẩn hóa không gian điện cực
    info = ref_info.copy()
    tmin = ref_tmin
    evoked_erp_t = _prepare_valid_topomap_evoked(
        t_stat_map / (np.max(np.abs(t_stat_map)) + 1e-8), info, tmin, comment="Normalized ERP t-map"
    )
    evoked_ig_tr = _prepare_valid_topomap_evoked(
        ig_trained / (np.max(np.abs(ig_trained)) + 1e-8), info, tmin, comment="Normalized Trained IG"
    )

    # Chọn các kênh đại diện từ trước ra sau (Frontal -> Central -> Parietal -> Occipital)
    roi_candidates = ["Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "C3", "C4", "O1", "O2"]
    roi_indices = [ch_names.index(c) for c in roi_candidates if c in ch_names]
    if not roi_indices:
        roi_indices = list(range(min(10, n_channels)))
    roi_labels = [ch_names[i] for i in roi_indices]

    # Vẽ Hình 10: Bảng 3 Heatmap (ERP |t| vs Trained |IG| vs Random |IG|) + Topomap P3b (350-450 ms) cùng vlim
    figstyle.apply()
    fig = plt.figure(figsize=(15.5, 5.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1.15, 0.75], wspace=0.28)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    gs_topo = gs[0, 3].subgridspec(2, 1, hspace=0.35)
    ax_topo_erp = fig.add_subplot(gs_topo[0, 0])
    ax_topo_ig = fig.add_subplot(gs_topo[1, 0])

    panels = [
        (np.abs(t_stat_map[roi_indices]), "ERP |t-statistic| (Target - Standard)\n[Physiological Ground Truth]", "YlOrRd"),
        (np.abs(ig_trained[roi_indices]), f"Trained EEGNet |IG| Attribution\n[Spearman r = {rho_trained:.3f}]", "YlOrRd"),
        (np.abs(ig_random[roi_indices]), f"Untrained Random EEGNet |IG|\n[Sanity Check: r = {rho_random:.3f}]", "YlOrRd"),
    ]

    for idx_ax, (ax, (mat, title, cmap)) in enumerate(zip(axes, panels)):
        vmax = np.percentile(mat, 98) if np.max(mat) > 0 else 1.0
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="upper",
            extent=[times[0], times[-1], len(roi_indices) - 0.5, -0.5],
            cmap=cmap,
            vmin=0,
            vmax=vmax,
        )
        ax.axvline(0, color="#222222", ls="-", lw=1.0)
        ax.axvline(0.3, color="#222222", ls="--", lw=0.9, alpha=0.8)
        ax.axvline(0.6, color="#222222", ls="--", lw=0.9, alpha=0.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel(figstyle.LABELS["time"])
        ax.grid(False)
        ax.set_yticks(range(len(roi_labels)))
        if idx_ax == 0:
            ax.set_yticklabels(roi_labels)
            ax.set_ylabel("Key Scalp Electrodes (Frontal → Parietal → Occipital)")
        else:
            ax.set_yticklabels([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Vẽ cặp Topomap P3b (trung bình 350–450 ms) từ mne.EvokedArray với cùng vlim=(-1, 1)
    p3b_mask = (evoked_erp_t.times >= 0.35) & (evoked_erp_t.times <= 0.45)
    erp_p3b_topo = np.abs(evoked_erp_t.data[:, p3b_mask]).mean(axis=1)
    ig_p3b_topo = np.abs(evoked_ig_tr.data[:, p3b_mask]).mean(axis=1)
    shared_vmax = float(max(np.max(erp_p3b_topo), np.max(ig_p3b_topo), 1e-6))

    mne.viz.plot_topomap(
        erp_p3b_topo, evoked_erp_t.info, axes=ax_topo_erp, show=False,
        vlim=(0, shared_vmax), cmap="YlOrRd", contours=4,
    )
    ax_topo_erp.set_title("ERP |t| Topo\n(350–450 ms)", fontsize=9.5, fontweight="bold")

    mne.viz.plot_topomap(
        ig_p3b_topo, evoked_ig_tr.info, axes=ax_topo_ig, show=False,
        vlim=(0, shared_vmax), cmap="YlOrRd", contours=4,
    )
    ax_topo_ig.set_title("Trained |IG| Topo\n(350–450 ms)", fontsize=9.5, fontweight="bold")

    fig.suptitle(
        "Figure 10: Integrated Gradients (IG) Spatiotemporal Saliency, Topomap & Random-Weight Sanity Check",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig_path = figstyle.save(fig, "fig10_ig_sanity_check", folder=FIGURES_DIR, save_pdf=True, close=True)

    # 4. Chạy Spatiotemporal Occlusion trên mạng EEGNet đã huấn luyện
    if df_occ_precomputed is not None:
        df_occ = df_occ_precomputed
        save_occlusion_outputs(df_occ, data_version=data_version)
    else:
        df_occ = run_eegnet_occlusion(
            trained_model, X_test_norm, y_test, times=times, ch_names=ch_names,
            data_version=data_version, fill_1d=fill_1d,
        )

    return {
        "rho_random": float(rho_random),
        "rho_trained": float(rho_trained),
        "fig_path": fig_path,
        "occlusion_df": df_occ,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", default="v0", choices=["v0", "v1"], help="Phiên bản dữ liệu (v0 hoặc v1)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Danh sách đối tượng (mặc định: tự động theo data-version)")
    parser.add_argument("--baseline-mode", default="zero_uv", choices=["zero_uv", "train_mean"],
                        help="Baseline cho IG & Occlusion: zero_uv (0 µV) hoặc train_mean (0.0 sau chuẩn hóa)")
    parser.add_argument("--skip-time-resolved", action="store_true", help="Bỏ qua bước 1 nếu đã chạy xong")
    args = parser.parse_args()

    if args.subjects is None:
        if args.data_version == "v0":
            args.subjects = SAMPLE_SUBJECTS
        else:
            ver_dir = Path(DERIVATIVES_DIR) / args.data_version
            found = sorted(
                p.name.split("_")[0].replace("sub-", "")
                for p in ver_dir.glob("sub-*_task-oddball-epo.fif")
            )
            args.subjects = found if found else SAMPLE_SUBJECTS

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    print("=" * 70)
    print("TV5 PIPELINE: HÌNH 1, TIME-RESOLVED DECODING, HAUFE PATTERN, IG & OCCLUSION")
    print("=" * 70)

    # 1. Vẽ Hình 1 (Sơ đồ Pipeline dự án)
    plot_pipeline_diagram()

    # 2. Đọc dữ liệu Epochs
    epochs_list, loaded_subs = load_subject_epochs(args.subjects, data_version=args.data_version)

    # 3. Chạy Giải mã theo thời gian (Tuần 1 & Hình 9)
    if not args.skip_time_resolved:
        run_time_resolved_decoding(epochs_list, loaded_subs, data_version=args.data_version)

    # 4. Tính Haufe Activation Pattern (Tuần 2)
    run_haufe_activation_pattern(epochs_list, loaded_subs, data_version=args.data_version)

    # 5. Chạy Integrated Gradients, Sanity Check (Hình 10) & Spatiotemporal Occlusion (Tuần 2)
    run_ig_and_sanity_check(epochs_list, loaded_subs, data_version=args.data_version, baseline_mode=args.baseline_mode)

    print("\n" + "=" * 70)
    print("HOÀN TẤT scripts/05_explain.py!")
    print("=" * 70)


if __name__ == "__main__":
    main()
