"""Ba thí nghiệm kiểm tra yếu tố gây nhiễu thực nghiệm (Confounds / RQ3 - TV5).

Kiểm chứng xem mô hình phân loại dựa vào sóng chú ý P3b (300–600 ms tại vùng đỉnh)
hay đang 'ăn gian' nhờ các biến gây nhiễu ngoài sự chú ý:
1. Confound 1 — Phản ứng vận động (bấm phím tay phải):
   - 1A (`CONF1A_OCCLUDE_MOTOR_C3`): Che khuất (occlusion) cụm kênh vận động quanh C3.
   - 1B (`CONF1B_PRE_300MS_ONLY`): Chỉ dùng dữ liệu trước 300 ms (trước cửa sổ P3b & bấm phím).
   - Đối chứng P3b (`CONF1C_P3B_300_600MS`): Chỉ lấy cửa sổ P3b (300–600 ms).
2. Confound 2 — Khác biệt thị giác vật lý sớm (1.22° vs 1.07°):
   - (`CONF2_EARLY_VISUAL_OCCIPITAL`): Chỉ dùng các kênh vùng chẩm (O1, O2, Oz, POz...) trong 80–200 ms.
3. Confound 3 — Tần suất / Độ hiếm (~12%):
   - (`CONF3_TARGET_VS_DISTRACTOR`): Phân loại trực tiếp Target (S6) vs. Distractor (S7) (cùng độ hiếm ~12%).

Xuất ra:
- `results/predictions_tv5_confounds.csv` (đúng hợp đồng cột của `evaluate.py` trên tập test `splits/folds.json`).
- `results/confounds_summary_metrics.csv` (bảng tổng hợp AUC & Balanced Accuracy của cả 3 thí nghiệm).
- `figures/fig_confounds_analysis.png` & `.pdf` (biểu đồ so sánh trực quan 3 thí nghiệm Confound).

Dùng:
    python scripts/06_confounds.py --data-version v0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eegoddball import figstyle
from eegoddball.config import (
    DERIVATIVES_DIR,
    FIGURES_DIR,
    FOLDS_PATH,
    RESULTS_DIR,
    SAMPLE_SUBJECTS,
    SEED,
    get_folds_hash,
)
from eegoddball.evaluate import REQUIRED_COLUMNS, validate_predictions

# Cụm kênh vỏ não vận động bán cầu trái (điều khiển tay phải bấm phím)
MOTOR_C3_CHANNELS = ["C3", "C1", "C5", "FC3", "FC1", "FC5", "CP3", "CP1", "CP5", "Cz"]

# Cụm kênh vùng chẩm thị giác sớm
OCCIPITAL_CHANNELS = ["O1", "O2", "Oz", "POz", "PO3", "PO4", "PO7", "PO8", "PO9", "PO10"]


def extract_spatiotemporal_features(
    epochs: mne.Epochs,
    tmin: float = 0.0,
    tmax: float = 0.8,
    include_channels: list[str] | None = None,
    exclude_channels: list[str] | None = None,
    n_time_bins: int = 16,
) -> np.ndarray:
    """Trích xuất đặc trưng trung bình theo các bin thời gian trên tập kênh được chọn."""
    ep_crop = epochs.copy().crop(tmin=tmin, tmax=tmax)
    ch_names = ep_crop.ch_names

    if include_channels is not None:
        valid_ch = [c for c in include_channels if c in ch_names]
        if valid_ch:
            ep_crop.pick(valid_ch)
    elif exclude_channels is not None:
        keep_ch = [c for c in ch_names if c not in exclude_channels]
        ep_crop.pick(keep_ch)

    data = ep_crop.get_data(copy=True)  # (n_trials, n_ch, n_times)
    n_trials, n_ch, n_times = data.shape
    bins = np.array_split(data, min(n_time_bins, n_times), axis=2)
    binned = np.stack([b.mean(axis=2) for b in bins], axis=2)
    return binned.reshape(n_trials, -1)


def build_classifier():
    """Bộ phân loại tuyến tính chuẩn hóa dùng chung cho các phép thử Confound."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=SEED),
    )


def run_confound_experiments(subjects: list[str], data_version: str = "v0"):
    """Chạy 3 thí nghiệm Confound theo đúng các fold trong splits/folds.json."""
    folds = json.loads(Path(FOLDS_PATH).read_text(encoding="utf-8"))
    folds_hash = get_folds_hash()
    ver_dir = Path(DERIVATIVES_DIR) / data_version

    # Định nghĩa các cấu hình thí nghiệm Confound cho bài toán Standard (0) vs Target (1)
    configs = [
        {
            "model_name": "CONF0_FULL_BASELINE",
            "label": "Baseline: Full Scalp (0–800 ms)",
            "tmin": 0.0,
            "tmax": 0.8,
            "include_channels": None,
            "exclude_channels": None,
        },
        {
            "model_name": "CONF1A_OCCLUDE_MOTOR_C3",
            "label": "Confound 1A: Motor C3 Occluded (0–800 ms)",
            "tmin": 0.0,
            "tmax": 0.8,
            "include_channels": None,
            "exclude_channels": MOTOR_C3_CHANNELS,
        },
        {
            "model_name": "CONF1B_PRE_300MS_ONLY",
            "label": "Confound 1B: Pre-P3b / Pre-Motor (0–300 ms)",
            "tmin": 0.0,
            "tmax": 0.3,
            "include_channels": None,
            "exclude_channels": None,
        },
        {
            "model_name": "CONF1C_P3B_WINDOW_ONLY",
            "label": "P3b Core Window Only (300–600 ms, No Motor C3)",
            "tmin": 0.3,
            "tmax": 0.6,
            "include_channels": None,
            "exclude_channels": MOTOR_C3_CHANNELS,
        },
        {
            "model_name": "CONF2_EARLY_VISUAL_OCCIPITAL",
            "label": "Confound 2: Early Visual Occipital O1/O2/POz (80–200 ms)",
            "tmin": 0.08,
            "tmax": 0.20,
            "include_channels": OCCIPITAL_CHANNELS,
            "exclude_channels": None,
        },
    ]

    # Đọc từng đối tượng 1 lần duy nhất và trích xuất sẵn đặc trưng cho mọi cấu hình (tiết kiệm 95% RAM)
    print(f"Đang trích xuất đặc trưng không-thời gian cho {len(subjects)} đối tượng ({data_version})...")
    features_by_cfg = {cfg["model_name"]: {} for cfg in configs}
    c3_sub_data = {}

    for sub in subjects:
        path = ver_dir / f"sub-{sub}_task-oddball-epo.fif"
        if not path.exists():
            raise FileNotFoundError(f"Chưa có file {path}")
        ep = mne.read_epochs(path, preload=True, verbose=False)
        md = ep.metadata

        # 1. Standard vs Target cho 5 cấu hình
        mask_bin = md["condition"].isin(["standard", "target"]).to_numpy()
        ep_bin = ep[np.flatnonzero(mask_bin)]
        y_bin = (ep_bin.metadata["condition"].to_numpy() == "target").astype(int)
        tids_bin = ep_bin.metadata["trial_id"].to_numpy(dtype=int)

        for cfg in configs:
            X_sub = extract_spatiotemporal_features(
                ep_bin,
                tmin=cfg["tmin"],
                tmax=cfg["tmax"],
                include_channels=cfg["include_channels"],
                exclude_channels=cfg["exclude_channels"],
            )
            features_by_cfg[cfg["model_name"]][sub] = (X_sub, y_bin, tids_bin)

        # 2. Target vs Distractor cho Confound 3
        mask_rare = md["condition"].isin(["distractor", "target"]).to_numpy()
        ep_rare = ep[np.flatnonzero(mask_rare)]
        X_rare = extract_spatiotemporal_features(ep_rare, tmin=0.0, tmax=0.8)
        y_rare = (ep_rare.metadata["condition"].to_numpy() == "target").astype(int)
        c3_sub_data[sub] = (X_rare, y_rare)
        del ep, ep_bin, ep_rare

    prediction_rows = []
    summary_records = []

    for cfg in configs:
        model_name = cfg["model_name"]
        print(f"\nĐang chạy cấu hình: {model_name} ({cfg['label']})...")
        sub_data = features_by_cfg[model_name]

        # 1. Đánh giá Within-Subject theo đúng splits/folds.json
        within_aucs = []
        within_baccs = []
        for sub in subjects:
            X_sub, y_sub, trial_ids = sub_data[sub]
            sub_folds = folds["within_subject"][sub]
            for fold_name, fold_info in sub_folds.items():
                test_ids = set(fold_info["test_trial_ids"])
                is_test = np.array([tid in test_ids for tid in trial_ids], dtype=bool)
                train_idx = np.flatnonzero(~is_test)
                test_idx = np.flatnonzero(is_test)

                clf = build_classifier()
                clf.fit(X_sub[train_idx], y_sub[train_idx])
                y_score = clf.predict_proba(X_sub[test_idx])[:, 1]
                y_pred = (y_score >= 0.5).astype(int)

                within_aucs.append(roc_auc_score(y_sub[test_idx], y_score))
                within_baccs.append(balanced_accuracy_score(y_sub[test_idx], y_pred))

                for tid, yt, ys in zip(trial_ids[test_idx], y_sub[test_idx], y_score):
                    prediction_rows.append({
                        "scheme": "within_subject",
                        "subject": sub,
                        "fold": fold_name,
                        "model": model_name,
                        "seed": SEED,
                        "trial_id": int(tid),
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "data_version": data_version,
                        "folds_hash": folds_hash,
                    })

        # 2. Đánh giá Cross-Subject theo đúng splits/folds.json
        cross_aucs = []
        cross_baccs = []
        for fold_name, fold_info in folds["cross_subject"].items():
            test_subs = [s for s in fold_info["test_subjects"] if s in subjects]
            train_subs = [s for s in subjects if s not in test_subs]
            if not test_subs or not train_subs:
                continue

            X_train = np.concatenate([sub_data[s][0] for s in train_subs], axis=0)
            y_train = np.concatenate([sub_data[s][1] for s in train_subs], axis=0)

            clf = build_classifier()
            clf.fit(X_train, y_train)

            for sub in test_subs:
                X_te, y_te, tids_te = sub_data[sub]
                y_score = clf.predict_proba(X_te)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
                cross_aucs.append(roc_auc_score(y_te, y_score))
                cross_baccs.append(balanced_accuracy_score(y_te, y_pred))

                for tid, yt, ys in zip(tids_te, y_te, y_score):
                    prediction_rows.append({
                        "scheme": "cross_subject",
                        "subject": sub,
                        "fold": fold_name,
                        "model": model_name,
                        "seed": SEED,
                        "trial_id": int(tid),
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "data_version": data_version,
                        "folds_hash": folds_hash,
                    })

        w_auc_mean, w_auc_std = float(np.mean(within_aucs)), float(np.std(within_aucs))
        c_auc_mean, c_auc_std = float(np.mean(cross_aucs)), float(np.std(cross_aucs))
        print(f"  -> Within-Subject AUC: {w_auc_mean:.4f} ± {w_auc_std:.4f} | Cross-Subject AUC: {c_auc_mean:.4f} ± {c_auc_std:.4f}")

        summary_records.append({
            "model": model_name,
            "description": cfg["label"],
            "within_auc_mean": w_auc_mean,
            "within_auc_std": w_auc_std,
            "within_bacc_mean": float(np.mean(within_baccs)),
            "cross_auc_mean": c_auc_mean,
            "cross_auc_std": c_auc_std,
            "cross_bacc_mean": float(np.mean(cross_baccs)),
        })

    # ==========================================================================
    # CONFOUND 3: TẦN SUẤT / ĐỘ HIẾM — PHÂN LOẠI TARGET (S6) VS. DISTRACTOR (S7)
    # ==========================================================================
    print("\nĐang chạy Confound 3: Target (S6, ~12%) vs. Distractor (S7, ~12%)...")
    c3_within_aucs, c3_within_baccs = [], []
    c3_cross_aucs, c3_cross_baccs = [], []
    for sub in subjects:
        X_sub, y_sub = c3_sub_data[sub]

        counts = np.bincount(y_sub, minlength=2)
        if counts.min() < 5:
            print(f"  [Confound 3] sub-{sub} chỉ có {counts[0]} distractor / {counts[1]} target (<5 mẫu) -> bỏ qua Within CV của người này.")
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        for tr_idx, te_idx in skf.split(X_sub, y_sub):
            clf = build_classifier()
            clf.fit(X_sub[tr_idx], y_sub[tr_idx])
            y_score = clf.predict_proba(X_sub[te_idx])[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            c3_within_aucs.append(roc_auc_score(y_sub[te_idx], y_score))
            c3_within_baccs.append(balanced_accuracy_score(y_sub[te_idx], y_pred))

    for fold_name, fold_info in folds["cross_subject"].items():
        test_subs = [s for s in fold_info["test_subjects"] if s in subjects]
        train_subs = [s for s in subjects if s not in test_subs]
        if not test_subs or not train_subs:
            continue
        X_tr = np.concatenate([c3_sub_data[s][0] for s in train_subs], axis=0)
        y_tr = np.concatenate([c3_sub_data[s][1] for s in train_subs], axis=0)
        clf = build_classifier()
        clf.fit(X_tr, y_tr)
        for sub in test_subs:
            X_te, y_te = c3_sub_data[sub]
            if len(np.unique(y_te)) < 2:
                continue
            y_score = clf.predict_proba(X_te)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            c3_cross_aucs.append(roc_auc_score(y_te, y_score))
            c3_cross_baccs.append(balanced_accuracy_score(y_te, y_pred))

    c3_w_mean, c3_w_std = float(np.mean(c3_within_aucs)), float(np.std(c3_within_aucs))
    c3_c_mean, c3_c_std = float(np.mean(c3_cross_aucs)), float(np.std(c3_cross_aucs))
    print(f"  -> Confound 3 (Target vs. Distractor) Within AUC: {c3_w_mean:.4f} ± {c3_w_std:.4f} | Cross AUC: {c3_c_mean:.4f} ± {c3_c_std:.4f}")

    summary_records.append({
        "model": "CONF3_TARGET_VS_DISTRACTOR",
        "description": "Confound 3: Rarity Control — Target (S6) vs. Distractor (S7)",
        "within_auc_mean": c3_w_mean,
        "within_auc_std": c3_w_std,
        "within_bacc_mean": float(np.mean(c3_within_baccs)),
        "cross_auc_mean": c3_c_mean,
        "cross_auc_std": c3_c_std,
        "cross_bacc_mean": float(np.mean(c3_cross_baccs)),
    })

    # Lưu và kiểm tra file dự đoán theo đúng hợp đồng của TV1 (`evaluate.py`)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pred_df = pd.DataFrame(prediction_rows)[REQUIRED_COLUMNS]
    pred_path = Path(RESULTS_DIR) / "predictions_tv5_confounds.csv"
    pred_df.to_csv(pred_path, index=False)

    # Tự kiểm tra hợp đồng dữ liệu với validate_predictions của TV1
    validate_predictions(pred_df.copy(), source=str(pred_path))
    print(f"\n[Hợp đồng dự đoán OK] Đã lưu và xác thực: {pred_path} ({len(pred_df)} dòng)")

    summary_df = pd.DataFrame(summary_records)
    summary_path = Path(RESULTS_DIR) / f"confounds_summary_{data_version}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Đã lưu bảng tóm tắt Confound tại: {summary_path}")

    # Vẽ biểu đồ trực quan hóa 3 thí nghiệm Confound
    plot_confounds_summary(summary_df)
    return summary_df


def plot_confounds_summary(summary_df: pd.DataFrame):
    """Vẽ biểu đồ cột ngang so sánh ROC AUC của Baseline và 3 thí nghiệm Confound."""
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(10, 5.2))

    y_pos = np.arange(len(summary_df))
    height = 0.36

    bars1 = ax.barh(
        y_pos - height / 2,
        summary_df["within_auc_mean"],
        xerr=summary_df["within_auc_std"],
        height=height,
        color=figstyle.COLORS["target"],
        alpha=0.88,
        label="Within-Subject (5-Fold CV)",
        capsize=3,
    )
    bars2 = ax.barh(
        y_pos + height / 2,
        summary_df["cross_auc_mean"],
        xerr=summary_df["cross_auc_std"],
        height=height,
        color=figstyle.COLORS["distractor"],
        alpha=0.88,
        label="Cross-Subject (LOSO/KFold)",
        capsize=3,
    )

    ax.axvline(0.5, color=figstyle.COLORS["baseline"], ls="--", lw=1.1, label="Chance Level (AUC = 0.50)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary_df["description"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel(figstyle.LABELS["auc"])
    ax.set_xlim(0.40, 1.05)
    ax.set_title("RQ3 Confound Verification: Motor (C3), Early Visual (O1/O2), and Stimulus Rarity Controls")
    ax.legend(loc="lower right")

    for bar in bars1:
        w = bar.get_width()
        ax.text(w + 0.015, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5, fontweight="bold")

    figstyle.save(fig, "fig_confounds_analysis", folder=FIGURES_DIR, save_pdf=True, close=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", default="v0", choices=["v0", "v1"], help="Phiên bản dữ liệu (v0 hoặc v1)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Danh sách đối tượng (mặc định: tự động theo data-version)")
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
    print("TV5 PIPELINE: 3 THÍ NGHIỆM KIỂM TRA YẾU TỐ GÂY NHIỄU (CONFOUNDS)")
    print("=" * 70)

    run_confound_experiments(args.subjects, data_version=args.data_version)

    print("\n" + "=" * 70)
    print("HOÀN TẤT scripts/06_confounds.py!")
    print("=" * 70)


if __name__ == "__main__":
    main()
