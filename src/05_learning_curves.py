"""TV4 · Learning curves của EEGNet từ results/history_tv4.csv.

Sở hữu: TV4. Hình dùng figstyle.apply()/save() của TV5 (quy định trong template PR).

Mỗi lượt chạy (scheme, subject, fold, seed) có loss train/val và AUC validation theo epoch.
Đường = median, dải = IQR (25–75%) giữa các lượt. Vì dừng sớm, các lượt dài ngắn khác nhau: ở epoch e chỉ tính các lượt còn
chạy tới e, và chỉ vẽ tới khi còn >= MIN_ACTIVE_FRAC số lượt (để đuôi không bị vài lượt chi phối).
Cột thứ ba là phân bố best_epoch (epoch có val loss nhỏ nhất — mô hình được giữ lại).

Chạy: python src/05_learning_curves.py
Ra:   figures/learning_curves_eegnet.png/.pdf và results/learning_curves_summary_tv4.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eegoddball import config as C          # noqa: E402
from eegoddball import figstyle             # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(C.RESULTS_DIR)
HISTORY = RESULTS_DIR / "history_tv4.csv"
SUMMARY = RESULTS_DIR / "learning_curves_summary_tv4.csv"
FIG_NAME = "learning_curves_eegnet"
MIN_ACTIVE_FRAC = 0.10
RUN = ["scheme", "subject", "fold", "seed"]
SCHEMES = [("within_subject", "Within-subject"), ("cross_subject", "Cross-subject")]
C_TRAIN = figstyle.COLORS["distractor"]
C_VAL = figstyle.COLORS["target"]
C_AUC = figstyle.COLORS["auc"]


def load_history(path=HISTORY):
    h = pd.read_csv(path, dtype={"subject": str})
    h["subject"] = h["subject"].fillna("all")          # cross: một mô hình cho nhiều người
    h = h[h.data_version == C.DATA_VERSION]
    if h.empty:
        raise SystemExit(f"{path}: không có dòng nào với data_version={C.DATA_VERSION}.")
    hashes = h.folds_hash.unique()
    if len(hashes) > 1:
        raise SystemExit(f"{path}: lẫn {len(hashes)} folds_hash — chạy lại 05_deep_learning.py.")
    return h


def per_epoch(h, col):
    """Median và IQR (25–75%) theo epoch trên các lượt còn chạy; cắt khi còn < MIN_ACTIVE_FRAC lượt."""
    g = h.groupby("epoch")[col]
    s = pd.DataFrame(dict(mid=g.median(), lo=g.quantile(.25), hi=g.quantile(.75), n=g.size()))
    n_runs = h.groupby(RUN).ngroups
    return s[s.n >= max(2, MIN_ACTIVE_FRAC * n_runs)]


def summarize(h):
    rows = []
    for scheme, sub in h.groupby("scheme"):
        runs = sub.groupby(RUN)
        best = runs.best_epoch.first()
        at_best = sub[sub.epoch == sub.best_epoch]
        rows.append(dict(
            scheme=scheme, n_runs=runs.ngroups,
            best_epoch_median=float(best.median()),
            best_epoch_q1=float(best.quantile(.25)), best_epoch_q3=float(best.quantile(.75)),
            n_best_epoch_le3=int((best <= 3).sum()),
            n_hit_max_epochs=int((runs.epoch.max() >= 150).sum()),
            train_loss_at_best=float(at_best.train_loss.mean()),
            val_loss_at_best=float(at_best.val_loss.mean()),
            val_auc_at_best_mean=float(at_best.val_auc.mean()),
            val_auc_at_best_sd=float(at_best.val_auc.std(ddof=0)),
        ))
    return pd.DataFrame(rows)


def band(ax, s, color, label):
    ax.plot(s.index, s["mid"], color=color, lw=1.6, label=label)
    ax.fill_between(s.index, s["lo"], s["hi"], color=color, alpha=.18, lw=0)


def plot(h):
    figstyle.apply()
    present = [(k, t) for k, t in SCHEMES if (h.scheme == k).any()]
    fig, axes = plt.subplots(len(present), 3, figsize=(10, 3.1 * len(present)),
                             squeeze=False, gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
    for row, (scheme, title) in zip(axes, present):
        sub = h[h.scheme == scheme]
        n_runs = sub.groupby(RUN).ngroups
        best = sub.groupby(RUN).best_epoch.first()
        med = best.median()

        ax = row[0]
        band(ax, per_epoch(sub, "train_loss"), C_TRAIN, "Train")
        band(ax, per_epoch(sub, "val_loss"), C_VAL, "Validation")
        ax.axvline(med, color=figstyle.COLORS["baseline"], ls=":", lw=1, label="Median best epoch")
        ax.set(title=f"{title}: loss ({n_runs} runs)", xlabel="Epoch", ylabel="Weighted cross-entropy")
        ax.legend(loc="upper right")

        ax = row[1]
        band(ax, per_epoch(sub, "val_auc"), C_AUC, "Validation")
        ax.axvline(med, color=figstyle.COLORS["baseline"], ls=":", lw=1)
        ax.axhline(.5, color=figstyle.COLORS["standard"], lw=.8, ls="--", label="Chance")
        ax.set(title=f"{title}: validation AUC", xlabel="Epoch", ylabel=figstyle.LABELS["auc"])
        ax.legend(loc="lower right")

        ax = row[2]
        bins = np.arange(0, best.max() + 11, 10) if len(best) > 30 else max(5, len(best) // 2)
        ax.hist(best, bins=bins, color=figstyle.COLORS["difference"], alpha=.8, edgecolor="white")
        ax.set(title=f"{title}: best epoch", xlabel="Best epoch (min. validation loss)",
               ylabel="Runs")
    fig.tight_layout()
    return fig


def main():
    h = load_history()
    summary = summarize(h)
    summary.to_csv(SUMMARY, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    if not (h.scheme == "cross_subject").any():
        print("Chưa có cross_subject trong history — hình chỉ có within.")
    figstyle.save(plot(h), FIG_NAME, close=True)


if __name__ == "__main__":
    main()
