"""Epoch v1 và ERP (TV2).

Các step:
  epochs  raw sạch -> derivatives/v1/sub-XX_task-oddball-epo.fif + self-check + số liệu QC (mặc định SAMPLE_SUBJECTS)
  qc      bảng QC + cờ + lưới ERP Pz -> results/qc_table.csv, reports/qc_report.html (mặc định SUBJECTS)
  evokeds ERP từng người + grand average -> derivatives/v1/erp/ (bỏ người trong excluded.json)
  cluster kiểm định cụm không gian–thời gian trên 2 sóng hiệu -> results/erp_clusters.csv, cluster_*.npz [--n-perm]
  peaks   mean amplitude / đỉnh / FAL50 P3a, P3b (cửa sổ theo tài liệu) -> results/erp_windows.json, erp_peaks*.csv
  figures Hình 2–5 -> figures/
  table   Table S1 QC -> results/table_S1_qc.csv/.md
  numbers mọi số liệu cho bài viết, kèm file nguồn -> results/numbers_for_writing.md

Dùng: python scripts/02_epochs_erp.py --step qc [--subjects 01 02]
"""
import argparse
import ast
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath
from scipy import stats

from eegoddball import config, figstyle

sys.path.insert(0, config.ROOT_DIR)  # create_v0_data.py nằm ở gốc repo
import create_v0_data  # noqa: E402

V1_DIR = Path(config.DERIVATIVES_DIR) / "v1"
QC_DIR = V1_DIR / "qc"
CLEAN_DIR = V1_DIR / "clean"
REPORTS_DIR = Path(config.ROOT_DIR) / "reports"
RESULTS_DIR = Path(config.RESULTS_DIR)
EXCLUDED_PATH = V1_DIR / "excluded.json"
ERP_DIR = V1_DIR / "erp"
GA_PATH = ERP_DIR / "grand_average-ave.fif"

RESPONSE_CODE = "S 1"
REJECT = dict(eeg=150e-6)

# chỉ số P3b nhanh cho QC (cửa sổ theo tài liệu, không chọn từ dữ liệu)
P3B_CH = "Pz"
P3B_WIN = (0.3, 0.6)
P3A_CH = "Fz"
P3A_WIN = (0.25, 0.40)
# sóng hiệu: tên comment -> (điều kiện hiếm, điều kiện so sánh)
CONTRASTS = {"target - standard": ("target", "standard"), "distractor - standard": ("distractor", "standard")}
# Giai đoạn 4, phương án (a) do TV2 chọn: cửa sổ theo tài liệu, không chọn từ sóng hiệu.
# window = đo mean amplitude; wide = tìm đỉnh và FAL50
COMPONENTS = {
    "P3a": {"channel": P3A_CH, "rare": "distractor", "window": P3A_WIN, "wide": (0.20, 0.45)},
    "P3b": {"channel": P3B_CH, "rare": "target", "window": P3B_WIN, "wide": (0.25, 0.70)},
}
CLUSTER_ALPHA = 0.05
FIG2_SUB = "01"
FIG2_CHANNELS = ["Fp1", "AFp2", "AF7", "AF8", "Fz"]
FIG2_WINDOW_S = 5
FIG5_TIMES = (0.30, 0.40, 0.50, 0.60)  # thời điểm topomap cố định cho Hình 5a, 5c và figS
# ngưỡng gắn cờ QC
MAX_PCT_REJECTED = 25
MIN_TARGET = 40
MIN_DISTRACTOR = 30  # thấp: người bấm nút với distractor (chỉ giữ thử nghiệm đúng)
ICA_MIN, ICA_MAX = 1, 3
MAX_BAD_CH = 13  # ~10% của 127 kênh
MIN_P3B_UV = 0


def events_tsv(sub):
    return (BIDSPath(subject=sub, task="oddball", datatype="eeg", root=config.DATA_DIR)
            .update(suffix="events", extension=".tsv").fpath)


def read_metadata(sub):
    """trial_id/condition theo TV1 (cùng cách đánh số với v0, folds.json), thêm hành vi và đúng/sai."""
    path = events_tsv(sub)
    metadata = create_v0_data.read_stimulus_metadata(path, config.get_expected_trials(sub))
    events_df = pd.read_csv(path, sep="\t")
    codes = events_df["event_type"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    resp_on = np.sort(events_df.loc[codes == RESPONSE_CODE, "onset"].to_numpy(float))
    stim_on = metadata["onset"].to_numpy(float)

    # mỗi phản hồi thuộc về kích thích gần nhất trước nó; không đặt ngưỡng RT
    idx = np.searchsorted(stim_on, resp_on, side="right") - 1
    assert (idx >= 0).all(), f"{path}: có phản hồi trước kích thích đầu tiên"
    rt = pd.Series(resp_on - stim_on[idx], index=idx)

    metadata["n_resp"] = rt.groupby(level=0).size().reindex(metadata.index, fill_value=0).to_numpy()
    metadata["rt"] = rt.groupby(level=0).min().reindex(metadata.index).to_numpy()  # RT phản hồi đầu tiên
    metadata["correct"] = (metadata["condition"] == "target") == (metadata["n_resp"] > 0)
    assert metadata["n_resp"].sum() == len(resp_on)
    return metadata


def make_epochs(raw, metadata):
    stim_on = metadata["onset"].to_numpy(float)
    ann = raw.annotations
    ann_on = ann.onset[ann.description == "stimulus"] - (raw.first_time if ann.orig_time else 0)
    assert len(ann_on) == len(stim_on) and np.allclose(ann_on, stim_on, atol=2e-3), "events.tsv lệch marker của raw"

    samples = raw.time_as_index(stim_on, use_rounding=True) + raw.first_samp
    events = np.column_stack([samples, np.zeros(len(metadata), int), metadata["event_code"].to_numpy(int)])
    assert len(events) == len(metadata) and (np.diff(samples) > 0).all()

    return mne.Epochs(raw, events, event_id=config.EVENT_ID, tmin=config.TMIN, tmax=config.TMAX,
                      baseline=(config.TMIN, 0), reject=REJECT, metadata=metadata, preload=True,
                      reject_by_annotation=False, event_repeated="error")


def self_check(path, metadata):
    """Đọc lại file đã lưu và kiểm tra hợp đồng dữ liệu v1."""
    epochs = mne.read_epochs(path)
    md = epochs.metadata
    assert epochs.event_id == config.EVENT_ID
    assert epochs.info["sfreq"] == config.SFREQ_NEW
    assert np.isclose(epochs.tmin, config.TMIN) and np.isclose(epochs.tmax, config.TMAX)
    assert epochs.info["custom_ref_applied"] and epochs.info["bads"] == []
    assert {"trial_id", "condition", "event_code", "correct", "n_resp", "rt"} <= set(md.columns)

    ids = md["trial_id"].to_numpy()
    assert len(np.unique(ids)) == len(ids) and set(ids) <= set(metadata["trial_id"])
    # trial_id không bị đánh số lại: vẫn trỏ đúng onset gốc và đúng vị trí trong events ban đầu
    assert np.array_equal(md["onset"].to_numpy(), metadata["onset"].to_numpy()[ids])
    assert np.array_equal(epochs.selection, ids)

    assert md["correct"].astype(bool).all()
    assert np.array_equal(epochs.events[:, 2], md["event_code"].to_numpy())
    assert np.array_equal(md["condition"].map(config.EVENT_ID).to_numpy(), md["event_code"].to_numpy())
    # đơn vị volt: reject 150 µV đỉnh-đỉnh nên sau baseline mọi giá trị đều nhỏ hơn 150e-6
    assert np.abs(epochs.get_data()).max() <= REJECT["eeg"]


def process(sub):
    qc_file = QC_DIR / f"sub-{sub}_qc.json"
    qc = json.loads(qc_file.read_text(encoding="utf-8"))
    assert qc["approved"], f"sub-{sub}: QC chưa duyệt"

    raw = mne.io.read_raw_fif(CLEAN_DIR / f"sub-{sub}_task-oddball_clean_raw.fif", preload=True)
    metadata = read_metadata(sub)
    epochs_all = make_epochs(raw, metadata)
    epochs = epochs_all[np.flatnonzero(epochs_all.metadata["correct"].to_numpy())]
    epochs.event_id = dict(config.EVENT_ID)  # giữ đủ 3 lớp cả khi một lớp không còn epoch nào

    out_path = V1_DIR / f"sub-{sub}_task-oddball-epo.fif"
    epochs.save(out_path, overwrite=True)
    self_check(out_path, metadata)

    channels = pd.Series([ch for log in epochs_all.drop_log for ch in log], dtype=str).value_counts()
    counts = epochs.metadata["condition"].value_counts()
    qc["epochs"] = {
        "n_stimuli": len(metadata),
        "n_rejected_150uV": len(metadata) - len(epochs_all),
        "reject_channels_top": channels.head(5).to_dict(),
        "correct_by_class": metadata.loc[metadata["correct"], "condition"].value_counts().to_dict(),
        "final_by_class": {c: int(counts.get(c, 0)) for c in config.EVENT_ID},
    }
    qc_file.write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sub-{sub}: self-check OK | {qc['epochs']['final_by_class']} | "
          f"reject {qc['epochs']['n_rejected_150uV']}/{len(metadata)} | {out_path}")


def as_dict(value):
    """bads/ica_exclude: dict, chuỗi của dict (JSON hoặc repr Python) hoặc list tên."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    return dict(value) if isinstance(value, dict) else dict.fromkeys(value)


def read_qc(sub):
    return json.loads((QC_DIR / f"sub-{sub}_qc.json").read_text(encoding="utf-8"))


def excluded():
    """Danh sách loại do TV2 tự quyết: "all" = mọi phân tích ERP, "distractor_contrast" = chỉ distractor − standard và P3a.
    Chưa có file thì trả về hai danh sách rỗng; hàm này không bao giờ tạo file."""
    data = json.loads(EXCLUDED_PATH.read_text(encoding="utf-8")) if EXCLUDED_PATH.exists() else {}
    out = {key: [str(s) for s in data.get(key, [])] for key in ("all", "distractor_contrast")}
    unknown = {s for subs in out.values() for s in subs} - set(config.SUBJECTS)
    assert not unknown, f"{EXCLUDED_PATH}: mã người không có trong config.SUBJECTS: {sorted(unknown)} (dạng '07')"
    return out


def make_qc_table(subjects):
    """Một dòng mỗi người, số liệu lấy từ sub-XX_qc.json; p3b_diff_uV và ERP Pz tính từ file epochs v1.
    % loại chỉ tính trên thử nghiệm đúng (correct_by_class -> final_by_class), tức phần mất do nhiễu 150 µV."""
    rows, pz = [], {}
    for sub in subjects:
        qc = read_qc(sub)
        n_before = sum(qc["epochs"]["correct_by_class"].values())
        final = {c: int(qc["epochs"]["final_by_class"].get(c, 0)) for c in config.EVENT_ID}

        epochs = mne.read_epochs(V1_DIR / f"sub-{sub}_task-oddball-epo.fif", preload=True)
        n_file = {c: len(epochs[c]) if c in epochs.metadata["condition"].values else 0 for c in config.EVENT_ID}
        assert n_file == final, f"sub-{sub}: QC JSON {final} khác file epochs {n_file}"
        pz[sub] = {c: epochs[c].average(picks=P3B_CH) for c in ("standard", "target")}
        diff = mne.combine_evoked([pz[sub]["target"], pz[sub]["standard"]], weights=[1, -1])

        rows.append({
            "subject": sub,
            "n_bad_ch": len(as_dict(qc["bads"])),
            "n_bridged": len(qc["bridged_pairs"]),
            "n_ica_removed": len(as_dict(qc["ica_exclude"])),
            "n_before": n_before,
            "n_standard": final["standard"],
            "n_target": final["target"],
            "n_distractor": final["distractor"],
            "pct_rejected": round(100 * (n_before - sum(final.values())) / n_before, 2),
            "p3b_diff_uV": round(1e6 * diff.copy().crop(*P3B_WIN).data.mean(), 2),
        })
    df = pd.DataFrame(rows)
    rules = {
        f"pct_rejected > {MAX_PCT_REJECTED}": df["pct_rejected"] > MAX_PCT_REJECTED,
        f"n_target < {MIN_TARGET}": df["n_target"] < MIN_TARGET,
        f"n_distractor < {MIN_DISTRACTOR}": df["n_distractor"] < MIN_DISTRACTOR,
        f"n_ica_removed ngoài {ICA_MIN}–{ICA_MAX}": ~df["n_ica_removed"].between(ICA_MIN, ICA_MAX),
        f"n_bad_ch > {MAX_BAD_CH}": df["n_bad_ch"] > MAX_BAD_CH,
        f"p3b_diff_uV <= {MIN_P3B_UV}": df["p3b_diff_uV"] <= MIN_P3B_UV,
    }
    df["flag"] = ["; ".join(rule for rule, hit in rules.items() if hit[i]) for i in df.index]
    return df, pz


def plot_pz_grid(pz, df):
    figstyle.apply()
    fig, axes = plt.subplots(6, 7, figsize=(16, 12), sharex=True, sharey=True)
    for ax, sub in zip(axes.flat, pz):
        for cond in ("standard", "target"):
            ev = pz[sub][cond]
            ax.plot(ev.times, 1e6 * ev.data[0], color=figstyle.COLORS[cond], lw=1, label=cond)
        ax.axvline(0, color="k", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        flagged = df.set_index("subject").loc[sub, "flag"] != ""
        ax.set_title(f"sub-{sub}" + (" *" if flagged else ""), color="tab:red" if flagged else "k", fontsize=9)
    for ax in axes.flat[len(pz):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(figstyle.LABELS["time"])
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{P3B_CH} (µV)")
    axes.flat[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(f"ERP tại {P3B_CH} từng người (* = bị gắn cờ QC)")
    return fig


def step_qc(subjects):
    df, pz = make_qc_table(subjects)
    RESULTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    df.to_csv(RESULTS_DIR / "qc_table.csv", index=False)
    flagged = df[df["flag"] != ""]

    # figstyle đặt savefig.bbox="tight" toàn cục, làm mne.Report lỗi "buffer is not large enough": chỉ dùng trong khối này
    with plt.rc_context():
        fig = plot_pz_grid(pz, df)
        grid_png = figstyle.save(fig, "qc_pz_grid", folder=str(REPORTS_DIR), save_pdf=False, close=True)

    report = mne.Report(title="TV2 QC – epochs v1")
    report.add_html(df.to_html(index=False), title="Bảng QC (results/qc_table.csv)")
    report.add_html(flagged[["subject", "flag"]].to_html(index=False) if len(flagged) else "<p>Không có</p>",
                    title=f"Người bị gắn cờ ({len(flagged)})")
    report.add_image(grid_png, title=f"ERP tại {P3B_CH} từng người")
    for sub in subjects:
        ica_file = QC_DIR / f"sub-{sub}-ica.fif"  # bản hiện hành, không dùng archive_pipeline_cu_01_03/
        if ica_file.exists():
            ica = mne.preprocessing.read_ica(ica_file)
            ica.exclude = [int(k) for k in as_dict(read_qc(sub)["ica_exclude"])]
            # inst=None: chỉ topomap thành phần; raw trước ICA không lưu (hình properties có ở qc/sub-XX/)
            report.add_ica(ica, title=f"sub-{sub} ICA (loại {ica.exclude})", inst=None, picks=ica.exclude,
                           tags=("ica",))
            plt.close("all")
    report.save(REPORTS_DIR / "qc_report.html", overwrite=True, open_browser=False)

    print(f"{len(df)} người | tổng kênh hỏng {df['n_bad_ch'].sum()} | tổng IC loại {df['n_ica_removed'].sum()} | "
          f"epoch còn lại {df[['n_standard', 'n_target', 'n_distractor']].sum().to_dict()}")
    print(f"excluded.json: {excluded()}")
    print(f"\nNgười bị gắn cờ ({len(flagged)}):")
    print(flagged.to_string(index=False))
    print(f"\n-> {RESULTS_DIR / 'qc_table.csv'}\n-> {REPORTS_DIR / 'qc_report.html'}")


def analysis_subjects(name):
    """Người dùng cho một điều kiện/sóng hiệu: bỏ "all"; mọi thứ có distractor bỏ thêm "distractor_contrast"."""
    exc = excluded()
    drop = set(exc["all"]) | (set(exc["distractor_contrast"]) if "distractor" in name else set())
    return [s for s in config.SUBJECTS if s not in drop]


def ave_path(sub):
    return ERP_DIR / f"sub-{sub}_task-oddball-ave.fif"


def subject_evokeds(sub):
    """3 điều kiện đã lưu + 2 sóng hiệu tính lại (trọng số 1, −1) của một người."""
    evs = {ev.comment: ev for ev in mne.read_evokeds(ave_path(sub))}
    for name, (rare, ref) in CONTRASTS.items():
        evs[name] = mne.combine_evoked([evs[rare], evs[ref]], weights=[1, -1])
        evs[name].comment = name
    return evs


def mean_uv(evoked, ch, win):
    return 1e6 * evoked.copy().pick([ch]).crop(*win).data.mean()


def step_evokeds():
    ERP_DIR.mkdir(exist_ok=True)
    subs = analysis_subjects("")
    baselines = set()
    for sub in subs:
        epochs = mne.read_epochs(V1_DIR / f"sub-{sub}_task-oddball-epo.fif", preload=True)
        evokeds = []
        for cond in config.EVENT_ID:
            ev = epochs[cond].average()
            ev.comment = cond
            if epochs.baseline is None:  # v1 trừ baseline lúc cắt epoch; nhánh này chỉ phòng khi hợp đồng đổi
                ev.apply_baseline((None, 0))
            evokeds.append(ev)
        baselines.add(None if epochs.baseline is None else tuple(round(b, 3) for b in epochs.baseline))
        mne.write_evokeds(ave_path(sub), evokeds, overwrite=True)
    print(f"Baseline của epochs: {baselines} (None = vừa trừ (None, 0) trên evoked; khác None = đã trừ sẵn, không trừ lại)")

    evs = {sub: subject_evokeds(sub) for sub in subs}
    ga = []
    for name in [*config.EVENT_ID, *CONTRASTS]:
        group = analysis_subjects(name)
        g = mne.grand_average([evs[s][name] for s in group])  # trọng số bằng nhau cho mỗi người
        g.comment = name
        ga.append(g)
        print(f"grand average '{name}': n = {g.nave}")
    mne.write_evokeds(GA_PATH, ga, overwrite=True)

    print("\nKiểm tra nhanh (trung bình qua người của mean amplitude trong cửa sổ):")
    for name, ch, win in [("target - standard", P3B_CH, P3B_WIN), ("distractor - standard", P3A_CH, P3A_WIN)]:
        vals = np.array([mean_uv(evs[s][name], ch, win) for s in analysis_subjects(name)])
        print(f"  {name} tại {ch} {int(win[0] * 1000)}–{int(win[1] * 1000)} ms: {vals.mean():.2f} ± {vals.std(ddof=1):.2f} µV "
              f"(n = {len(vals)}, {(vals > 0).sum()} người dương)")

    ga_diff = next(g for g in ga if g.comment == "target - standard")
    fig = ga_diff.plot_joint(times="peaks", title=f"Grand average target − standard (n = {ga_diff.nave})", show=False)
    REPORTS_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORTS_DIR / "ga_target_minus_standard_joint.png", dpi=150)
    plt.close("all")
    print(f"\n-> {ERP_DIR}\n-> {GA_PATH}\n-> {REPORTS_DIR / 'ga_target_minus_standard_joint.png'}")


def slug(name):
    return name.replace(" - ", "_minus_")


def read_ga():
    return {ev.comment: ev for ev in mne.read_evokeds(GA_PATH)}


def fmt_p(p):
    return "< .001" if p < .001 else f"{p:.3f}".lstrip("0")


# ---------------- Giai đoạn 3: kiểm định cụm ----------------
def step_cluster(n_perm):
    """Kiểm định cụm 1 mẫu (sóng hiệu khác 0?) trên ERP từng người: đơn vị là người, 0–0,8 s, mọi kênh."""
    info = read_ga()["target - standard"].info
    adjacency, ch_names = mne.channels.find_ch_adjacency(info, ch_type="eeg")
    fig = mne.viz.plot_ch_adjacency(info, adjacency, ch_names, kind="2d", edit=False)
    fig.savefig(REPORTS_DIR / "ch_adjacency.png", dpi=150)
    plt.close(fig)

    rows = []
    for name in CONTRASTS:
        subs = analysis_subjects(name)
        evs = [subject_evokeds(s)[name].crop(0, config.TMAX) for s in subs]
        assert all(ev.ch_names == ch_names for ev in evs)
        X = np.stack([ev.data.T for ev in evs])  # (người, thời điểm, kênh)
        times = evs[0].times
        threshold = stats.t.ppf(1 - CLUSTER_ALPHA / 2, len(subs) - 1)
        t0 = time.perf_counter()
        T_obs, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            X, threshold=threshold, n_permutations=n_perm, tail=0, adjacency=adjacency, out_type="mask",
            seed=config.SEED, n_jobs=-1)
        print(f"{name}: n = {len(subs)}, {n_perm} hoán vị, {len(clusters)} cụm, {time.perf_counter() - t0:.0f} s")

        sig_mask = np.zeros_like(T_obs, bool)
        for mask, p in zip(clusters, p_values):
            if p < CLUSTER_ALPHA:
                sig_mask |= mask
                t_idx, ch_idx = mask.any(axis=1), mask.any(axis=0)
                mass = T_obs[mask].sum()
                rows.append({"contrast": name, "n": len(subs), "n_perm": n_perm, "p": p, "sign": "+" if mass > 0 else "-",
                             "t_start_ms": round(1000 * times[t_idx][0]), "t_end_ms": round(1000 * times[t_idx][-1]),
                             "n_channels": int(ch_idx.sum()), "cluster_mass": round(float(mass), 1),
                             "channels": " ".join(np.array(ch_names)[ch_idx])})
        np.savez(RESULTS_DIR / f"cluster_{slug(name)}.npz", T_obs=T_obs, sig_mask=sig_mask, times=times,
                 ch_names=np.array(ch_names), n=len(subs), n_perm=n_perm)
    df = pd.DataFrame(rows, columns=["contrast", "n", "n_perm", "p", "sign", "t_start_ms", "t_end_ms", "n_channels",
                                     "cluster_mass", "channels"])
    df.to_csv(RESULTS_DIR / "erp_clusters.csv", index=False)
    print(df.drop(columns="channels").to_string(index=False))
    print(f"-> {RESULTS_DIR / 'erp_clusters.csv'}, cluster_*.npz, {REPORTS_DIR / 'ch_adjacency.png'}")


# ---------------- Giai đoạn 4: P3a / P3b ----------------
def fal50(evoked, ch, win):
    """Fractional area latency 50% trên phần dương trong cửa sổ; không có phần dương -> NaN."""
    ev = evoked.copy().pick([ch]).crop(*win)
    area = np.cumsum(np.clip(ev.data[0], 0, None))
    return np.nan if area[-1] == 0 else ev.times[np.searchsorted(area, 0.5 * area[-1])]


def step_peaks():
    windows = {"method": "literature windows (a); not selected from difference waves",
               "components": {comp: {"channel": c["channel"], "rare": c["rare"], "window_s": list(c["window"]),
                                     "wide_window_s": list(c["wide"])} for comp, c in COMPONENTS.items()}}
    (RESULTS_DIR / "erp_windows.json").write_text(json.dumps(windows, indent=2), encoding="utf-8")

    rows = []
    for comp, c in COMPONENTS.items():
        ch = c["channel"]
        for sub in analysis_subjects(c["rare"]):
            evs = subject_evokeds(sub)
            for role, cond in (("rare", c["rare"]), ("standard", "standard")):
                try:
                    _, lat, amp = evs[cond].copy().pick([ch]).get_peak(tmin=c["wide"][0], tmax=c["wide"][1], mode="pos",
                                                                      return_amplitude=True)
                except ValueError:  # không có giá trị dương trong cửa sổ
                    lat = amp = np.nan
                rows.append({"subject": sub, "component": comp, "channel": ch, "role": role, "condition": cond,
                             "mean_uV": mean_uv(evs[cond], ch, c["window"]), "peak_uV": 1e6 * amp,
                             "peak_lat_ms": 1000 * lat, "fal50_ms": 1000 * fal50(evs[cond], ch, c["wide"])})
    peaks = pd.DataFrame(rows).round(3)
    peaks.to_csv(RESULTS_DIR / "erp_peaks.csv", index=False)

    summary = []
    for comp, c in COMPONENTS.items():
        p = peaks[peaks["component"] == comp]
        amp = p.pivot(index="subject", columns="role", values="mean_uV")
        rare, std = amp["rare"], amp["standard"]
        d = rare - std
        t, p_t = stats.ttest_rel(rare, std)
        w, p_w = stats.wilcoxon(rare, std)
        lat = p[p["role"] == "rare"]
        q = lambda s: np.nanpercentile(s, [50, 25, 75])  # noqa: E731
        fal, pk = q(lat["fal50_ms"]), q(lat["peak_lat_ms"])
        summary.append({
            "component": comp, "channel": c["channel"], "rare": c["rare"], "n": len(d),
            "window_ms": f"{1000 * c['window'][0]:.0f}-{1000 * c['window'][1]:.0f}",
            "wide_window_ms": f"{1000 * c['wide'][0]:.0f}-{1000 * c['wide'][1]:.0f}",
            "rare_mean_uV": rare.mean(), "rare_sd_uV": rare.std(ddof=1),
            "standard_mean_uV": std.mean(), "standard_sd_uV": std.std(ddof=1),
            "diff_mean_uV": d.mean(), "diff_sd_uV": d.std(ddof=1),
            "t": t, "df": len(d) - 1, "p_t": p_t, "wilcoxon_W": w, "p_wilcoxon": p_w, "dz": d.mean() / d.std(ddof=1),
            "rare_fal50_median_ms": fal[0], "rare_fal50_q1_ms": fal[1], "rare_fal50_q3_ms": fal[2],
            "rare_peak_lat_median_ms": pk[0], "rare_peak_lat_q1_ms": pk[1], "rare_peak_lat_q3_ms": pk[2],
            "n_no_positive_peak": int(lat["peak_lat_ms"].isna().sum()),
        })
    summary = pd.DataFrame(summary).round(3)
    summary.to_csv(RESULTS_DIR / "erp_peaks_summary.csv", index=False)

    print(f"{'Component':<9} | Channel | Window (ms) | Rare M±SD (µV) | Standard M±SD (µV) | t(df) | p | dz | "
          f"FAL50 rare median [IQR] (ms) | Peak lat rare median [IQR] (ms)")
    for _, r in summary.iterrows():
        print(f"{r['component']:<9} | {r['channel']} | {r['window_ms']} | {r['rare_mean_uV']:.2f} ± {r['rare_sd_uV']:.2f} | "
              f"{r['standard_mean_uV']:.2f} ± {r['standard_sd_uV']:.2f} | t({r['df']}) = {r['t']:.2f} | {fmt_p(r['p_t'])} | "
              f"{r['dz']:.2f} | {r['rare_fal50_median_ms']:.0f} [{r['rare_fal50_q1_ms']:.0f}–{r['rare_fal50_q3_ms']:.0f}] | "
              f"{r['rare_peak_lat_median_ms']:.0f} [{r['rare_peak_lat_q1_ms']:.0f}–{r['rare_peak_lat_q3_ms']:.0f}]"
              f"  (n = {r['n']}, Wilcoxon p {fmt_p(r['p_wilcoxon'])})")
    print(f"-> {RESULTS_DIR / 'erp_windows.json'}, erp_peaks.csv, erp_peaks_summary.csv")


# ---------------- Giai đoạn 5: hình ----------------
def fig4_erp(windows):
    """Fz | Pz, ERP từng người -> trung bình và CI 95% qua người; tô xám cửa sổ đo của thành phần."""
    evs = {s: subject_evokeds(s) for s in analysis_subjects("")}
    data, colors, styles = {}, {}, {}
    for cond in config.EVENT_ID:
        subs = analysis_subjects(cond)
        label = f"{cond} (n = {len(subs)})"
        data[label], colors[label], styles[label] = [evs[s][cond] for s in subs], figstyle.COLORS[cond], \
            figstyle.LINESTYLES[cond]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for i, (ax, comp) in enumerate(zip(axes, ["P3a", "P3b"])):
        c = windows["components"][comp]
        mne.viz.plot_compare_evokeds(data, picks=c["channel"], axes=ax, ci=0.95, colors=colors, linestyles=styles,
                                     title=c["channel"], legend="upper left" if i == 0 else False, show_sensors=False,
                                     truncate_xaxis=False, truncate_yaxis=False, vlines=[], show=False)
        ax.axvspan(*c["window_s"], color="0.88", zorder=0, lw=0)
        ax.set_xlabel(figstyle.LABELS["time"])
        ax.set_ylabel(figstyle.LABELS["amplitude"] if i == 0 else "")
    # plot_compare_evokeds đặt ylim rộng hơn cần: lấy theo dữ liệu các đường ERP, chừa chỗ cho dải CI; sharey dùng chung
    ys = np.concatenate([line.get_ydata() for ax in axes for line in ax.lines if len(line.get_ydata()) > 2])
    axes[0].set_ylim(ys.min() - 1.5, ys.max() + 1.5)
    for ax in axes:
        ax.yaxis.set_major_locator(matplotlib.ticker.AutoLocator())  # MNE cố định ticks theo ylim cũ
        # lưới dọc nét đứt của figstyle tại -0.2 s đè lên trục y thành vạch thừa; giữ lưới ngang
        ax.set_xlim(config.TMIN, config.TMAX)
        ax.grid(False, axis="x")
        ax.axvline(0, color=figstyle.COLORS["baseline"], lw=0.8)
    return fig


def full_mask(ga, npz):
    """Đặt sig_mask (thời điểm 0–0,8 s × kênh) vào trục thời gian đầy đủ của grand average (kênh × thời điểm)."""
    mask = np.zeros(ga.data.shape, bool)
    start = int(np.argmin(np.abs(ga.times - npz["times"][0])))
    stop = start + len(npz["times"])
    assert np.allclose(ga.times[start:stop], npz["times"]) and list(npz["ch_names"]) == ga.ch_names
    mask[:, start:stop] = npz["sig_mask"].T
    return mask


def plot_image_by_region(ga, mask, vmax, label):
    """Ảnh kênh × thời gian, kênh xếp theo vùng (REGIONS), vạch ngăn và tên vùng trên trục y; vùng đầu tiên ở trên."""
    order = [ch for region in REGIONS for ch in ga.ch_names if region_of(ch) == region]
    idx = [ga.ch_names.index(ch) for ch in order]
    fig = ga.copy().reorder_channels(order).plot_image(mask=mask[idx], mask_style="mask", clim=dict(eeg=[-vmax, vmax]),
                                                      show=False, titles=dict(eeg=label))
    ax = fig.axes[0]
    lo = ax.images[0].get_extent()[2]  # cạnh dưới của hàng kênh đầu tiên
    sizes = [sum(region_of(ch) == region for ch in order) for region in REGIONS]
    edges = lo + np.cumsum([0, *sizes])
    for y in edges[1:-1]:
        ax.axhline(y, color="k", lw=0.8)
    ax.set_yticks((edges[:-1] + edges[1:]) / 2, list(REGIONS))
    ax.set_ylim(edges[-1], edges[0])  # đảo trục: frontal ở trên
    ax.set_ylabel("Channels (by region)")
    ax.grid(False)
    fig.set_size_inches(10, 4.5)
    return fig


def fig5_contrast(name, prefix):
    ga = read_ga()[name]
    mask = full_mask(ga, np.load(RESULTS_DIR / f"cluster_{slug(name)}.npz"))
    times = list(FIG5_TIMES)
    vmax = 1e6 * np.abs(ga.data[:, ga.time_as_index(times)]).max()  # cùng thang màu cho 5a, 5b, 5c
    vlim = (-vmax, vmax)
    label = name.replace(" - ", " − ")
    fig = ga.plot_joint(times=times, title=f"{label} (n = {ga.nave})", topomap_args=dict(vlim=vlim), show=False)
    figstyle.save(fig, f"{prefix[0]}_joint", close=True)
    figstyle.save(plot_image_by_region(ga, mask, vmax, label), f"{prefix[1]}_image", close=True)
    fig = ga.plot_topomap(times=times, mask=mask, vlim=vlim, show=False, mask_params=dict(markersize=4))
    figstyle.save(fig, f"{prefix[2]}_topomap", close=True)
    print(f"  {name}: thời điểm topomap {[round(1000 * t) for t in times]} ms, vlim ±{vmax:.1f} µV, "
          f"{mask.any(axis=0).sum()} thời điểm có ý nghĩa")


def load_ica(sub):
    ica = mne.preprocessing.read_ica(QC_DIR / f"sub-{sub}-ica.fif")  # bản hiện hành, không dùng archive_pipeline_cu_01_03/
    ica.exclude = sorted(int(k) for k in as_dict(read_qc(sub)["ica_exclude"]))
    return ica


def fig3_components(ica):
    figs = ica.plot_components(picks=ica.exclude, show=False)
    for k, fig in enumerate(figs if isinstance(figs, list) else [figs]):
        figstyle.save(fig, "fig3_ica_components" + (f"_{k}" if k else ""), close=True)


def rebuild_filtered_raw(sub):
    """Raw đã lọc 0,1–40 Hz, nội suy cầu nối, TRƯỚC ICA: gọi đúng load_raw + prepare của 01_preprocess (không sửa file 01)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pre = importlib.import_module("01_preprocess")  # main() của 01 nằm trong if __name__ == "__main__"
    qc = read_qc(sub)
    raw = pre.load_raw(sub)
    pre.prepare(raw, list(as_dict(qc["bads"])), qc["bridged_pairs"])
    return raw, pre.stim_onsets(raw)


def fig2_and_properties(sub):
    """Hình 2 + hình properties IC; chỉ vẽ nếu bản tạo lại, sau các bước của clean(), khớp clean_raw đã lưu."""
    raw, onsets = rebuild_filtered_raw(sub)
    ica = load_ica(sub)
    after = raw.copy()
    ica.apply(after)

    # một lần chớp mắt điển hình trong lúc làm bài: biên độ Fp1 trung vị trong các lần phát hiện được
    blinks = (mne.preprocessing.find_eog_events(raw, ch_name="Fp1")[:, 0] - raw.first_samp) / raw.info["sfreq"]
    half = FIG2_WINDOW_S / 2
    blinks = blinks[(blinks > onsets.min() + half) & (blinks < onsets.max() - half)]
    amp = np.abs(raw.get_data(picks="Fp1")[0, raw.time_as_index(blinks)])
    t_blink = blinks[np.argsort(amp)[len(amp) // 2]]
    seg = dict(picks=FIG2_CHANNELS, tmin=t_blink - half, tmax=t_blink + half)
    before_seg, after_seg = 1e6 * raw.get_data(**seg), 1e6 * after.get_data(**seg)

    # kiểm chứng: lặp lại phần còn lại của clean() rồi so với file đã lưu (lưu dạng float32)
    after.interpolate_bads(reset_bads=True)
    after.set_eeg_reference("average", projection=False)
    after.resample(config.SFREQ_NEW)
    saved = mne.io.read_raw_fif(CLEAN_DIR / f"sub-{sub}_task-oddball_clean_raw.fif", preload=True)
    assert after.ch_names == saved.ch_names and after.n_times == saved.n_times
    max_diff = np.abs(after.get_data() - saved.get_data()).max()
    max_abs = np.abs(saved.get_data()).max()
    del after, saved
    ok = max_diff < 1e-9  # < 0,001 µV; sai số làm tròn float32 cỡ 1e-7 × biên độ
    print(f"  kiểm chứng tạo lại sub-{sub}: sai khác lớn nhất {1e6 * max_diff:.2e} µV "
          f"(biên độ lớn nhất {1e6 * max_abs:.0f} µV) -> {'KHỚP' if ok else 'KHÔNG KHỚP: bỏ Hình 2 và properties'}")
    if not ok:
        return

    t = np.linspace(-half, half, before_seg.shape[1])
    fig, axes = plt.subplots(len(FIG2_CHANNELS), 1, figsize=(8, 7), sharex=True)
    for ax, ch, b, a in zip(axes, FIG2_CHANNELS, before_seg, after_seg):
        ax.plot(t, b, color="0.75", lw=1, label="before ICA")
        ax.plot(t, a, color="k", lw=0.8, label="after ICA")
        ax.set_ylabel(f"{ch}\n(µV)")
    axes[0].legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel(f"Time relative to blink (s); blink at {t_blink:.1f} s of recording")
    fig.suptitle(f"sub-{sub}: 0.1–40 Hz, FCz reference, ICs {ica.exclude} removed")
    figstyle.save(fig, "fig2_ica_before_after", close=True)

    raw.filter(l_freq=1.0, h_freq=None)  # đúng dữ liệu lúc fit ICA trong 01 (propose)
    for ic, fig in zip(ica.exclude, ica.plot_properties(raw, picks=ica.exclude, show=False)):
        figstyle.save(fig, f"figS_ica_properties_IC{ic}", close=True)


def step_figures():
    windows = json.loads((RESULTS_DIR / "erp_windows.json").read_text(encoding="utf-8"))
    with plt.rc_context():
        figstyle.apply()
        print("Hình 4")
        figstyle.save(fig4_erp(windows), "fig4_erp_fz_pz", close=True)
        print("Hình 5")
        fig5_contrast("target - standard", ["fig5a_target_standard", "fig5b_target_standard", "fig5c_target_standard"])
        fig5_contrast("distractor - standard", ["figS_distractor_standard"] * 3)
        print("Hình 3")
        fig3_components(load_ica(FIG2_SUB))
        print("Hình 2 (+ properties IC)")
        fig2_and_properties(FIG2_SUB)
        plt.close("all")


# ---------------- Giai đoạn 6: Table S1 ----------------
def step_table():
    df = pd.read_csv(RESULTS_DIR / "qc_table.csv", dtype={"subject": str}, keep_default_na=False)
    exc = excluded()
    erp = ~df["subject"].isin(exc["all"])
    dis = erp & ~df["subject"].isin(exc["distractor_contrast"])
    table = pd.DataFrame({
        "Participant": "sub-" + df["subject"],
        "Interpolated channels": df["n_bad_ch"],
        "Bridged pairs": df["n_bridged"],
        "ICA components removed": df["n_ica_removed"],
        "Epochs rejected (%)": df["pct_rejected"].round(1),
        "Standard": df["n_standard"],
        "Target": df["n_target"],
        "Distractor": df["n_distractor"],
        "Included (ERP)": np.where(erp, "Yes", "No"),
        "Included (distractor contrast)": np.where(dis, "Yes", "No"),
    })
    numeric = table.columns[1:8]
    # cột Distractor tóm tắt trên người được giữ cho so sánh distractor; các cột khác trên người được giữ cho ERP
    kept = {col: table.loc[dis if col == "Distractor" else erp, col] for col in numeric}
    mean_row = {"Participant": "Mean ± SD", **{c: f"{s.mean():.1f} ± {s.std(ddof=1):.1f}" for c, s in kept.items()}}
    range_row = {"Participant": "Range", **{c: f"{s.min():g}–{s.max():g}" for c, s in kept.items()}}
    out = pd.concat([table.astype(str), pd.DataFrame([mean_row, range_row])], ignore_index=True).fillna("")
    out.to_csv(RESULTS_DIR / "table_S1_qc.csv", index=False)
    note = (f"\n\nMean ± SD and Range: participants included in ERP analyses (n = {erp.sum()}); "
            f"Distractor column: participants included in the distractor contrast (n = {dis.sum()}). "
            f"Epochs rejected (%): correct-trial epochs exceeding {REJECT['eeg'] * 1e6:.0f} µV peak-to-peak.\n")
    (RESULTS_DIR / "table_S1_qc.md").write_text("**Table S1.** Per-participant preprocessing quality control.\n\n"
                                               + out.to_markdown(index=False) + note, encoding="utf-8")

    k = {c: (s.mean(), s.std(ddof=1), s.min(), s.max()) for c, s in kept.items()}
    n_bridge = int((table.loc[erp, "Bridged pairs"] > 0).sum())
    sentence = (
        f"Bad channels were identified in {int((table.loc[erp, 'Interpolated channels'] > 0).sum())} of {erp.sum()} "
        f"participants and interpolated with spherical splines ({k['Interpolated channels'][0]:.1f} ± "
        f"{k['Interpolated channels'][1]:.1f} channels per participant, range {k['Interpolated channels'][2]:g}–"
        f"{k['Interpolated channels'][3]:g}). Gel bridges were detected in {n_bridge} participants "
        f"({k['Bridged pairs'][0]:.1f} ± {k['Bridged pairs'][1]:.1f} electrode pairs, range {k['Bridged pairs'][2]:g}–"
        f"{k['Bridged pairs'][3]:g}) and the bridged electrodes were interpolated using virtual channels. "
        f"ICA removed {k['ICA components removed'][0]:.1f} ± {k['ICA components removed'][1]:.1f} ocular components per "
        f"participant (range {k['ICA components removed'][2]:g}–{k['ICA components removed'][3]:g}). Epochs exceeding "
        f"{REJECT['eeg'] * 1e6:.0f} µV peak-to-peak were rejected ({k['Epochs rejected (%)'][0]:.1f} ± "
        f"{k['Epochs rejected (%)'][1]:.1f}% of correct-trial epochs, range {k['Epochs rejected (%)'][2]:g}–"
        f"{k['Epochs rejected (%)'][3]:g}%), leaving {k['Standard'][0]:.1f} ± {k['Standard'][1]:.1f} standard and "
        f"{k['Target'][0]:.1f} ± {k['Target'][1]:.1f} target epochs per participant (n = {erp.sum()}) and "
        f"{k['Distractor'][0]:.1f} ± {k['Distractor'][1]:.1f} distractor epochs (n = {dis.sum()}).")
    (RESULTS_DIR / "table_S1_methods_sentence.md").write_text(sentence + "\n", encoding="utf-8")
    print(out.to_markdown(index=False) + note)
    print(sentence)
    print(f"-> {RESULTS_DIR / 'table_S1_qc.csv'}, table_S1_qc.md, table_S1_methods_sentence.md")


# ---------------- Giai đoạn 8: số liệu cho bài viết ----------------
# vùng theo tiền tố tên kênh 10-05 (bỏ số, "h", "z"); chỉ để mô tả cụm
REGIONS = {
    "frontal": ["Fp", "AFp", "AF", "AFF", "F", "FFC"],
    "central": ["FC", "FCC", "C", "CCP", "CP"],
    "parietal": ["CPP", "P", "PPO"],
    "occipital": ["PO", "POO", "O", "OI"],
    "temporal": ["FT", "FFT", "FTT", "T", "TTP", "TP", "TPP"],
}


def region_of(ch):
    prefix = ch.rstrip("z").rstrip("h").rstrip("0123456789")
    return next(r for r, prefixes in REGIONS.items() if prefix in prefixes)


def p_text(p):
    return f"p {fmt_p(p)}" if p < .001 else f"p = {fmt_p(p)}"


def src_line(path, needle):
    """Số dòng đầu tiên chứa needle: để mỗi tham số tiền xử lý trỏ đúng chỗ trong code."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    hits = [i + 1 for i, line in enumerate(lines) if needle in line]
    assert hits, f"không thấy '{needle}' trong {path}"
    return f"{Path(path).relative_to(config.ROOT_DIR).as_posix()}:{hits[0]}"


def step_numbers():
    import mne_bids
    pre_path = Path(__file__).resolve().parent / "01_preprocess.py"
    this_path = Path(__file__).resolve()
    sys.path.insert(0, str(pre_path.parent))
    pre = importlib.import_module("01_preprocess")
    R = "results/"
    qc = pd.read_csv(RESULTS_DIR / "qc_table.csv", dtype={"subject": str}, keep_default_na=False)
    s1 = pd.read_csv(RESULTS_DIR / "table_S1_qc.csv", dtype=str).set_index("Participant")
    peaks = pd.read_csv(RESULTS_DIR / "erp_peaks_summary.csv")
    clusters = pd.read_csv(RESULTS_DIR / "erp_clusters.csv")
    windows = json.loads((RESULTS_DIR / "erp_windows.json").read_text(encoding="utf-8"))
    ga = read_ga()
    exc = excluded()
    out = ["# Numbers for writing (TV2)", "",
           "Generated by `python scripts/02_epochs_erp.py --step numbers`; do not edit by hand. "
           "Each number is followed by its source file.", ""]

    out += ["## 1. Participants", ""]
    for name in [*config.EVENT_ID, *CONTRASTS]:
        out.append(f"- n for {name}: {ga[name].nave} (derivatives/v1/erp/grand_average-ave.fif, nave)")
    out.append(f"- Excluded from all ERP analyses: {', '.join(exc['all']) or 'none'} (derivatives/v1/excluded.json)")
    for sub in exc["distractor_contrast"]:
        r = qc.set_index("subject").loc[sub]
        out.append(f"- sub-{sub} excluded from distractor − standard and P3a: correct distractor epochs left = "
                   f"{r['n_distractor']} (responded to distractors; QC flag: {r['flag']}) "
                   f"(derivatives/v1/excluded.json, {R}qc_table.csv)")
    kept_flagged = qc[(qc["flag"] != "") & ~qc["subject"].isin(exc["all"] + exc["distractor_contrast"])]
    for _, r in kept_flagged.iterrows():
        out.append(f"- sub-{r['subject']} flagged but retained: {r['flag']} ({R}qc_table.csv)")

    out += ["", f"## 2. Preprocessing QC (mean ± SD and range over included participants; {R}table_S1_qc.csv)", ""]
    for col in s1.columns[:7]:
        out.append(f"- {col}: {s1.loc['Mean ± SD', col]} (range {s1.loc['Range', col]})")
    out.append(f"- Methods sentence with these numbers: {R}table_S1_methods_sentence.md")

    out += ["", f"## 3. P3a / P3b (windows: {windows['method']}; {R}erp_windows.json, {R}erp_peaks_summary.csv)", "",
            "| Component | Channel | Window (ms) | n | Rare M ± SD (µV) | Standard M ± SD (µV) | Difference M ± SD (µV) "
            "| t(df) | p | Wilcoxon p | dz | FAL50 rare, median [IQR] (ms) | Peak latency rare, median [IQR] (ms) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in peaks.iterrows():
        out.append(
            f"| {r['component']} ({r['rare']}) | {r['channel']} | {r['window_ms'].replace('-', '–')} | {r['n']} "
            f"| {r['rare_mean_uV']:.2f} ± {r['rare_sd_uV']:.2f} | {r['standard_mean_uV']:.2f} ± {r['standard_sd_uV']:.2f} "
            f"| {r['diff_mean_uV']:.2f} ± {r['diff_sd_uV']:.2f} | t({r['df']}) = {r['t']:.2f} | {p_text(r['p_t'])} "
            f"| {p_text(r['p_wilcoxon'])} | {r['dz']:.2f} "
            f"| {r['rare_fal50_median_ms']:.0f} [{r['rare_fal50_q1_ms']:.0f}–{r['rare_fal50_q3_ms']:.0f}] "
            f"| {r['rare_peak_lat_median_ms']:.0f} [{r['rare_peak_lat_q1_ms']:.0f}–{r['rare_peak_lat_q3_ms']:.0f}] |")
    for _, r in peaks.iterrows():
        out.append(f"\n- {r['component']}: mean amplitude in {r['window_ms']} ms; peak and FAL50 searched in "
                   f"{r['wide_window_ms']} ms; {r['n_no_positive_peak']} of {r['n']} participants had no positive "
                   f"{r['rare']} peak and are left out of the latency medians.")

    out += ["", f"## 4. Significant clusters ({R}erp_clusters.csv, {R}cluster_*.npz)", "",
            "Spatio-temporal cluster permutation test, one-sample on participants' difference waves, 0–800 ms, "
            f"cluster-forming threshold two-tailed p < {CLUSTER_ALPHA}, {int(clusters['n_perm'].iloc[0])} permutations, "
            f"seed {config.SEED}. Times rounded to 10 ms. Channel and region counts are the union over the whole "
            "cluster duration, not channels significant at every time point. "
            "Do not read cluster onset or extent as effect onset or location.",
            "", "| Contrast | n | Sign | Time (ms) | Channels | Regions (channels in cluster / channels in region) | p |",
            "|---|---|---|---|---|---|---|"]
    all_ch = list(np.load(RESULTS_DIR / f"cluster_{slug('target - standard')}.npz")["ch_names"])
    region_total = pd.Series([region_of(ch) for ch in all_ch]).value_counts()
    for _, r in clusters.iterrows():
        counts = pd.Series([region_of(ch) for ch in r["channels"].split()]).value_counts()
        regions = ", ".join(f"{reg} {counts.get(reg, 0)}/{region_total[reg]}" for reg in REGIONS if counts.get(reg, 0))
        out.append(f"| {r['contrast'].replace(' - ', ' − ')} | {r['n']} | {r['sign']} "
                   f"| {round(r['t_start_ms'], -1):.0f}–{round(r['t_end_ms'], -1):.0f} | {r['n_channels']} | {regions} "
                   f"| {p_text(r['p'])} |")
    out.append("\nRegion = channel-name prefix (" + "; ".join(f"{k}: {', '.join(v)}" for k, v in REGIONS.items()) + ").")

    out += ["", "## 5. Software", ""]
    out += [f"- Python {sys.version.split()[0]}, MNE-Python {mne.__version__}, mne-bids {mne_bids.__version__}, "
            f"NumPy {np.__version__}, SciPy {__import__('scipy').__version__}, pandas {pd.__version__} "
            "(environment that ran this step)"]

    ica = mne.preprocessing.read_ica(QC_DIR / f"sub-{FIG2_SUB}-ica.fif")
    out += ["", "## 6. Preprocessing parameters", ""]
    out += [
        f"- Band-pass {config.L_FREQ}–{config.H_FREQ} Hz (MNE default FIR) on continuous data "
        f"(src/eegoddball/config.py L_FREQ/H_FREQ; {src_line(pre_path, 'raw.filter(config.L_FREQ')})",
        f"- Recording cropped at last stimulus + {pre.TAIL_S} s before filtering ({src_line(pre_path, 'TAIL_S =')})",
        f"- Montage: 10-05 positions ({src_line(pre_path, 'def make_montage')})",
        f"- Bad-channel proposal (task periods only): flat if SD < {pre.FLAT_UV} µV; noisy if 8–40 Hz SD ≥ "
        f"{pre.NOISY_RATIO}× the 6 nearest neighbours and median correlation with the 4 nearest < {pre.NOISY_R}; "
        f"bursts if ≥ {pre.BURST_PCT}% of 1-s windows > 3× neighbours and ≥ {pre.BURST_PCT}% of epochs > "
        f"{pre.REJECT_UV} µV; shorted if a bridged group spans > {pre.SHORT_EXTENT_MM} mm. "
        f"Proposals reviewed and approved per participant ({src_line(pre_path, 'rules = {')}; "
        "derivatives/v1/qc/sub-XX_qc.json approved_by)",
        f"- Bad channels interpolated with spherical splines after ICA ({src_line(pre_path, 'raw.interpolate_bads')})",
        f"- Gel bridges: mne.preprocessing.compute_bridged_electrodes; interpolated with virtual channels, "
        f"groups up to {pre.BRIDGE_GROUP_LIMIT} channels ({src_line(pre_path, 'compute_bridged_electrodes')}; "
        f"{src_line(pre_path, 'bad_limit=BRIDGE_GROUP_LIMIT')})",
        f"- ICA: {ica.method}, {pre.ICA_COMPONENTS} components, random_state = {config.SEED}, fitted on a 1 Hz "
        f"high-passed copy, decim = 3 ({src_line(pre_path, 'mne.preprocessing.ICA(')}; "
        f"{src_line(pre_path, 'filter(l_freq=1.0')}; {src_line(pre_path, 'decim=3')}; "
        f"derivatives/v1/qc/sub-{FIG2_SUB}-ica.fif method)",
        f"- Ocular IC candidates: ica.find_bads_eog with {list(pre.EOG_FALLBACK)} (fallback "
        f"{list(pre.EOG_FALLBACK.values())} if bad) and a bipolar HEOG ({pre.HEOG_PAIRS[0][0]}−{pre.HEOG_PAIRS[0][1]}, "
        f"else {pre.HEOG_PAIRS[1][0]}−{pre.HEOG_PAIRS[1][1]}); final selection by visual review, reason per IC in "
        f"sub-XX_qc.json ica_exclude ({src_line(pre_path, 'find_bads_eog(raw_ica')})",
        f"- Re-reference: average ({src_line(pre_path, 'set_eeg_reference')}); resampled to {config.SFREQ_NEW} Hz "
        f"(config.SFREQ_NEW; {src_line(pre_path, 'raw.resample(config.SFREQ_NEW')})",
        f"- Epochs {config.TMIN} to {config.TMAX} s, baseline {config.TMIN} to 0 s; correct trials only (target with a "
        f"response, standard/distractor without); rejection > {REJECT['eeg'] * 1e6:.0f} µV peak-to-peak "
        f"(config.TMIN/TMAX; {src_line(this_path, 'baseline=(config.TMIN, 0)')}; {src_line(this_path, 'REJECT = ')})",
    ]

    path = RESULTS_DIR / "numbers_for_writing.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n-> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", choices=["epochs", "qc", "evokeds", "cluster", "peaks", "figures", "table", "numbers"],
                        default="epochs")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--n-perm", type=int, default=5000)
    args = parser.parse_args()

    # tên kênh và thông báo có tiếng Việt: tránh lỗi mã hóa khi stdout bị chuyển hướng trên Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    mne.set_log_level("error")
    if args.step == "qc":
        step_qc(args.subjects or config.SUBJECTS)
        return
    steps = {"evokeds": step_evokeds, "cluster": lambda: step_cluster(args.n_perm), "peaks": step_peaks,
             "figures": step_figures, "table": step_table, "numbers": step_numbers}
    if args.step in steps:
        RESULTS_DIR.mkdir(exist_ok=True)
        REPORTS_DIR.mkdir(exist_ok=True)
        steps[args.step]()
        return

    failed = []
    for sub in args.subjects or config.SAMPLE_SUBJECTS:
        try:
            process(sub)
        except Exception:  # chạy loạt: ghi lại người lỗi, không bỏ dở những người còn lại
            traceback.print_exc(file=sys.stdout)
            failed.append(sub)
    if failed:
        sys.exit(f"LỖI ở {len(failed)} người: {failed}")


if __name__ == "__main__":
    main()
