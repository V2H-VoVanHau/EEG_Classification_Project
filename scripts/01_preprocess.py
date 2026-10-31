"""Tiền xử lý v1 (TV2): raw BIDS -> raw sạch trung gian + bản ghi QC.

Trạng thái mỗi người đọc từ derivatives/v1/qc/sub-XX_qc.json:
- Chưa có QC hoặc approved=false: chỉ ĐỀ XUẤT (kênh hỏng, cầu nối, thành phần ICA, hình), không loại gì.
  Sửa "bads" trong QC rồi chạy lại nếu cần fit lại ICA; khi đã chốt thì điền "ica_exclude" và đặt approved=true.
- approved=true: áp dụng đúng bads và ica_exclude trong QC, lưu raw sạch.

Dùng: python scripts/01_preprocess.py [--subjects 01 02]   (mặc định SAMPLE_SUBJECTS)
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids
from scipy.sparse.csgraph import connected_components

from eegoddball import config

V1_DIR = Path(config.DERIVATIVES_DIR) / "v1"
QC_DIR = V1_DIR / "qc"
CLEAN_DIR = V1_DIR / "clean"

N_CHANNELS = 127
ICA_COMPONENTS = 30
# giữ sau kích thích cuối: hơn nửa độ dài lọc 0.1 Hz (33 s) để phần đuôi (tháo mũ, tới 75000 µV) không vọng vào epoch
TAIL_S = 20
FLAT_UV = 0.5
# tiêu chí kênh hỏng, đã đối chiếu với quyết định duyệt tay của sub-01..03
NOISY_RATIO = 2
NOISY_R = 0.5
BURST_PCT = 15
REJECT_UV = 150  # cùng ngưỡng reject ở 02_epochs_erp.py
# nhóm cầu nối: mô phỏng trên người không có cầu nối cho thấy kênh ảo tốt hơn spline đến 11 kênh, đường kính đến ~100 mm
BRIDGE_GROUP_LIMIT = 11
# nhóm rộng hơn (sub-28: 16 kênh giống hệt nhau, 199 mm) là chập kênh: trung bình của chúng không ứng với vị trí nào
SHORT_EXTENT_MM = 120
# kênh thay thế khi kênh trán chính là bad (cũng là kênh trán, ngay cạnh)
EOG_FALLBACK = {"Fp1": "AFp1", "Fp2": "AFp2"}
HEOG_PAIRS = [("F7", "F8"), ("AF7", "AF8")]


def qc_path(sub):
    return QC_DIR / f"sub-{sub}_qc.json"


def ica_path(sub):
    return QC_DIR / f"sub-{sub}-ica.fif"


def clean_raw_path(sub):
    return CLEAN_DIR / f"sub-{sub}_task-oddball_clean_raw.fif"


def write_qc(sub, qc):
    qc_path(sub).write_text(json.dumps(qc, ensure_ascii=False, indent=2), encoding="utf-8")


def make_montage():
    # MNE >= 1.13 đổi tên standard_1005 -> colin27_1005 (cùng tọa độ); giữ tương thích bản cũ
    name = "colin27_1005" if "colin27_1005" in mne.channels.get_builtin_montages() else "standard_1005"
    pos = mne.channels.make_standard_montage(name).get_positions()
    # O9/O10 là tên khác của I1/I2 trong hệ 10-05 (Oostenveld & Praamstra 2001): mượn tọa độ, giữ tên kênh
    pos["ch_pos"]["O9"], pos["ch_pos"]["O10"] = pos["ch_pos"]["I1"], pos["ch_pos"]["I2"]
    return mne.channels.make_dig_montage(**pos)


def stim_onsets(raw):
    ann = raw.annotations
    return ann.onset[ann.description == "stimulus"] - (raw.first_time if ann.orig_time else 0)


def load_raw(sub):
    bids_path = BIDSPath(subject=sub, task="oddball", datatype="eeg", root=config.DATA_DIR)
    # sub-13, 41, 42: .vhdr khác thứ tự channels.tsv; "reorder" theo tên, đưa về cùng thứ tự với mọi người
    raw = read_raw_bids(bids_path, on_ch_mismatch="reorder", verbose="error").load_data()
    # cắt trước khi lọc, nếu không nhiễu phần đuôi sẽ vọng ngược vào dữ liệu làm bài
    raw.crop(tmax=min(stim_onsets(raw).max() + TAIL_S, raw.times[-1]))
    # dataset không có electrodes.tsv: đổi tên 2 kênh tai cho khớp montage chuẩn
    raw.rename_channels({"TP9 LEFT EAR": "TP9", "TP10 RIGHT EAR": "TP10"})
    raw.set_montage(make_montage())
    assert len(raw.ch_names) == N_CHANNELS, f"sub-{sub}: có {len(raw.ch_names)} kênh"
    return raw


def channel_stats(raw):
    """Chỉ số từng kênh CHỈ lúc làm bài, theo cửa sổ và so với lân cận: nhiễu thoáng qua lúc nghỉ không chi phối,
    vùng nhiều alpha hay nhiều mắt không bị báo nhầm vì lân cận cũng lớn."""
    sf = int(config.SFREQ_NEW)
    copy = raw.copy().filter(config.L_FREQ, config.H_FREQ).resample(sf)
    onsets = (stim_onsets(raw) * sf).astype(int)
    mask = np.zeros(copy.n_times, bool)
    for onset in onsets:
        mask[max(onset - sf, 0): onset + int(1.5 * sf)] = True

    def window_std(x):
        n = x.shape[1] // sf
        in_task = mask[:n * sf].reshape(n, sf).mean(axis=1) > 0.5
        return x[:, :n * sf].reshape(len(x), n, sf).std(axis=2)[:, in_task]

    def median_r(x, neighbours, win_s=10):
        rs = []
        for k in range(x.shape[1] // (sf * win_s)):
            start, stop = k * sf * win_s, (k + 1) * sf * win_s
            if mask[start:stop].mean() >= 0.9:
                corr = np.corrcoef(x[:, start:stop])
                rs.append([corr[i, neighbours[i]].mean() for i in range(len(corr))])
        return np.nanmedian(rs, axis=0)

    pos = copy.get_montage().get_positions()["ch_pos"]
    xyz = np.array([pos[ch] for ch in copy.ch_names])
    dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    nearest = np.argsort(dist, axis=1)

    data = copy.get_data() * 1e6
    broad = mne.filter.filter_data(data, sf, 1, 40)
    level = np.median(window_std(broad), axis=1)
    # average reference chỉ trên kênh không quá lớn, tham chiếu gốc FCz làm kênh lân cận nhỏ giả tạo
    ref_chs = level < 3 * np.median(level)
    broad -= broad[ref_chs].mean(axis=0)
    beta = mne.filter.filter_data(broad, sf, 8, 40)

    win_broad, win_beta = window_std(broad), window_std(beta)
    std_broad, std_beta = np.median(win_broad, axis=1), np.median(win_beta, axis=1)

    avg = data - data[ref_chs].mean(axis=0)
    tmin, tmax = int(-config.TMIN * sf), int(config.TMAX * sf) + 1
    trials = np.stack([avg[:, s - tmin: s + tmax] for s in onsets if s >= tmin and s + tmax <= avg.shape[1]])

    stats = pd.DataFrame({
        "std_uV": std_broad,
        "beta_vs_neighbours": std_beta / np.median(std_beta[nearest[:, :6]], axis=1),
        "beta_r_neighbours": median_r(beta, nearest[:, :4]),
        "burst_pct": 100 * (win_broad > 3 * np.median(win_broad[nearest[:, :6]], axis=1)).mean(axis=1),
        "over_150uV_pct": 100 * (np.ptp(trials, axis=2) > REJECT_UV).mean(axis=0),
    }, index=copy.ch_names).round(2)

    rules = {
        "phẳng": stats["std_uV"] < FLAT_UV,
        "nhiễu riêng 8–40 Hz": (stats["beta_vs_neighbours"] >= NOISY_RATIO) & (stats["beta_r_neighbours"] < NOISY_R),
        "từng đoạn rất lớn": (stats["burst_pct"] >= BURST_PCT) & (stats["over_150uV_pct"] >= BURST_PCT),
    }
    stats["reason"] = [", ".join(r for r, hit in rules.items() if hit[ch]) for ch in copy.ch_names]
    return stats


def bridged_groups(raw, bridged_idx):
    nodes = sorted({i for pair in bridged_idx for i in pair})
    if not nodes:
        return []
    graph = np.zeros((len(nodes), len(nodes)))
    for i, j in bridged_idx:
        graph[nodes.index(i), nodes.index(j)] = 1
    _, labels = connected_components(graph, directed=False)
    return [[raw.ch_names[n] for n, lab in zip(nodes, labels) if lab == k] for k in sorted(set(labels))]


def group_extent_mm(raw, group):
    pos = raw.get_montage().get_positions()["ch_pos"]
    xyz = np.array([pos[ch] for ch in group])
    return 1000 * np.linalg.norm(xyz[:, None] - xyz[None], axis=2).max()


def prepare(raw, bads, bridged_pairs):
    """Lọc, gán bads, nội suy cầu nối: dùng chung cho đề xuất và làm sạch để ICA khớp dữ liệu đã duyệt."""
    raw.filter(config.L_FREQ, config.H_FREQ)
    raw.info["bads"] = list(bads)
    if bridged_pairs:
        # hàm MNE tạm bỏ info["bads"] nên kênh hỏng lọt vào spline, làm nhiễu lan ra cả đầu: chỉ đưa kênh tốt vào
        good = raw.copy().drop_channels(list(bads))
        names = good.ch_names
        mne.preprocessing.interpolate_bridged_electrodes(
            good, [(names.index(a), names.index(b)) for a, b in bridged_pairs], bad_limit=BRIDGE_GROUP_LIMIT)
        raw[names] = good.get_data()
    assert raw.info["bads"] == list(bads)


def save_figures(sub, raw_ica, ica, picks, psd):
    fig_dir = QC_DIR / f"sub-{sub}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    psd.plot(average=False, amplitude=False, show=False).savefig(fig_dir / "psd.png", dpi=90)
    for page, fig in enumerate(ica.plot_components(show=False)):
        fig.savefig(fig_dir / f"ica_components_{page}.png", dpi=90)
    for ic, fig in zip(picks, ica.plot_properties(raw_ica, picks=picks, show=False)):
        fig.savefig(fig_dir / f"ica_properties_{ic:02d}.png", dpi=90)
    with mne.viz.use_browser_backend("matplotlib"):
        fig = ica.plot_sources(raw_ica, picks=picks, start=300, stop=320, show=False, show_scrollbars=False)
    fig.savefig(fig_dir / "ica_sources_300_320s.png", dpi=90)
    plt.close("all")


def propose(sub, qc):
    """Chỉ đề xuất, không loại gì; ICA được lưu để bước làm sạch dùng đúng thành phần đã xem."""
    raw = load_raw(sub)
    stats = channel_stats(raw)
    psd = raw.compute_psd(fmax=100)
    bridged_idx, _ = mne.preprocessing.compute_bridged_electrodes(raw)
    bridged_pairs = sorted((raw.ch_names[i], raw.ch_names[j]) for i, j in bridged_idx)
    groups = bridged_groups(raw, bridged_idx)
    shorted = [ch for group in groups if group_extent_mm(raw, group) > SHORT_EXTENT_MM for ch in group]

    flagged = stats[stats["reason"] != ""]
    suggested = {ch: stats.loc[ch, "reason"] for ch in flagged.index}
    suggested.update({ch: "chập kênh: giống hệt nhiều kênh ở xa" for ch in shorted})
    bads = qc["bads"] if qc else suggested
    # kênh hỏng được nội suy từ kênh tốt, không tính vào cặp cầu nối nữa
    bridged_pairs = [pair for pair in bridged_pairs if not set(pair) & set(bads)]

    eog_chs = [EOG_FALLBACK[ch] if ch in bads else ch for ch in EOG_FALLBACK]
    heog_pair = next((pair for pair in HEOG_PAIRS if not set(pair) & set(bads)), None)
    if heog_pair is None:
        raise ValueError(f"sub-{sub}: mọi cặp trong HEOG_PAIRS đều có kênh bad {list(bads)}; cần thêm cặp khác.")

    prepare(raw, bads, bridged_pairs)
    raw_ica = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=ICA_COMPONENTS, random_state=config.SEED, max_iter="auto")
    ica.fit(raw_ica, decim=3)
    ica.save(ica_path(sub), overwrite=True)

    eog_idx, eog_scores = ica.find_bads_eog(raw_ica, ch_name=eog_chs)
    raw_heog = mne.set_bipolar_reference(raw_ica, *heog_pair, ch_name="HEOG", drop_refs=False, copy=True)
    heog_idx, heog_scores = ica.find_bads_eog(raw_heog, ch_name="HEOG")
    del raw_heog

    scores = pd.DataFrame(np.vstack([*eog_scores, heog_scores]).T, columns=[*eog_chs, "HEOG"]).round(3)
    candidates = sorted({int(k) for k in [*eog_idx, *heog_idx]})
    top = [int(k) for k in scores.abs().max(axis=1).nlargest(6).index]
    picks = sorted({*candidates, *top[:3]})
    save_figures(sub, raw_ica, ica, picks, psd)

    qc = {
        "subject": sub,
        "approved": False,
        "bads": bads,
        "bridged_pairs": [list(pair) for pair in bridged_pairs],
        "eog_proxies": eog_chs,
        "heog_pair": list(heog_pair),
        "ica_exclude": {},
        "proposal": {
            "suggested_bads": suggested,
            "flagged_channels": flagged.to_dict(orient="index"),
            "bridged_groups": groups,
            "ica_candidates": candidates,
            "ica_scores_top": {k: scores.loc[k].to_dict() for k in top},
            "ica_explained_var_pct": {
                k: round(100 * ica.get_explained_variance_ratio(raw_ica, components=[k])["eeg"], 1) for k in top},
        },
    }
    write_qc(sub, qc)
    print(f"sub-{sub}: ĐỀ XUẤT bads={list(bads)}, {len(bridged_pairs)} cặp cầu nối, "
          f"ICA gợi ý {candidates} -> chờ duyệt: {qc_path(sub)}")


def clean(sub, qc):
    """Áp dụng đúng bản ghi đã duyệt: ICA -> nội suy bads -> average reference -> hạ mẫu."""
    bads = qc["bads"]
    raw = load_raw(sub)
    prepare(raw, bads, qc["bridged_pairs"])

    ica = mne.preprocessing.read_ica(ica_path(sub))
    if set(ica.ch_names) != set(raw.ch_names) - set(bads):
        raise ValueError(f"sub-{sub}: bads trong QC khác lúc fit ICA. Đặt approved=false rồi chạy lại để fit lại.")
    ica.exclude = [int(k) for k in qc["ica_exclude"]]
    ica.apply(raw)

    raw.interpolate_bads(reset_bads=True)
    raw.set_eeg_reference("average", projection=False)
    raw.resample(config.SFREQ_NEW)

    assert raw.info["sfreq"] == config.SFREQ_NEW
    assert len(raw.ch_names) == N_CHANNELS and raw.info["bads"] == []
    assert raw.info["custom_ref_applied"]
    assert (raw.info["highpass"], raw.info["lowpass"]) == (config.L_FREQ, config.H_FREQ)

    raw.save(clean_raw_path(sub), overwrite=True)
    print(f"sub-{sub}: đã làm sạch (bads={list(bads)}, loại IC {ica.exclude}): {clean_raw_path(sub)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", default=config.SAMPLE_SUBJECTS)
    args = parser.parse_args()

    # tên kênh và thông báo có tiếng Việt: tránh lỗi mã hóa khi stdout bị chuyển hướng trên Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    mne.set_log_level("error")
    mne.viz.set_browser_backend("matplotlib")
    QC_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for sub in args.subjects:
        qc = json.loads(qc_path(sub).read_text(encoding="utf-8")) if qc_path(sub).exists() else None
        try:
            if qc and qc["approved"]:
                clean(sub, qc)
            else:
                propose(sub, qc)
        except Exception:  # chạy loạt: ghi lại người lỗi, không bỏ dở những người còn lại
            traceback.print_exc(file=sys.stdout)
            failed.append(sub)
        plt.close("all")
    if failed:
        sys.exit(f"LỖI ở {len(failed)} người: {failed}")


if __name__ == "__main__":
    main()
