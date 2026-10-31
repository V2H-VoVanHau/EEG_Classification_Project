"""Cắt epoch v1 (TV2): raw sạch -> derivatives/v1/sub-XX_task-oddball-epo.fif + self-check + số liệu QC.

Cần chạy scripts/01_preprocess.py và duyệt QC trước.

Dùng: python scripts/02_epochs_erp.py [--subjects 01 02]   (mặc định SAMPLE_SUBJECTS)
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath

from eegoddball import config

sys.path.insert(0, config.ROOT_DIR)  # create_v0_data.py nằm ở gốc repo
import create_v0_data  # noqa: E402

V1_DIR = Path(config.DERIVATIVES_DIR) / "v1"
QC_DIR = V1_DIR / "qc"
CLEAN_DIR = V1_DIR / "clean"

RESPONSE_CODE = "S 1"
REJECT = dict(eeg=150e-6)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", nargs="+", default=config.SAMPLE_SUBJECTS)
    args = parser.parse_args()

    # tên kênh và thông báo có tiếng Việt: tránh lỗi mã hóa khi stdout bị chuyển hướng trên Windows
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    mne.set_log_level("error")
    failed = []
    for sub in args.subjects:
        try:
            process(sub)
        except Exception:  # chạy loạt: ghi lại người lỗi, không bỏ dở những người còn lại
            traceback.print_exc(file=sys.stdout)
            failed.append(sub)
    if failed:
        sys.exit(f"LỖI ở {len(failed)} người: {failed}")


if __name__ == "__main__":
    main()
