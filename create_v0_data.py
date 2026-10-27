from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids

from eegoddball.config import (
    DATA_DIR,
    DERIVATIVES_DIR,
    SAMPLE_SUBJECTS,
    EVENT_ID,
    STIMULUS_CODES,
    EXPECTED_TRIALS,
    SFREQ_NEW,
    L_FREQ,
    H_FREQ,
    TMIN,
    TMAX,
)


def read_stimulus_metadata(events_path):
    """Đọc kích thích và gắn trial_id trước khi cắt epoch."""
    events_df = pd.read_csv(events_path, sep="\t")

    required_columns = {"onset", "event_type"}
    missing_columns = required_columns - set(events_df.columns)

    if missing_columns:
        raise ValueError(
            f"{events_path}: thiếu cột {sorted(missing_columns)}"
        )

    # Chuẩn hóa khoảng trắng: "S  5" -> "S 5"
    event_types = (
        events_df["event_type"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Chỉ lấy ba loại kích thích, không lấy sự kiện bấm phím
    is_stimulus = event_types.isin(STIMULUS_CODES)

    metadata = events_df.loc[is_stimulus].copy()

    # Lưu vị trí dòng gốc để thuận tiện đối chiếu events.tsv
    metadata["source_row"] = np.flatnonzero(
        is_stimulus.to_numpy()
    )

    metadata["condition"] = (
        event_types.loc[is_stimulus]
        .map(STIMULUS_CODES)
        .to_numpy()
    )

    # Giữ nguyên thứ tự kích thích trong events.tsv
    metadata = metadata.reset_index(drop=True)

    if len(metadata) != EXPECTED_TRIALS:
        raise ValueError(
            f"{events_path}: tìm thấy {len(metadata)} kích thích, "
            f"yêu cầu {EXPECTED_TRIALS}. "
            "Hãy kiểm tra event_type trước khi tiếp tục."
        )

    # Mã cố định của kích thích gốc: 0..659
    # Không đánh số lại mã này sau khi loại epoch ở v1
    metadata["trial_id"] = np.arange(
        len(metadata),
        dtype=int,
    )

    metadata["event_code"] = (
        metadata["condition"]
        .map(EVENT_ID)
        .astype(int)
    )

    metadata["onset"] = pd.to_numeric(
        metadata["onset"],
        errors="raise",
    )

    onsets = metadata["onset"].to_numpy(dtype=float)

    if (
        not np.isfinite(onsets).all()
        or (onsets < 0).any()
        or (np.diff(onsets) < 0).any()
    ):
        raise ValueError(
            f"{events_path}: onset phải hữu hạn, "
            "không âm và theo thứ tự thời gian."
        )

    return metadata


def create_v0_data():
    # Script này luôn tạo bản nháp v0
    v0_dir = Path(DERIVATIVES_DIR) / "v0"
    v0_dir.mkdir(parents=True, exist_ok=True)

    for sub in SAMPLE_SUBJECTS:
        print(f"\nĐang xử lý sub-{sub}...")

        bids_path = BIDSPath(
            subject=sub,
            task="oddball",
            datatype="eeg",
            root=DATA_DIR,
        )

        raw = read_raw_bids(
            bids_path=bids_path,
            verbose=False,
        )
        raw.load_data()

        # Đọc trực tiếp events.tsv thay vì lấy nhãn từ annotation
        events_path = (
            bids_path.copy()
            .update(
                suffix="events",
                extension=".tsv",
            )
            .fpath
        )

        if not events_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {events_path}. "
                "Nếu dữ liệu có session/run, cần khai báo "
                "session/run tương ứng trong BIDSPath."
            )

        metadata = read_stimulus_metadata(events_path)

        # Lọc và hạ mẫu trước khi chuyển onset sang chỉ số mẫu
        raw.filter(
            l_freq=L_FREQ,
            h_freq=H_FREQ,
            phase="zero",
            verbose=False,
        )

        raw.resample(
            SFREQ_NEW,
            verbose=False,
        )

        # onset tính bằng giây từ đầu bản ghi.
        # Chuyển sang chỉ số mẫu ở tần số lấy mẫu mới.
        event_samples = (
            raw.time_as_index(
                metadata["onset"].to_numpy(dtype=float),
                use_rounding=True,
            )
            + raw.first_samp
        )

        if (np.diff(event_samples) <= 0).any():
            raise ValueError(
                f"sub-{sub}: có kích thích trùng mẫu hoặc "
                "không đúng thứ tự sau khi hạ mẫu."
            )

        events = np.column_stack(
            [
                event_samples,
                np.zeros(len(metadata), dtype=int),
                metadata["event_code"].to_numpy(dtype=int),
            ]
        ).astype(int)

        epochs = mne.Epochs(
            raw,
            events,
            event_id=EVENT_ID,
            metadata=metadata,
            tmin=TMIN,
            tmax=TMAX,
            baseline=(None, 0),
            preload=True,
            reject=None,
            flat=None,
            # v0 chỉ là bản nháp, chưa loại nhiễu theo annotation
            reject_by_annotation=False,
            event_repeated="error",
            verbose=False,
        )

        # Không âm thầm lưu v0 thiếu kích thích
        if len(epochs) != EXPECTED_TRIALS:
            dropped = [
                (int(trial_id), reasons)
                for trial_id, reasons in zip(
                    metadata["trial_id"],
                    epochs.drop_log,
                )
                if reasons
            ]

            raise ValueError(
                f"sub-{sub}: chỉ còn {len(epochs)}/"
                f"{EXPECTED_TRIALS} epoch. "
                f"Các trial bị loại: {dropped}"
            )

        if not np.array_equal(
            epochs.events[:, 2],
            epochs.metadata["event_code"].to_numpy(dtype=int),
        ):
            raise ValueError(
                f"sub-{sub}: nhãn events không khớp metadata."
            )

        output_path = (
            v0_dir / f"sub-{sub}_task-oddball-epo.fif"
        )

        epochs.save(
            output_path,
            overwrite=True,
            verbose=False,
        )

        print(f"Đã lưu: {output_path}")
        print(f"Tổng số epoch: {len(epochs)}")
        print(f"event_id: {epochs.event_id}")
        print(
            "Số epoch từng lớp:",
            epochs.metadata["condition"]
            .value_counts()
            .to_dict(),
        )
        print(
            "trial_id:",
            int(epochs.metadata["trial_id"].min()),
            "đến",
            int(epochs.metadata["trial_id"].max()),
        )


if __name__ == "__main__":
    create_v0_data()