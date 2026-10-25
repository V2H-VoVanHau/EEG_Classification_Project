import os
import pandas as pd
import mne
from mne_bids import BIDSPath, read_raw_bids
from src.eegoddball.config import DATA_DIR, DERIVATIVES_DIR, SAMPLE_SUBJECTS, EVENT_ID, SFREQ_NEW, L_FREQ, H_FREQ, TMIN, TMAX

def create_v0_data():
    v0_dir = os.path.join(DERIVATIVES_DIR, "v0")
    os.makedirs(v0_dir, exist_ok=True)
    
    for sub in SAMPLE_SUBJECTS:
        print(f"\n đang xử lý dữ liệu nháp cho v0 cho sub-{sub}")
        #đọc dữ liệu BIDS
        bp = BIDSPath(subject=sub, task="oddball", datatype="eeg", root=DATA_DIR)
        raw = read_raw_bids(bids_path=bp, verbose=False)
        raw.load_data()
        #lọc dải (0.1 - 40 Hz) và giảm tần số mẫu xuống 250 Hz
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, phase="zero", verbose=False)
        raw.resample(SFREQ_NEW, verbose=False)
        
        #đọc sự kiện
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        #cắt epoch và gắn metadata chứ trial_id
        epochs = mne.Epochs(
            raw, events, event_id=event_id_map,
            tmin=TMIN, tmax=TMAX, baseline=(None, 0),
            preload=True, verbose=False
        )
        #lưu file vào derivatives/v0/
        out_path = os.path.join(v0_dir, f"sub-{sub}_task-oddball-epo.fif")
        epochs.save(out_path, overwrite=True, verbose=False)
        print(f" đã tạo thành công : {out_path} (tổng số epoch: {len(epochs)})")
if __name__ == "__main__":
    create_v0_data()
