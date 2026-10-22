import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import mne
from src.eegoddball.config import DERIVATIVES_DIR, SPLIT_DIR, SUBJECTS, SAMPLE_SUBJECTS, SEED

def create_folds():
    os.makedirs(SPLIT_DIR, exist_ok=True)
    folds_data = {"within_subject": {}, "cross_subject": {}}
    
    
    subjects_arr = np.array(SAMPLE_SUBJECTS)
    n_splits_cross = min(5, len(SAMPLE_SUBJECTS))
    kf_cross = KFold(n_splits=n_splits_cross, shuffle=True, random_state=SEED)
    
    for fold_idx, (_, test_indices) in enumerate(kf_cross.split(subjects_arr)):
        test_subs= subjects_arr[test_indices].tolist()
        folds_data["cross_subject"][f"fold_{fold_idx}"] = {"test_subjects": test_subs}
    
    
    skf_within = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    v0_dir = os.path.join(DERIVATIVES_DIR, "v0")
    
    for sub in SAMPLE_SUBJECTS:
        fif_path = os.path.join(v0_dir, f"sub-{sub}-epo.fif")
        if not os.path.exists(fif_path): 
            fif_path = os.path.join(v0_dir, f"sub-{sub}_task-oddball-epo.fif")
        if not os.path.exists(fif_path):
            print(f"chưa thấy file {fif-path}, bỏ qua sub-{sub}")
            continue
        
        epochs = mne.read_epochs(fif_path, preload=False, verbose=False)
        trial_ids = np.arange(len(epochs))
        labels = epochs.events[:, -1]
        
        sub_folds = {}
    for fold_idx, (_, test_indices) in enumerate (skf_within.split(trial_ids, labels)):
        sub_folds[f"fold_{fold_idx}"] = {"test_trail_ids" : trial_ids[test_indices].tolist()}
        folds_data["within_subject"][sub] = sub_folds
    
    
    out_json = os.path.join(SPLIT_DIR, "fold.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(folds_data, f, indent=2)
    print(f"đã để test tại: {out_json}")

if __name__ == "__main__":
    create_folds()