"""Kiểm thử tự động cho các thành phần giải thích mô hình & Confound của TV5."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import mne
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def _load_script_module(module_name: str, rel_path: str):
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(module_name, root / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eegnet_tv4_compatibility_and_integrated_gradients_shapes(tmp_path):
    explain_mod = _load_script_module("explain_05", "scripts/05_explain.py")
    n_ch, n_times = 12, 103
    arch = dict(F1=8, D=2, F2=16, k1=64, drop=0.5)
    model = explain_mod.EEGNet(n_ch=n_ch, n_times=n_times, n_classes=2, **arch)
    x = torch.randn(4, n_ch, n_times)
    logits = model(x)
    assert logits.shape == (4, 2)

    # Kiểm tra đóng gói và nạp lại đúng chuẩn bàn giao của TV4 (derivatives/models/<stem>.*)
    stem = "within_EEGNet_sub-01_seed42_fold0"
    torch.save(model.state_dict(), tmp_path / f"{stem}.pt")
    np.savez(tmp_path / f"{stem}_norm.npz", mean=np.zeros(n_ch, dtype=np.float32), std=np.ones(n_ch, dtype=np.float32))
    np.savez(tmp_path / f"{stem}_test.npz", subject=np.array(["01", "01"]), trial_id=np.array([1, 5]))
    meta = dict(n_ch=n_ch, n_times=n_times, sfreq=128.0, tmin=0.0, tmax=0.8, arch=arch)
    (tmp_path / f"{stem}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    loaded_model, nrm, test_info, loaded_meta = explain_mod.load_tv4_bundle(stem, models_dir=tmp_path)
    assert loaded_meta["n_times"] == 103
    assert nrm["mean"].shape == (n_ch,)
    assert list(test_info["trial_id"]) == [1, 5]

    ig_map = explain_mod.compute_ig_map(loaded_model, x)
    assert ig_map.shape == (n_ch, n_times)
    assert np.isfinite(ig_map).all()


def test_spatiotemporal_occlusion_on_eegnet():
    explain_mod = _load_script_module("explain_05", "scripts/05_explain.py")
    ch_names = ["Fz", "Cz", "Pz", "CPz", "C3", "C4", "O1", "O2"]
    n_ch, n_times = len(ch_names), 101
    times = np.linspace(-0.2, 0.8, n_times)
    model = explain_mod.EEGNet(n_ch=n_ch, n_times=n_times, n_classes=2)

    rng = np.random.default_rng(42)
    X_test = rng.normal(0, 1, size=(12, n_ch, n_times)).astype(np.float32)
    y_test = np.array([0, 1] * 6, dtype=np.float32)

    df_occ = explain_mod.run_eegnet_occlusion(
        model, X_test, y_test, times=times, ch_names=ch_names, save_outputs=False
    )
    assert len(df_occ) == 9
    assert {"condition", "baseline_auc", "occluded_auc", "delta_auc"} <= set(df_occ.columns)
    assert np.isfinite(df_occ["delta_auc"]).all()

    b_zero_uv = explain_mod.compute_norm_zero_baseline(np.array([2.0, -4.0]), np.array([2.0, 2.0]), "zero_uv")
    np.testing.assert_allclose(b_zero_uv, [-1.0, 2.0])
    b_mean = explain_mod.compute_norm_zero_baseline(np.array([2.0, -4.0]), np.array([2.0, 2.0]), "train_mean")
    np.testing.assert_allclose(b_mean, [0.0, 0.0])


def test_confound_feature_extraction_channel_masks():
    conf_mod = _load_script_module("confounds_06", "scripts/06_confounds.py")
    ch_names = ["Fz", "Cz", "Pz", "C3", "C4", "O1", "O2", "POz"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types="eeg")
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1e-6, size=(6, len(ch_names), 251))
    events = np.column_stack([np.arange(6) * 300, np.zeros(6, dtype=int), np.array([1, 2, 1, 2, 1, 2])])
    epochs = mne.EpochsArray(
        data, info, events=events, tmin=-0.2, event_id={"standard": 1, "target": 2}, verbose=False
    )

    # Full channels (8 kênh x 8 time bins = 64 features)
    X_full = conf_mod.extract_spatiotemporal_features(epochs, tmin=0.0, tmax=0.8, n_time_bins=8)
    assert X_full.shape == (6, 8 * 8)

    # Exclude Motor C3 & Cz (còn 6 kênh x 8 time bins = 48 features)
    X_no_motor = conf_mod.extract_spatiotemporal_features(
        epochs, tmin=0.0, tmax=0.8, exclude_channels=conf_mod.MOTOR_C3_CHANNELS, n_time_bins=8
    )
    assert X_no_motor.shape == (6, 6 * 8)

    # Include Occipital O1, O2, POz in 80-200ms (3 kênh x 8 time bins = 24 features)
    X_occ = conf_mod.extract_spatiotemporal_features(
        epochs, tmin=0.08, tmax=0.20, include_channels=conf_mod.OCCIPITAL_CHANNELS, n_time_bins=8
    )
    assert X_occ.shape == (6, 3 * 8)
