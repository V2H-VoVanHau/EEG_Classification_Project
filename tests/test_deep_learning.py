"""Kiểm thử cho src/05_deep_learning.py (TV4) trên dữ liệu giả — không cần dữ liệu thật.

Chạy: python -m pytest tests/test_deep_learning.py -q
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

mne = pytest.importorskip("mne")
torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "src" / "05_deep_learning.py"


@pytest.fixture()
def dl(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("dl05", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name, sub in [("EPOCHS_DIR", "v0"), ("RESULTS_DIR", "results"), ("MODELS_DIR", "models")]:
        (tmp_path / sub).mkdir()
        monkeypatch.setattr(mod, name, tmp_path / sub)
    monkeypatch.setattr(mod, "TRAIN", dict(mod.TRAIN, max_epochs=3, patience=2))
    return mod


def make_epochs(path, sub, n=200, seed=0, with_trial_id=True):
    """Epochs giả 8 kênh, 250 Hz, -0,2..0,8 s; target có một sóng dương ~350 ms."""
    rng = np.random.default_rng(seed)
    sfreq, tmin = 250.0, -0.2
    times = tmin + np.arange(251) / sfreq
    kind = rng.choice(["standard", "target", "distractor"], size=n, p=[0.76, 0.12, 0.12])
    kind[:2] = ["standard", "target"]
    X = rng.normal(0, 10e-6, (n, 8, len(times)))
    X[kind == "target"] += 8e-6 * np.exp(-((times - 0.35) ** 2) / 0.0128)
    event_id = {"standard": 1, "target": 2, "distractor": 3}
    events = np.c_[np.arange(n) * 300 + 100, np.zeros(n, int), [event_id[k] for k in kind]]
    meta = pd.DataFrame(dict(trial_id=np.arange(n))) if with_trial_id else None
    info = mne.create_info([f"E{i}" for i in range(8)], sfreq, "eeg")
    ep = mne.EpochsArray(X, info, events=events, tmin=tmin, event_id=event_id,
                         metadata=meta, verbose="error")
    ep.save(path / f"sub-{sub}_task-oddball-epo.fif", overwrite=True, verbose="error")
    return kind


def tv1_folds(subs, n=200, k=5, seed=42):
    """Đúng định dạng splits/fold.json mà TV1 đang commit."""
    rng = np.random.default_rng(seed)
    within = {s: {f"fold_{i}": {"test_trial_ids": sorted(map(int, part))}
                  for i, part in enumerate(np.array_split(rng.permutation(n), k))} for s in subs}
    cross = {f"fold_{i}": {"test_subjects": [s]} for i, s in enumerate(subs)}
    return {"within_subject": within, "cross_subject": cross}


def doc_folds(subs, n=200, k=5, seed=42):
    """Định dạng trong tài liệu 13.1.3: within dạng list, cross dạng list."""
    tv1 = tv1_folds(subs, n, k, seed)
    within = {s: [{"test": f["test_trial_ids"]} for f in folds.values()]
              for s, folds in tv1["within_subject"].items()}
    cross = [{"test": f["test_subjects"]} for f in tv1["cross_subject"].values()]
    return {"within": within, "cross": cross}


def test_both_folds_schemas_read_the_same(dl):
    a, b = tv1_folds(["01", "02"]), doc_folds(["01", "02"])
    assert dl.within_subjects(a) == dl.within_subjects(b) == ["01", "02"]
    for k in range(5):
        np.testing.assert_array_equal(dl.within_test_ids(a, "01", k), dl.within_test_ids(b, "01", k))
    assert dl.n_cross_folds(a) == dl.n_cross_folds(b) == 2
    assert dl.cross_test_subjects(a, 1) == dl.cross_test_subjects(b, 1) == ["02"]


def test_fold_keys_sorted_numerically(dl):
    folds = {"within_subject": {"01": {f"fold_{i}": {"test": [i]} for i in (10, 2, 0, 1)}}}
    assert [int(dl.within_test_ids(folds, "01", k)[0]) for k in range(4)] == [0, 1, 2, 10]


def test_missing_trial_id_is_an_error(dl):
    """Không đánh số theo vị trí: v1 loại epoch nên vị trí lệch khỏi folds.json (13.1.3)."""
    make_epochs(dl.EPOCHS_DIR, "01", with_trial_id=False)
    with pytest.raises(ValueError, match="trial_id"):
        dl.load_subject("01")


def test_load_subject_keeps_trial_id_and_drops_distractor(dl):
    kind = make_epochs(dl.EPOCHS_DIR, "01")
    d = dl.load_subject("01")
    keep = np.flatnonzero(kind != "distractor")
    np.testing.assert_array_equal(d["trial_id"], keep)
    np.testing.assert_array_equal(d["y"], (kind[keep] == "target").astype(int))
    assert d["X"].shape[1:] == (8, 103)               # 128 Hz, 0..0,8 s


def test_rows_from_other_folds_hash_are_dropped(dl):
    path = dl.RESULTS_DIR / "predictions_tv4.csv"
    row = dict(scheme="within", subject="01", fold=0, model="EEGNet", seed=42, trial_id=1,
               y_true=1, y_score=0.9, data_version="v0", folds_hash="OLD")
    dl.write_predictions(pd.DataFrame([row]), path)
    dl.write_predictions(pd.DataFrame([dict(row, seed=43, folds_hash="NEW")]), path)
    out = pd.read_csv(path)
    assert out.folds_hash.unique().tolist() == ["NEW"] and out.seed.tolist() == [43]


def test_within_end_to_end_no_leakage_and_bundle(dl):
    kind = make_epochs(dl.EPOCHS_DIR, "01")
    folds = tv1_folds(["01"])
    args = SimpleNamespace(dry_run=False, val_by_subject=False)
    dl.run_within(["01"], folds, "hash0", [0], torch.device("cpu"), args)

    pred = pd.read_csv(dl.RESULTS_DIR / "predictions_tv4.csv", dtype={"subject": str})
    assert list(pred.columns) == dl.PRED_COLS
    # mỗi trial standard/target nằm trong tập test đúng một lần
    assert sorted(pred.trial_id) == sorted(np.flatnonzero(kind != "distractor"))
    for k in range(5):
        expected = set(dl.within_test_ids(folds, "01", k)) & set(np.flatnonzero(kind != "distractor"))
        assert set(pred[pred.fold == k].trial_id) == expected
    assert pred.y_score.between(0, 1).all()

    model, nrm, test, meta = dl.load_model("within_EEGNet_sub-01_seed0_fold0")
    assert set(test["trial_id"]) == set(pred[pred.fold == 0].trial_id)
    assert meta["n_times"] == 103 and nrm["mean"].shape == (8,)

    hist = pd.read_csv(dl.RESULTS_DIR / "history_tv4.csv")
    dl.run_within(["01"], folds, "hash0", [0], torch.device("cpu"), args)   # chạy lại
    assert len(pd.read_csv(dl.RESULTS_DIR / "history_tv4.csv")) == len(hist)
    assert len(pd.read_csv(dl.RESULTS_DIR / "predictions_tv4.csv")) == len(pred)


def test_cross_subject_holds_out_whole_people(dl):
    for i, s in enumerate(["01", "02", "03"]):
        make_epochs(dl.EPOCHS_DIR, s, seed=i)
    args = SimpleNamespace(dry_run=False, val_by_subject=True)
    dl.run_cross(["01", "02", "03"], tv1_folds(["01", "02", "03"]), "hash0", [0],
                 torch.device("cpu"), args)
    pred = pd.read_csv(dl.RESULTS_DIR / "predictions_tv4.csv", dtype={"subject": str})
    assert pred.groupby("fold").subject.unique().map(list).to_dict() == {0: ["01"], 1: ["02"], 2: ["03"]}
    meta = json.loads((dl.MODELS_DIR / "cross_EEGNet_seed0_fold1_meta.json").read_text("utf-8"))
    assert meta["val_by_subject"] is True and meta["subject"] is None


def test_validation_split_never_touches_test(dl):
    y = np.r_[np.zeros(170, int), np.ones(30, int)]
    rest = np.arange(40, 200)
    tr, va = dl.split_validation(rest, y)
    assert not set(tr) & set(va)
    assert set(tr) | set(va) == set(rest)
    assert abs(len(va) - 0.2 * len(rest)) <= 1
