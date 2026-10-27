"""
05_deep_learning.py — TV4 · Học sâu (EEGNet)
Chủ file: TV4. Người review: TV5 (theo vòng review 13.1.9).

Phạm vi tuần 1–3
  * Tuần 1: EEGNet bằng PyTorch, chạy được trên v0 của 1 người.
  * Tuần 2: vòng 5 bước theo folds.json của TV1; trọng số lớp; validation tách từ phần train
    (20 %, giữ tỷ lệ lớp, seed 42); early stopping trên validation; xuất results/predictions_tv4.csv
    đúng hợp đồng 13.1.5 (TV1 chấm điểm, TV4 không tự chấm).
  * Tuần 3: within-subject và cross-subject, 3 seed; gói bàn giao cho TV5 (13.1.6) trong
    derivatives/models/: <stem>.pt + _test.npz (trial_id test) + _norm.npz + _meta.json;
    lịch sử loss từng epoch ở results/history_tv4.csv (learning curves tuần 5).

Cách chạy (từ thư mục gốc dự án)
  python src/05_deep_learning.py --scheme within --subjects 01 --seeds 42  # mốc tuần 1: 1 người, 1 seed
  python src/05_deep_learning.py --scheme both                              # within + cross, mọi seed
  python src/05_deep_learning.py --scheme cross --val-by-subject           # validation tách theo người
  python src/05_deep_learning.py --scheme within --dry-run                  # không cần torch: chỉ kiểm tra
                                                                            # fold / validation / định dạng file
  python -m pytest tests/test_deep_learning.py -q                           # kiểm thử trên dữ liệu giả

Các khóa đọc từ src/eegoddball/config.py của TV1 (thiếu khóa nào thì dùng mặc định bên dưới và cảnh báo):
  DATA_VERSION ("v0"/"v1"), DERIVATIVES_DIR, EPOCHS_DIR, SPLIT_DIR, FOLDS_PATH, RESULTS_DIR, MODELS_DIR,
  SUBJECTS, SEED (42), VAL_SIZE (0.2), DL_SEEDS, FOLDS_HASH (chuỗi hoặc hàm).
folds.json: đọc được cả dạng trong tài liệu ({"within", "cross"}) lẫn dạng TV1 đang commit
  ({"within_subject": {"01": {"fold_0": {"test_trial_ids": [...]}}}, "cross_subject": {...}}).
  folds_hash = sha256 đầy đủ của folds.json (giống config.get_folds_hash của TV1).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:  # torch chỉ bắt buộc khi huấn luyện thật; --dry-run chạy được không cần torch
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None

try:
    from eegoddball import config as C  # config.py của TV1 theo README: src/eegoddball/config.py
except ImportError:
    try:
        import config as C  # dự phòng: config.py nằm cùng thư mục src/ hoặc trên PYTHONPATH
    except ImportError:
        C = None

for _stream in (sys.stdout, sys.stderr):  # console Windows (cp1252) không in được tiếng Việt
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ----------------------------------------------------------------------------------------------
# 0. Tham số
# ----------------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_missing_cfg: list[str] = []


def cfg(name, default):
    """Đọc tham số chung từ config.py (một nguồn sự thật); ghi nhận khóa nào đang dùng mặc định."""
    if C is not None and hasattr(C, name):
        return getattr(C, name)
    _missing_cfg.append(name)
    return default


DATA_VERSION = cfg("DATA_VERSION", "v0")
DERIVATIVES_DIR = Path(cfg("DERIVATIVES_DIR", PROJECT_ROOT / "derivatives"))
EPOCHS_DIR = Path(cfg("EPOCHS_DIR", DERIVATIVES_DIR / DATA_VERSION))
FOLDS_PATH = Path(cfg("FOLDS_PATH", Path(cfg("SPLIT_DIR", PROJECT_ROOT / "splits")) / "folds.json"))
RESULTS_DIR = Path(cfg("RESULTS_DIR", PROJECT_ROOT / "results"))
MODELS_DIR = Path(cfg("MODELS_DIR", DERIVATIVES_DIR / "models"))  # README: mô hình nằm trong derivatives/
SEED = cfg("SEED", 42)            # seed của quy tắc tách validation (chung cả nhóm)
VAL_SIZE = cfg("VAL_SIZE", 0.2)
DL_SEEDS = list(cfg("DL_SEEDS", [42, 43, 44]))  # seed khởi tạo mạng; 3–5 seed theo 13.1.2

# Tham số riêng của học sâu (TV4 sở hữu). Ghi đúng các con số này vào mục 3.5 của báo cáo.
SFREQ_DL = 128.0                  # Hz — hạ mẫu cho học sâu (9.2)
BASELINE = (-0.2, 0.0)            # s — hiệu chỉnh đường nền như Phần 6
TMIN, TMAX = 0.0, 0.8             # s — cửa sổ đưa vào mạng → 103 mẫu ở 128 Hz
EVENT_NEG, EVENT_POS = "standard", "target"   # tên sự kiện theo hợp đồng dữ liệu 13.1.7

ARCH = dict(F1=8, D=2, F2=16, k1=64, drop=0.5)  # k1 = 64 mẫu = 0,5 s ở 128 Hz
TRAIN = dict(batch_size=64, lr=1e-3, max_epochs=150, patience=20)

MODEL_NAME = "EEGNet"
PRED_COLS = ["scheme", "subject", "fold", "model", "seed", "trial_id",
             "y_true", "y_score", "data_version", "folds_hash"]
RUN_KEYS = ["scheme", "model", "seed", "subject", "fold"]


# ----------------------------------------------------------------------------------------------
# 1. EEGNet (Lawhern et al., 2018) — TV5 import lớp này để chạy Integrated Gradients / occlusion
# ----------------------------------------------------------------------------------------------
if torch is not None:
    class EEGNet(nn.Module):
        """EEGNet-8,2. Đầu vào (batch, n_ch, n_times) đã chuẩn hóa; đầu ra logit 2 lớp
        (cột 1 = target)."""

        def __init__(self, n_ch, n_times, n_classes=2, F1=8, D=2, F2=16, k1=64, drop=0.5):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, k1), padding=(0, k1 // 2), bias=False),       # lọc thời gian
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False),           # lọc không gian
                nn.BatchNorm2d(F1 * D), nn.ELU(),
                nn.AvgPool2d((1, 4)), nn.Dropout(drop))
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, 1, bias=False),                               # trộn bản đồ đặc trưng
                nn.BatchNorm2d(F2), nn.ELU(),
                nn.AvgPool2d((1, 8)), nn.Dropout(drop))
            with torch.no_grad():
                n_feat = self.block2(self.block1(torch.zeros(1, 1, n_ch, n_times))).numel()
            self.classify = nn.Linear(n_feat, n_classes)

        def forward(self, x):                                   # x: (batch, n_ch, n_times)
            x = self.block2(self.block1(x.unsqueeze(1)))
            return self.classify(x.flatten(1))


def get_device():
    if torch is None:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------------------
# 2. Đọc dữ liệu (chỉ đọc epochs của TV1/TV2, không sửa)
# ----------------------------------------------------------------------------------------------
def norm_sub(s) -> str:
    s = str(s)
    return s[4:] if s.startswith("sub-") else s


def _event_codes(event_id: dict, name: str) -> list[int]:
    """Mã sự kiện ứng với một lớp theo hợp đồng 13.1.7 ("target", cả dạng "Stimulus/target")."""
    return [v for k, v in event_id.items() if k == name or name in k.split("/")]


def load_subject(sub: str) -> dict:
    """Đọc epochs v0/v1 của một người → mảng cho học sâu.
    Trả về X (n_trials, n_ch, n_times) đơn vị µV, y (1 = target), trial_id gốc (0–659)."""
    import mne

    files = sorted(EPOCHS_DIR.glob(f"sub-{sub}*-epo.fif"))
    if len(files) != 1:
        raise FileNotFoundError(f"Cần đúng 1 file sub-{sub}*-epo.fif trong {EPOCHS_DIR}, thấy {files}")
    ep = mne.read_epochs(files[0], preload=True, verbose="error")
    if ep.metadata is None or "trial_id" not in ep.metadata:
        # Không tự đánh số theo vị trí: v1 loại epoch nên vị trí sẽ lệch khỏi folds.json
        raise ValueError(f"{files[0].name}: thiếu cột trial_id trong metadata (hợp đồng 13.1.3)")

    neg_codes, pos_codes = _event_codes(ep.event_id, EVENT_NEG), _event_codes(ep.event_id, EVENT_POS)
    if not neg_codes or not pos_codes:
        raise ValueError(f"{files[0].name}: không tìm thấy sự kiện {EVENT_NEG}/{EVENT_POS} "
                         f"trong event_id={ep.event_id} (hợp đồng 13.1.7)")
    ep = ep[np.isin(ep.events[:, 2], neg_codes + pos_codes)]   # bài toán chính: frequent vs target
    if ep.baseline is None:
        ep.apply_baseline(BASELINE, verbose="error")
    if not np.isclose(ep.info["sfreq"], SFREQ_DL):
        ep.resample(SFREQ_DL, verbose="error")
    ep.crop(TMIN, TMAX)

    X = (ep.get_data(picks="eeg") * 1e6).astype(np.float32)   # V → µV
    y = np.isin(ep.events[:, 2], pos_codes).astype(np.int64)
    trial_id = ep.metadata["trial_id"].to_numpy().astype(np.int64)
    ch_names = [ep.ch_names[i] for i in mne.pick_types(ep.info, eeg=True)]
    return dict(X=X, y=y, trial_id=trial_id, ch_names=ch_names,
                sfreq=float(ep.info["sfreq"]), tmin=float(ep.tmin))


def check_subject_data(sub, d):
    assert len(np.unique(d["trial_id"])) == len(d["trial_id"]), f"sub-{sub}: trial_id bị trùng"
    assert set(np.unique(d["y"])) == {0, 1}, f"sub-{sub}: không đủ hai lớp"


# ----------------------------------------------------------------------------------------------
# 3. folds.json (luật của TV1) và quy tắc tách validation
# ----------------------------------------------------------------------------------------------
def load_folds() -> tuple[dict, str]:
    folds = json.loads(FOLDS_PATH.read_text(encoding="utf-8"))
    # Cách tính của TV1: sha256 đầy đủ của folds.json. Tính lại ngay lúc chạy thay vì tin
    # C.FOLDS_HASH (là None nếu config được import trước khi có folds.json).
    h = hashlib.sha256(FOLDS_PATH.read_bytes()).hexdigest()
    cfg_hash = getattr(C, "FOLDS_HASH", None) if C is not None else None
    if callable(cfg_hash):
        cfg_hash = cfg_hash()
    if cfg_hash is not None and cfg_hash != h:
        raise ValueError(f"FOLDS_HASH trong config ({cfg_hash}) khác sha256 của {FOLDS_PATH} ({h}).")
    return folds, h


# Đọc được cả hai cách viết folds.json:
#   tài liệu 13.1.3 : {"within": {"01": [{"test": [...]}, ...]}, "cross": [{"test": ["01"]}, ...]}
#   bản TV1 commit  : {"within_subject": {"01": {"fold_0": {"test_trial_ids": [...]}}},
#                      "cross_subject": {"fold_0": {"test_subjects": ["01"]}}, "metadata": {...}}
_TEST_KEYS = ("test", "test_trial_ids", "test_subjects")


def _section(folds, scheme):
    for key in (scheme, f"{scheme}_subject"):
        if key in folds:
            return folds[key]
    raise KeyError(f"folds.json không có mục '{scheme}' (thấy {list(folds)})")


def _fold_list(entries) -> list:
    """Danh sách fold theo thứ tự k = 0, 1, ...; nhận list hoặc dict {"fold_0": ...}."""
    if isinstance(entries, dict):
        keys = sorted(entries, key=lambda s: int(str(s).rsplit("_", 1)[-1]))
        return [entries[k] for k in keys]
    return list(entries)


def _test_list(entry):
    if isinstance(entry, dict):
        for key in _TEST_KEYS:
            if key in entry:
                return entry[key]
        raise KeyError(f"fold không có khóa tập test nào trong {_TEST_KEYS}: {list(entry)}")
    return entry


def _within_entry(folds, sub) -> list:
    w = _section(folds, "within")
    for key in (sub, f"sub-{sub}", str(int(sub)) if sub.isdigit() else None):
        if key is not None and key in w:
            return _fold_list(w[key])
    raise KeyError(f"folds.json không có within cho sub-{sub} (có: {list(w)})")


def within_subjects(folds) -> list[str]:
    return [norm_sub(s) for s in _section(folds, "within")]


def within_test_ids(folds, sub, k) -> np.ndarray:
    return np.asarray(_test_list(_within_entry(folds, sub)[k]), dtype=np.int64)


def n_cross_folds(folds) -> int:
    return len(_fold_list(_section(folds, "cross")))


def cross_test_subjects(folds, k) -> list[str]:
    return [norm_sub(s) for s in _test_list(_fold_list(_section(folds, "cross"))[k])]


def split_validation(rest_idx, y, groups=None):
    """Bước 3: validation chỉ lấy từ phần còn lại sau khi đã cất tập test.
    groups=None → quy tắc chung của nhóm (20 %, stratify, seed 42).
    groups=mảng mã người → tách theo người (tùy chọn --val-by-subject cho cross-subject)."""
    if groups is None:
        tr, va = train_test_split(rest_idx, test_size=VAL_SIZE, stratify=y[rest_idx],
                                  random_state=SEED)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
        a, b = next(gss.split(rest_idx, y[rest_idx], groups[rest_idx]))
        tr, va = rest_idx[a], rest_idx[b]
    return np.sort(tr), np.sort(va)


# ----------------------------------------------------------------------------------------------
# 4. Chuẩn hóa — thống kê CHỈ từ tập train của fold
# ----------------------------------------------------------------------------------------------
def fit_norm(X):
    mean = X.mean(axis=(0, 2)).astype(np.float32)
    std = (X.std(axis=(0, 2)) + 1e-6).astype(np.float32)
    return dict(mean=mean, std=std)


def apply_norm(X, nrm):
    return (X - nrm["mean"][None, :, None]) / nrm["std"][None, :, None]


# ----------------------------------------------------------------------------------------------
# 5. Huấn luyện và dự đoán
# ----------------------------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _batched(model, X, device, bs=256):
    out = []
    for i in range(0, len(X), bs):
        out.append(model(torch.as_tensor(X[i:i + bs], device=device)))
    return torch.cat(out)


@torch.no_grad() if torch is not None else (lambda f: f)
def predict_proba(model, X, device):
    model.eval()
    return torch.softmax(_batched(model, X, device), dim=1)[:, 1].cpu().numpy()


def train_eegnet(Xtr, ytr, Xva, yva, seed, device, hp=TRAIN, arch=ARCH):
    """Bước 4: huấn luyện trên train, early stopping theo loss validation.
    Trả về (mô hình ở epoch tốt nhất, số epoch tốt nhất, lịch sử từng epoch)."""
    set_seed(seed)
    model = EEGNet(Xtr.shape[1], Xtr.shape[2], **arch).to(device)
    counts = np.bincount(ytr, minlength=2)
    w = torch.tensor(len(ytr) / (2.0 * counts), dtype=torch.float32, device=device)  # trọng số lớp
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])

    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                    batch_size=hp["batch_size"], shuffle=True, generator=gen,
                    drop_last=(len(ytr) % hp["batch_size"] == 1))  # tránh batch 1 mẫu cho BatchNorm
    yva_t = torch.from_numpy(yva).to(device)

    best, best_epoch, wait, best_state, history = np.inf, 0, 0, None, []
    for epoch in range(1, hp["max_epochs"] + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(yb), n + len(yb)

        model.eval()
        with torch.no_grad():
            logits = _batched(model, Xva, device)
            val_loss = loss_fn(logits, yva_t).item()
            p_va = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        history.append(dict(epoch=epoch, train_loss=tot / n, val_loss=val_loss,
                            val_auc=roc_auc_score(yva, p_va)))

        if val_loss < best - 1e-6:
            best, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= hp["patience"]:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, history


def dry_run_fit_predict(Xtr, ytr, Xte, seed):
    """Chỉ dùng cho --dry-run: mô hình tuyến tính thay EEGNet để kiểm tra luồng dữ liệu.
    Kết quả ghi ra file riêng, KHÔNG nộp cho TV1 như kết quả học sâu."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1e-3, class_weight="balanced", max_iter=500, random_state=seed)
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return clf.predict_proba(Xte.reshape(len(Xte), -1))[:, 1]


# ----------------------------------------------------------------------------------------------
# 6. Lưu kết quả: file dự đoán, gói bàn giao cho TV5, lịch sử huấn luyện
# ----------------------------------------------------------------------------------------------
def model_stem(scheme, seed, fold, sub=None):
    who = f"_sub-{sub}" if sub is not None else ""
    return f"{scheme}_{MODEL_NAME}{who}_seed{seed}_fold{fold}"


def save_bundle(stem, model, test_subject, test_trial_id, nrm, meta):
    """Gói bàn giao 13.1.6: state_dict + danh sách test + thông số chuẩn hóa + meta."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / f"{stem}.pt")
    np.savez(MODELS_DIR / f"{stem}_test.npz", subject=np.asarray(test_subject, dtype=str),
             trial_id=np.asarray(test_trial_id, dtype=np.int64))
    np.savez(MODELS_DIR / f"{stem}_norm.npz", mean=nrm["mean"], std=nrm["std"])
    (MODELS_DIR / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                                  encoding="utf-8")


def load_model(stem, device="cpu"):
    """Cho TV5: nạp lại mô hình đã huấn luyện mà không phải sửa code của TV4.
        import importlib; dl = importlib.import_module("05_deep_learning")
        model, nrm, test, meta = dl.load_model("within_EEGNet_sub-01_seed0_fold0")
        X_in = dl.apply_norm(X_uV, nrm)   # X_uV: epochs (n, n_ch, n_times) µV, 128 Hz, 0–0,8 s
    """
    meta = json.loads((MODELS_DIR / f"{stem}_meta.json").read_text(encoding="utf-8"))
    model = EEGNet(meta["n_ch"], meta["n_times"], **meta["arch"])
    model.load_state_dict(torch.load(MODELS_DIR / f"{stem}.pt", map_location=device))
    model.to(device).eval()
    nrm = dict(np.load(MODELS_DIR / f"{stem}_norm.npz"))
    test = dict(np.load(MODELS_DIR / f"{stem}_test.npz"))
    return model, nrm, test, meta


def _drop_stale(old: pd.DataFrame, new: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Bỏ dòng cũ chạy trên data_version/folds.json khác: máy chấm của TV1 từ chối file lẫn phiên bản."""
    if not {"data_version", "folds_hash"} <= set(old.columns):
        warnings.warn(f"{path.name}: bản cũ không có data_version/folds_hash — bỏ toàn bộ.")
        return old.iloc[0:0]
    cur = (str(new["data_version"].iloc[0]), str(new["folds_hash"].iloc[0]))
    same = (old["data_version"].astype(str) == cur[0]) & (old["folds_hash"].astype(str) == cur[1])
    if not same.all():
        warnings.warn(f"{path.name}: bỏ {int((~same).sum())} dòng cũ chạy trên data_version/folds_hash "
                      f"khác {cur}.")
    return old[same]


def write_predictions(new: pd.DataFrame, path: Path):
    """Ghi file dự đoán; chạy lại cùng (scheme, model, seed, subject, fold) thì thay dòng cũ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = new[PRED_COLS]
    if path.exists():
        old = _drop_stale(pd.read_csv(path, dtype={"subject": str}), new, path)
        done = new[RUN_KEYS].drop_duplicates().astype(str)
        key_old = old[RUN_KEYS].astype(str).agg("|".join, axis=1)
        key_new = set(done.agg("|".join, axis=1))
        new = pd.concat([old[~key_old.isin(key_new)], new], ignore_index=True)
    new = new.sort_values(["scheme", "model", "seed", "subject", "fold", "trial_id"])
    tmp = path.with_suffix(".tmp")
    new.to_csv(tmp, index=False)
    tmp.replace(path)


def append_history(rows: list[dict], path: Path):
    """Chạy lại cùng (scheme, subject, fold, seed) thì thay lịch sử cũ thay vì ghi trùng."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if path.exists():
        keys = ["scheme", "subject", "fold", "seed"]
        old = _drop_stale(pd.read_csv(path, dtype={"subject": str}), df, path)
        key_old = old[keys].astype(str).agg("|".join, axis=1)
        key_new = set(df[keys].astype(str).agg("|".join, axis=1))
        df = pd.concat([old[~key_old.isin(key_new)], df], ignore_index=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


# ----------------------------------------------------------------------------------------------
# 7. Một fold = 5 bước (13.1.3)
# ----------------------------------------------------------------------------------------------
def run_fold(*, scheme, fold, X, y, trial_id, subject, test_mask, seeds, device, args,
             folds_hash, ch_names, sub_label=None, val_groups=None):
    """Bước 1 (đọc luật) đã làm ở hàm gọi → test_mask.
    Bước 2: cất tập test. Bước 3: tách validation. Bước 4: huấn luyện. Bước 5: dự đoán đúng 1 lần."""
    te = np.flatnonzero(test_mask)                       # 2. cất tập test, không động vào đến bước 5
    rest = np.flatnonzero(~test_mask)
    tr, va = split_validation(rest, y, val_groups)       # 3. validation chỉ từ phần còn lại
    assert not (set(tr) & set(te)) and not (set(va) & set(te)) and not (set(tr) & set(va))

    nrm = fit_norm(X[tr])                                # thống kê chuẩn hóa chỉ từ train
    Xtr, Xva, Xte = apply_norm(X[tr], nrm), apply_norm(X[va], nrm), apply_norm(X[te], nrm)

    preds, hist = [], []
    for seed in seeds:
        t0 = time.time()
        if args.dry_run:
            p = dry_run_fit_predict(Xtr, y[tr], Xte, seed)
            best_epoch, history = None, []
        else:
            model, best_epoch, history = train_eegnet(Xtr, y[tr], Xva, y[va], seed, device)  # 4.
            p = predict_proba(model, Xte, device)                                            # 5.
            stem = model_stem(scheme, seed, fold, sub_label)
            meta = dict(scheme=scheme, model=MODEL_NAME, seed=seed, fold=fold, subject=sub_label,
                        n_ch=int(X.shape[1]), n_times=int(X.shape[2]), sfreq=SFREQ_DL,
                        tmin=TMIN, tmax=TMAX, baseline=list(BASELINE), unit="uV",
                        ch_names=list(ch_names), arch=ARCH, train=TRAIN, best_epoch=best_epoch,
                        n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)),
                        val_by_subject=val_groups is not None,
                        data_version=DATA_VERSION, folds_hash=folds_hash)
            save_bundle(stem, model, subject[te], trial_id[te], nrm, meta)

        preds.append(pd.DataFrame(dict(
            scheme=scheme, subject=subject[te], fold=fold, model=MODEL_NAME if not args.dry_run
            else "DRYRUN-LogReg", seed=seed, trial_id=trial_id[te], y_true=y[te], y_score=p,
            data_version=DATA_VERSION, folds_hash=folds_hash)))
        for h in history:
            hist.append(dict(scheme=scheme, subject=sub_label or "ALL", fold=fold, seed=seed,
                             best_epoch=best_epoch, data_version=DATA_VERSION,
                             folds_hash=folds_hash, **h))
        print(f"  [{scheme}] {sub_label or 'ALL'} fold {fold} seed {seed}: "
              f"train={len(tr)} val={len(va)} test={len(te)} (target={int(y[te].sum())}) "
              f"best_epoch={best_epoch} — {time.time() - t0:.1f}s", flush=True)
    return pd.concat(preds, ignore_index=True), hist


def run_within(subjects, folds, folds_hash, seeds, device, args):
    preds, hist = [], []
    in_folds = set(within_subjects(folds))
    for sub in subjects:
        if sub not in in_folds:
            print(f"sub-{sub}: folds.json chưa có fold within cho người này — bỏ qua (báo TV1).")
            continue
        d = load_subject(sub)
        check_subject_data(sub, d)
        n_folds = len(_within_entry(folds, sub))
        n_absent = sum(len(np.setdiff1d(within_test_ids(folds, sub, k), d["trial_id"]))
                       for k in range(n_folds))
        print(f"sub-{sub}: {len(d['y'])} epoch (target={int(d['y'].sum())}); {n_absent} trial_id "
              f"trong folds.json không có ở đây (distractor hoặc epoch TV2 đã loại) — bình thường.")
        for k in range(n_folds):
            test_ids = within_test_ids(folds, sub, k)                 # 1. đọc luật của TV1
            test_mask = np.isin(d["trial_id"], test_ids)
            p, h = run_fold(scheme="within", fold=k, X=d["X"], y=d["y"], trial_id=d["trial_id"],
                            subject=np.full(len(d["y"]), sub), test_mask=test_mask, seeds=seeds,
                            device=device, args=args, folds_hash=folds_hash,
                            ch_names=d["ch_names"], sub_label=sub)
            preds.append(p)
            hist += h
        flush_outputs(preds, hist, args)
        preds, hist = [], []


def run_cross(subjects, folds, folds_hash, seeds, device, args):
    data = {}
    for sub in subjects:
        data[sub] = load_subject(sub)
        check_subject_data(sub, data[sub])
    ref = data[subjects[0]]["ch_names"]
    for sub in subjects:
        if data[sub]["ch_names"] != ref:
            raise ValueError(f"sub-{sub}: danh sách/thứ tự kênh khác sub-{subjects[0]} — "
                             "cross-subject cần cùng kênh (báo TV2).")
    X = np.concatenate([data[s]["X"] for s in subjects])
    y = np.concatenate([data[s]["y"] for s in subjects])
    tid = np.concatenate([data[s]["trial_id"] for s in subjects])
    subj = np.concatenate([np.full(len(data[s]["y"]), s) for s in subjects])
    del data

    preds, hist = [], []
    for k in range(n_cross_folds(folds)):
        test_subs = cross_test_subjects(folds, k)                    # 1. đọc luật của TV1
        test_mask = np.isin(subj, test_subs)
        if not test_mask.any() or test_mask.all():
            print(f"  [cross] fold {k}: không có người test (hoặc không còn người train) "
                  f"trong dữ liệu hiện có — bỏ qua.")
            continue
        p, h = run_fold(scheme="cross", fold=k, X=X, y=y, trial_id=tid, subject=subj,
                        test_mask=test_mask, seeds=seeds, device=device, args=args,
                        folds_hash=folds_hash, ch_names=ref,
                        val_groups=subj if args.val_by_subject else None)
        preds.append(p)
        hist += h
        flush_outputs(preds, hist, args)
        preds, hist = [], []


def flush_outputs(preds, hist, args):
    """Ghi sau mỗi người (within) / mỗi fold (cross) để không mất kết quả nếu bị ngắt giữa chừng."""
    if not preds:
        return
    name = "predictions_tv4_dryrun.csv" if args.dry_run else "predictions_tv4.csv"
    write_predictions(pd.concat(preds, ignore_index=True), RESULTS_DIR / name)
    if not args.dry_run:
        append_history(hist, RESULTS_DIR / "history_tv4.csv")


# ----------------------------------------------------------------------------------------------
# 8. Dòng lệnh
# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="TV4 · EEGNet theo folds.json")
    ap.add_argument("--scheme", choices=["within", "cross", "both"], default="within")
    ap.add_argument("--subjects", nargs="*", help="vd: 01 02 03 (mặc định: SUBJECTS trong config)")
    ap.add_argument("--seeds", nargs="*", type=int, help=f"mặc định {DL_SEEDS}")
    ap.add_argument("--max-epochs", type=int, help="ghi đè TRAIN['max_epochs'] (thử nhanh)")
    ap.add_argument("--val-by-subject", action="store_true",
                    help="cross-subject: tách validation theo người thay vì theo epoch")
    ap.add_argument("--dry-run", action="store_true",
                    help="không cần torch: thay EEGNet bằng LogReg để kiểm tra fold/định dạng")
    args = ap.parse_args(argv)

    if _missing_cfg:
        warnings.warn(f"config.py thiếu {_missing_cfg} — đang dùng giá trị mặc định trong file này.")
    if not args.dry_run and torch is None:
        sys.exit("Chưa cài PyTorch. Cài torch hoặc chạy --dry-run để kiểm tra luồng dữ liệu.")
    if args.max_epochs:
        TRAIN["max_epochs"] = args.max_epochs

    folds, folds_hash = load_folds()
    if args.subjects:
        subjects = [norm_sub(s) for s in args.subjects]
    else:  # mặc định: người trong SUBJECTS của config mà đã có file epochs (v0 mới có 3 người)
        subjects = [norm_sub(s) for s in cfg("SUBJECTS", within_subjects(folds))
                    if any(EPOCHS_DIR.glob(f"sub-{norm_sub(s)}*-epo.fif"))]
    seeds = args.seeds if args.seeds else DL_SEEDS
    device = get_device()
    print(f"data={DATA_VERSION} ({EPOCHS_DIR}) | folds_hash={folds_hash} | device={device} | "
          f"subjects={len(subjects)} | seeds={seeds} | dry_run={args.dry_run}")

    if args.scheme in ("within", "both"):
        run_within(subjects, folds, folds_hash, seeds, device, args)
    if args.scheme in ("cross", "both"):
        run_cross(subjects, folds, folds_hash, seeds, device, args)
    print("Xong.")


if __name__ == "__main__":
    main()
