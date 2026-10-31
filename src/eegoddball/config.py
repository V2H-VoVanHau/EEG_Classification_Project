import os
import hashlib
from pathlib import Path

# 1. Đường dẫn thư mục

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(ROOT_DIR, "data", "ds004621")
DERIVATIVES_DIR = os.path.join(ROOT_DIR, "derivatives")
SPLIT_DIR = os.path.join(ROOT_DIR, "splits")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# Tự động tạo thư mục đầu ra
for path in [
    DERIVATIVES_DIR,
    SPLIT_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
]:
    os.makedirs(path, exist_ok=True)

# 2. Phiên bản dữ liệu

DATA_VERSION = "v1"

# Đường dẫn epochs đang sử dụng
EPOCHS_DIR = os.path.join(DERIVATIVES_DIR, DATA_VERSION)

# 3. Tham số tiền xử lý

SFREQ_NEW = 250
L_FREQ = 0.1
H_FREQ = 40
TMIN = -0.2
TMAX = 0.8

# 4. Danh sách đối tượng

SUBJECTS = [f"{i:02d}" for i in range(1, 43)]
SAMPLE_SUBJECTS = ["01", "02", "03"]

# 5. Nhãn sự kiện

# Mã nhãn được lưu trong epochs
EVENT_ID = {
    "standard": 1,
    "target": 2,
    "distractor": 3,
}

# Ánh xạ từ cột event_type trong events.tsv
STIMULUS_CODES = {
    "S 5": "standard",
    "S 6": "target",
    "S 7": "distractor",
}

# Số kích thích gốc của mỗi đối tượng
EXPECTED_TRIALS = 660

# Đã đối chiếu events.tsv: sub-23, 38 dừng giữa khối rồi chạy lại khối cuối;
# sub-24, 37 mất một marker standard
EXPECTED_TRIALS_BY_SUBJECT = {"23": 692, "24": 659, "37": 659, "38": 775}


def get_expected_trials(sub):
    return EXPECTED_TRIALS_BY_SUBJECT.get(sub, EXPECTED_TRIALS)


# 6. Chia tập và seed

SEED = 42

# Số fold within-subject.
# Cross-subject không được vượt quá số người đang dùng.
N_SPLITS = 5

# Tỷ lệ validation lấy từ phần còn lại sau khi bỏ test
VAL_SIZE = 0.2

# Seed chạy lặp mô hình học sâu.
# Thống nhất danh sách này với TV4 trước khi chạy chính thức.
DL_SEEDS = [42, 43, 44]
# 7. File folds và mã băm
FOLDS_PATH = Path(SPLIT_DIR) / "folds.json"


def get_folds_hash():
    """Tính SHA-256 từ nội dung folds.json hiện tại."""
    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Chưa có file {FOLDS_PATH}. "
            "Hãy chạy script tạo folds trước khi ghi dự đoán."
        )

    return hashlib.sha256(
        FOLDS_PATH.read_bytes()
    ).hexdigest()


# Cho phép import config khi chưa tạo folds.json.
# Không được ghi None vào cột folds_hash của file dự đoán.
FOLDS_HASH = (
    get_folds_hash()
    if FOLDS_PATH.is_file()
    else None
)

# Tham số thống kê
N_BOOTSTRAP = 10000
N_PERMUTATIONS = 10000
PERMUTATION_EXACT_MAX_N = 16