import os
#Đường dẫn thư mục tiêu chuẩn
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data" , "ds004621")
DERIVATIVES_DIR = os.path.join(ROOT_DIR, "derivatives")
SPLIT_DIR = os.path.join(ROOT_DIR, "splits")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")
# Tự động tạo cấu trúc thư mục nếu chưa có
for path in [DERIVATIVES_DIR, SPLIT_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(path, exist_ok=True)
#Tham số tiền xử lý & cắt Epoch cố định
SEED = 42
SFREQ_NEW = 250
L_FREQ = 0.1
H_FREQ = 40
TMIN, TMAX = -0.2, 0.8
#danh sách các đối tượng thí nghiệm
SUBJECTS = [f"{i:02d}" for i in range(1, 43)]
SAMPLE_SUBJECTS = ["01", "02", "03"]
#Mã sự kiện cố định
EVENT_ID = { "standard": 1, "target": 2, "distractor": 3 }