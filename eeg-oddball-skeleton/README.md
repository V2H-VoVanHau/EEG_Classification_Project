# EEG Oddball: single-trial classification, ML vs Deep Learning

Mini-project: phân loại kích thích **target** và **frequent (standard)** từ EEG một lần thử, trên bộ dữ liệu
Nencki-Symfonia (OpenNeuro ds004621). So sánh học máy tiêu chuẩn (LDA, SVM, RF) với học sâu (EEGNet)
và giải thích đặc trưng mà mô hình học được.

## Cấu trúc thư mục

Repo hiện chỉ có **khung thư mục**. Các file bên dưới là tên **dự kiến**: chủ file tự tạo file của mình đúng vị trí và tên này.

```text
eeg-oddball-project/
├── .github/
│   └── pull_request_template.md # checklist trước khi gửi PR
├── src/eegoddball/              # code DÙNG CHUNG — import được từ mọi script
│   ├── config.py                # TV1  · mọi tham số: đường dẫn, seed, dải lọc, cửa sổ epoch
│   ├── splits.py                # TV1  · tạo/đọc folds.json, tách train/val/test
│   ├── predictions.py           # TV1  · định dạng file dự đoán (hợp đồng nộp bài)
│   ├── evaluate.py              # TV1  · máy chấm: kiểm tra cùng đề, metric, Wilcoxon, bootstrap
│   ├── data.py                  # TV2  · định nghĩa trial_id, cắt epoch, nạp X/y
│   ├── features.py              # TV3  · đặc trưng thủ công
│   ├── models.py                # TV4  · kiến trúc EEGNet
│   └── figstyle.py              # TV5  · màu, font, lưu hình 300 dpi
├── scripts/                     # các BƯỚC của pipeline, chạy theo số thứ tự
│   ├── 00_download.sh           # TV1  · tải dữ liệu oddball
│   ├── 00_make_folds.py         # TV1  · tạo splits/folds.json (1 lần, tuần 1)
│   ├── 00_make_v0.py            # TV1  · dữ liệu nháp v0 cho 3 người
│   ├── 01_preprocess.py         # TV2  · lọc, kênh xấu, ICA, báo cáo QC
│   ├── 02_epochs_erp.py         # TV2  · epochs v1 + ERP, Hình 4–5
│   ├── 03_ml_models.py          # TV3  · LDA, LogReg, SVM, RF → predictions_tv3.csv
│   ├── 04_deep_learning.py      # TV4  · EEGNet → predictions_tv4_*.csv + mô hình .pt
│   ├── 05_explain.py            # TV5  · giải mã theo thời gian, Integrated Gradients, occlusion
│   ├── 06_confounds.py          # TV5  · thí nghiệm confound → predictions_tv5_confounds.csv
│   └── 07_evaluate.py           # TV1  · chấm điểm mọi mô hình, RQ1–RQ2, Hình 7–8
├── tests/                       # kiểm thử (mỗi người thêm cho phần mình)
├── splits/folds.json            # "đề thi" — được commit, không ai tự sửa
├── data/                        # dữ liệu gốc        (KHÔNG commit)
├── derivatives/                 # v0, v1, mô hình    (KHÔNG commit)
├── results/                     # file dự đoán, bảng điểm (commit)
├── figures/                     # hình cho báo cáo và slide (commit)
├── notebooks/                   # chỉ để khám phá, không dùng cho kết quả chính thức
├── report/  ·  slides/          # báo cáo và slide
├── run_all.py                   # TV1 · chạy cả pipeline (tạo sau)
├── requirements.txt  ·  pyproject.toml
```

Phân biệt hai loại thư mục code: **`src/eegoddball/` là thư viện** (hàm dùng chung, không tự chạy),
**`scripts/` là các bước** (mỗi file chạy được độc lập, đọc đầu vào từ bước trước và ghi đầu ra cho bước sau).

## Luồng dữ liệu

```text
data/ds004621 ──01──▶ derivatives/clean_raw ──02──▶ derivatives/v1 ──┬─03─▶ results/predictions_tv3.csv ─┐
       │                                                              ├─04─▶ results/predictions_tv4_*.csv ┼─07─▶ results/summary.csv
       └──00──▶ splits/folds.json · derivatives/v0 (nháp, tuần 1)     ├─06─▶ results/predictions_tv5_*.csv ┘      figures/fig7, fig8
                                                                      └─05─▶ figures/fig9, fig10 (+ mô hình .pt của bước 04)
```

## Cài đặt (mỗi người làm 1 lần)

```bash
git clone <url-repo> && cd eeg-oddball-project
conda create -n eegproj python=3.11 -y && conda activate eegproj
pip install -r requirements.txt
pip install -e .              # để mọi script import được `eegoddball` khi src/ đã có code
```

## Năm luật của repo

1. **Một file, một chủ** (chủ file ghi ở cột bên phải cây thư mục). Muốn đổi file của người khác → mở issue hoặc PR.
2. **Tham số chỉ nằm trong `config.py`.** Không viết cứng dải lọc, seed, đường dẫn trong script.
3. **Tập test chỉ do `splits/folds.json` quy định**, theo `trial_id` (mã gốc của kích thích), không theo vị trí.
   Ở mỗi fold: bỏ tập test ra trước → tách validation từ phần còn lại → huấn luyện → dự đoán test đúng 1 lần.
4. **Mô hình chỉ nộp dự đoán, TV1 chấm điểm.** Mọi file dự đoán ghi qua `PredictionWriter`; máy chấm từ chối
   nếu các mô hình không thi cùng tập test, hoặc lẫn phiên bản dữ liệu/folds.
5. **Không commit dữ liệu.** Mỗi người một nhánh (`tv2-ica`, `tv4-eegnet`...), PR nhỏ.

## Việc cần làm ngay tuần 1

- [ ] TV2 mở `events.tsv` thật, sửa `EVENT_COLUMN` và `EVENT_MAP` trong `config.py` (qua PR cho TV1)
- [ ] TV1 tạo `config.py`, `folds.json` và dữ liệu nháp v0
