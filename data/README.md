# data/

Dữ liệu gốc (ds004621, phiên bản 1.0.3) đặt ở `data/ds004621/`. Thư mục này **không** được commit.

Tải bằng script của TV1 (`scripts/00_download.sh`), hoặc trực tiếp:

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds004621 data/ds004621 --exclude "*" --include "*.json" --include "*.tsv" --include "*task-oddball*"
```
