import os
import subprocess
from src.eegoddball.config import DATA_DIR, SAMPLE_SUBJECTS

def download_sample_data():
    """Tải metadata và dữ liệu Oddball của 3 người mẫu ban đầu từ OpenNeuro (ds004621).""" 
    os.makedirs(DATA_DIR, exist_ok=True)
    
    #tải metadata
    print("[1/2] tải metadata chung ")
    cmd_meta = (
        f'aws s3 sync --no-sign-request s3://openneuro.org/ds004621 {DATA_DIR} '
        '--exclude "*" --include "*.json" --include "*.tsv"'
    )
    subprocess.run(cmd_meta, shell=True, check=True)
    
    #tải riêng file tác vụ Oddball của 3 đối tượng test
    for sub in SAMPLE_SUBJECTS:
        print(f"[2/2] tải dữ liệu Oddball cho đối tượng {sub}")
        cmd_data = (
            f'aws s3 sync --no-sign-request s3://openneuro.org/ds004621/sub-{sub} {DATA_DIR}/sub-{sub} '
            '--exclude "*" --include "*task-oddball*"'
        )
        subprocess.run(cmd_data, shell=True, check=True)
        
    print("Tải dữ liệu mẫu hoàn tất. Dữ liệu được lưu tại:", DATA_DIR)

if __name__ == "__main__":
    download_sample_data()