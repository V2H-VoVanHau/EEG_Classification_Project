from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
import numpy as np
import src.eegoddball.config as C
from src.eegoddball.config import RESULTS_DIR
from extract_features import fit_xdawn, apply_xdawn
from process_data import load_data

def train_rf_pipeline_with_feat(sub_id, fold_name, file_name='rf_predictions.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    
    # 1. Load dữ liệu dạng 3D (flatten=False để dùng xDAWN)
    X_base, y_base, X_test, y_test, test_indices = load_data(sub_id, fold_name, flatten=False)

    print(f"Train: {X_base.shape[0]} | Test: {X_test.shape[0]}") 

    # 2. Trích xuất đặc trưng bằng xDAWN (Tránh data leakage)
    xdawn_model = fit_xdawn(X_base, y_base)
    
    X_base_xdawn = apply_xdawn(xdawn_model, X_base)
    X_test_xdawn = apply_xdawn(xdawn_model, X_test)
    
    # TỐI ƯU TÍNH TOÁN: Duỗi phẳng các đặc trưng không gian đã được nén
    X_train_features = X_base_xdawn.reshape(X_base_xdawn.shape[0], -1)
    X_test_features = X_test_xdawn.reshape(X_test_xdawn.shape[0], -1)

    print(f"Number of features after extracting: {X_train_features.shape[1]}")

    # 3. Khởi tạo mô hình Random Forest
    classifier = make_pipeline(
        # RF không nhạy cảm với scale nhưng giữ StandardScaler để đồng nhất cấu trúc Pipeline
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,          # Tăng số lượng cây quyết định để có kết quả ổn định
            class_weight='balanced',   # Xử lý mất cân bằng lớp của dữ liệu P300 (Oddball)
            n_jobs=-1,                 # QUAN TRỌNG: Dùng tất cả core CPU để chạy song song chống lag
            max_depth=10,              # Giới hạn độ sâu để tránh overfitting trên dữ liệu nhiễu cao
            random_state=42            # Cố định seed
        )
    )

    # 4. Huấn luyện và dự đoán
    classifier.fit(X_train_features, y_base)
    
    # Lấy xác suất của class 1 (Target)
    test_probs = classifier.predict_proba(X_test_features)[:, 1] 

    # 5. Lưu kết quả đúng định dạng
    results_df = pd.DataFrame({
        'scheme': 'within_subject',
        'subject': sub_id,
        'fold': fold_name,
        'model': 'RandomForest-xDAWN',
        'seed': 42,
        'trial_id': test_indices,
        'y_true': y_test,
        'y_score': test_probs,
        'data_version': C.DATA_VERSION,
        'folds_hash': C.get_folds_hash()
    })
    
    results_df.to_csv(output, index=False)
    print(f"[Hoàn tất] File dự đoán lưu tại: {output}")
    
    return results_df