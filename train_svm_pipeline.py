from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import os
import numpy as np
from extract_features import *
from src.eegoddball.config import RESULTS_DIR
import src.eegoddball.config as C
from process_data import load_data

def train_svm_pipeline_with_feat(sub_id, fold_name, file_name='svm_predictions.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    
    # 1. Load dữ liệu dạng 3D (bắt buộc flatten=False để dùng xDAWN)
    X_base, y_base, X_test, y_test, test_indices = load_data(sub_id, fold_name, flatten=False)

    print(f"Train: {X_base.shape[0]} | Test: {X_test.shape[0]}") 

    # 2. Trích xuất đặc trưng không gian bằng xDAWN (Tránh data leakage)
    xdawn_model = fit_xdawn(X_base, y_base)
    
    X_base_xdawn = apply_xdawn(xdawn_model, X_base)
    X_test_xdawn = apply_xdawn(xdawn_model, X_test)
    
    # TỐI ƯU TÍNH TOÁN: Chỉ dùng đặc trưng từ xDAWN (số chiều thấp)
    X_train_features = X_base_xdawn.reshape(X_base_xdawn.shape[0], -1)
    X_test_features = X_test_xdawn.reshape(X_test_xdawn.shape[0], -1)

    print(f"Number of features after extracting: {X_train_features.shape[1]}")

    # 3. Khởi tạo mô hình Support Vector Machine
    classifier = make_pipeline(
        StandardScaler(),
        SVC(
            kernel='linear',             # Dùng 'rbf' (phi tuyến) hoặc 'linear' (tuyến tính)
            probability=True,         
            class_weight='balanced',  # Xử lý mất cân bằng lớp của dữ liệu Oddball
            random_state=42
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
        'model': 'SVM-xDAWN',
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