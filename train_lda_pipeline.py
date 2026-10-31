import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, classification_report
from process_data import load_data
from src.eegoddball.config import RESULTS_DIR
from extract_features import *

def train_lda_pipeline(sub_id, fold_name, file_name = 'lda_predictions_with_features.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    X_base, y_base, X_test, y_test, test_indices = load_data(sub_id, fold_name, flatten = True)

    X_train, X_val, y_train, y_val = train_test_split(
        X_base, y_base,
        test_size = 0.2,
        random_state = 42,
        stratify = y_base
    )

    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    classifier = LinearDiscriminantAnalysis(solver='svd')
    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_val)
    print("[Validation] Balanced Accuracy:", balanced_accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    test_preds = classifier.predict(X_test)
    target_class_idx = np.where(classifier.classes_ == 10001)[0][0]
    test_probs = classifier.predict_proba(X_test)[:, target_class_idx] if len(np.unique(y_train)) == 2 else np.max(classifier.predict_proba(X_test), axis=1)

    results_df = pd.DataFrame({
        'trial_id': test_indices,
        'true_label': y_test,
        'prediction': test_preds,
        'probability': test_probs
    })
    
    results_df.to_csv(output, index=False)
    print(f"[Hoàn tất] File dự đoán lưu tại: {output}")
    
    return results_df

def combine_features(X_raw, X_xdawn, sfreq = 250):
    feat_xdawn = X_xdawn.reshape(X_xdawn.shape[0], -1)
    feat_window = extract_window_amplitude(X_raw)
    feat_band = extract_bandpower(X_raw, sfreq=sfreq)
    feat_se = extract_spectral_entropy(X_raw, sfreq=sfreq)
    feat_wav = extract_wavelet_features(X_raw)

    return np.hstack([feat_xdawn, feat_window, feat_band, feat_se, feat_wav])

def train_lda_pipeline_with_feat(sub_id, fold_name, file_name = 'lda_predictions.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    X_base, y_base, X_test_raw, y_test, test_indices = load_data(sub_id, fold_name, flatten = False)

    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_base, y_base,
        test_size = 0.2,
        random_state = 42,
        stratify = y_base
    )

    print(f"Train: {X_train_raw.shape[0]} | Val: {X_val_raw.shape[0]} | Test: {X_test_raw.shape[0]}")

    X_train_xdawn, X_val_xdawn = extract_xdawn_features(X_train_raw, y_train, X_val_raw)
    _, X_test_xdawn = extract_xdawn_features(X_train_raw, y_train, X_test_raw)

    X_train = combine_features(X_train_raw, X_train_xdawn)
    X_val = combine_features(X_val_raw, X_val_xdawn)
    X_test = combine_features(X_test_raw, X_test_xdawn)

    print(f"Number of features after extracting: {X_train.shape[1]}")

    classifier = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    classifier.fit(X_train, y_train)

    preds = classifier.predict(X_val)
    print("[Validation] Balanced Accuracy:", balanced_accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    test_preds = classifier.predict(X_test)
    target_class_idx = np.where(classifier.classes_ == 10001)[0][0]
    test_probs = classifier.predict_proba(X_test)[:, target_class_idx] if len(np.unique(y_train)) == 2 else np.max(classifier.predict_proba(X_test), axis=1)

    results_df = pd.DataFrame({
        'trial_id': test_indices,
        'true_label': y_test,
        'prediction': test_preds,
        'probability': test_probs
    })
    
    results_df.to_csv(output, index=False)
    print(f"[Hoàn tất] File dự đoán lưu tại: {output}")
    
    return results_df