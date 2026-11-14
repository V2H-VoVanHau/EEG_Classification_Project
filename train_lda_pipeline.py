import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score, classification_report
from process_data import load_data
from src.eegoddball.config import RESULTS_DIR
from extract_features import *
from src.eegoddball import config as C
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def train_lda_pipeline(sub_id, fold_name, file_name = 'lda_predictions.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    X_base, y_base, X_test, y_test, test_indices = load_data(sub_id, fold_name, flatten = True)

    print(f"Train: {X_base.shape[0]} | Test: {X_test.shape[0]}") 

    classifier = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    classifier.fit(X_base, y_base)

    test_probs = classifier.predict_proba(X_test)[:, 1] 

    results_df = pd.DataFrame({
        'scheme': 'within_subject',
        'subject': sub_id,
        'fold': fold_name,
        'model': 'LDA-shrink',
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

def combine_features(X_raw, X_xdawn, sfreq = 250):
    feat_xdawn = X_xdawn.reshape(X_xdawn.shape[0], -1)
    feat_window = extract_window_amplitude(X_raw)
    feat_band = extract_bandpower(X_raw, sfreq=sfreq)
    feat_se = extract_spectral_entropy(X_raw, sfreq=sfreq)
    feat_wav = extract_wavelet_features(X_raw)

    return np.hstack([feat_xdawn, feat_window, feat_band, feat_se, feat_wav])

def train_lda_pipeline_with_feat(sub_id, fold_name, file_name = 'lda_predictions_with_features.csv'):
    output = os.path.join(RESULTS_DIR, file_name)
    X_base, y_base, X_test_raw, y_test, test_indices = load_data(sub_id, fold_name, flatten = False)

    print(f"Train: {X_base.shape[0]} | Test: {X_test_raw.shape[0]}")

    xdawn_model = fit_xdawn(X_base, y_base)

    X_base_xdawn = apply_xdawn(xdawn_model, X_base)
    X_test_xdawn = apply_xdawn(xdawn_model, X_test_raw)
    
    X_base_features = combine_features(X_base, X_base_xdawn)
    X_test_features = combine_features(X_test_raw, X_test_xdawn)

    print(f"Number of features after extracting: {X_base_features.shape[1]}")

    classifier = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    )

    classifier.fit(X_base_features, y_base)

    test_probs = classifier.predict_proba(X_test_features)[:, 1]

    results_df = pd.DataFrame({
        'scheme': 'within_subject',
        'subject': sub_id,
        'fold': fold_name,
        'model': 'LDA-shrink-xDawn',
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