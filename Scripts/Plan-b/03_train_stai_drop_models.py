"""
03_train_stai_drop_models.py
Trains and evaluates classification models (Random Forest, XGBoost, SVM, KNN)
to predict whether a physiological window corresponds to a lowered state anxiety
state relative to baseline (Delta < 0), using strict subject-grouped splitting.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'WESAD').exists())
OUTPUT_DIR = REPO_ROOT / 'Data' / 'Preprocessed n STAI-labeled'
MODELS_DIR = REPO_ROOT / 'Models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    pkl_files = sorted(OUTPUT_DIR.glob('S*_stai.pkl'))
    if not pkl_files:
        raise FileNotFoundError(f"No S*_stai.pkl files found in {OUTPUT_DIR}")

    dfs = []
    for path in pkl_files:
        df = pd.read_pickle(path)
        if 'condition_stai_mean' not in df.columns or 'baseline_stai_mean' not in df.columns:
            continue
        df['stai_delta'] = df['condition_stai_mean'] - df['baseline_stai_mean']
        df['stai_drop'] = (df['stai_delta'] < 0).astype(int)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    metadata_columns = {
        'binary_anxiety', 'subject_id', 'condition', 'window_start_seconds',
        'window_end_seconds', 'condition_stai_mean', 'baseline_stai_mean',
        'stai_delta', 'stai_drop'
    }
    feature_columns = [c for c in combined_df.columns if c not in metadata_columns]

    X = combined_df[feature_columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
    y = combined_df['stai_drop'].astype(int)
    groups = combined_df['subject_id'].astype(int)

    print(f"Dataset loaded. Shape: {combined_df.shape}")
    print("Class distribution for STAI Drop (0 = Delta >= 0, 1 = Delta < 0):")
    print(y.value_counts())

    if y.nunique() < 2:
        raise ValueError("Both classes (0 and 1) must be present in the dataset.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    train_subjects = sorted(set(groups.iloc[train_idx]))
    test_subjects = sorted(set(groups.iloc[test_idx]))

    print(f"Train subjects: {train_subjects} | Test subjects: {test_subjects}")
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }

    results = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan

        print(f"{name} Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        model_path = MODELS_DIR / f"stai_drop_{name.lower().replace(' ', '_')}_model.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")


if __name__ == '__main__':
    main()
