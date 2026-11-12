import sys
sys.path.append(r'e:\Rizqy\Kuliah\Skripsi\repo-skripsi\skripsi-venv\Lib\site-packages')

import pickle
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

# Custom Pickler for NumPy 1.x compatibility (protocol 4, rewrites numpy._core -> numpy.core)
class Numpy1CompatPickler(pickle._Pickler):
    def save_global(self, obj, name=None):
        if getattr(obj, '__module__', None) and obj.__module__.startswith('numpy._core'):
            obj.__module__ = obj.__module__.replace('numpy._core', 'numpy.core')
        super().save_global(obj, name=name)

def dump_numpy1_compat(obj, f):
    Numpy1CompatPickler(f, protocol=4).dump(obj)

DATA_3CLASS_DIR = Path('Data/6th try/3class')
DATA_2CLASS_DIR = Path('Data/6th try/2class')
DATA_2CLASS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_2CLASS_DIR = Path('Models/6th try/2class')
MODELS_2CLASS_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

print('1. Generating 2-class datasets in Data/6th try/2class/...')
dfs = []
for sid in SUBJECTS:
    f_in = DATA_3CLASS_DIR / f'S{sid}_stai_3class_60s_shift6s.pkl'
    f_out = DATA_2CLASS_DIR / f'S{sid}_stai_2class_60s_shift6s.pkl'
    
    df = pd.read_pickle(f_in)
    # Ensure binary labeling: 0 for delta_stai_sum <= 0, 1 for delta_stai_sum > 0
    df['anxiety_label'] = (df['delta_stai_sum'] > 0).astype(int)
    
    # Save with NumPy 1.x compat
    with open(f_out, 'wb') as f:
        dump_numpy1_compat(df, f)
    
    c0 = int((df['anxiety_label'] == 0).sum())
    c1 = int((df['anxiety_label'] == 1).sum())
    print(f'  S{sid}: saved {len(df)} windows (Class 0: {c0}, Class 1: {c1}) -> {f_out.name}')
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
metadata_columns = {
    'anxiety_label', 'binary_anxiety', 'three_class_anxiety', 'subject_id',
    'condition', 'window_start_seconds', 'window_end_seconds',
    'condition_stai_mean', 'baseline_stai_mean', 'delta_stai',
    'condition_stai_sum', 'baseline_stai_sum', 'delta_stai_sum', 'delta_stai_mean'
}
feature_columns = [col for col in combined_df.columns if col not in metadata_columns]
X = combined_df[feature_columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
y = combined_df['anxiety_label'].astype(int)
groups = combined_df['subject_id'].astype(int)

print(f'\nCombined 2-Class Dataset:')
print(f'  Shape: {combined_df.shape}')
print(f'  Features: {len(feature_columns)}')
print(f'  Class distribution: {y.value_counts().to_dict()}')

# Validation Scheme A (Naive Random Split)
print('\n==================== VALIDATION SCHEME A (NAIVE RANDOM SPLIT) ====================')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

models = {
    'XGBoost': XGBClassifier(n_estimators=100, eval_metric='logloss', random_state=0),
    'Random Forest (max_depth=25, updated)': RandomForestClassifier(max_depth=25, random_state=0),
    'Random Forest (max_depth=4, Alkurdi)': RandomForestClassifier(max_depth=4, random_state=0),
    'SVM (RBF)': SVC(kernel='rbf', random_state=0),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM Linear (C=1.0)': SVC(kernel='linear', C=1.0, random_state=0),
    'SVM Linear (C=0.9)': SVC(kernel='linear', C=0.9, random_state=0),
    'LDA': LinearDiscriminantAnalysis(),
}

scheme_a_results = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(X_train_sc, y_train)
    y_pred = m.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average='macro')
    f1_cls = f1_score(y_test, y_pred, average=None, labels=[0, 1])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    train_time = round(time.time() - t0, 2)
    
    scheme_a_results[name] = {
        'accuracy': round(float(acc), 4),
        'balanced_accuracy': round(float(bacc), 4),
        'macro_f1': round(float(f1_m), 4),
        'class_0_f1': round(float(f1_cls[0]), 4),
        'class_1_f1': round(float(f1_cls[1]), 4),
        'confusion_matrix': cm.tolist(),
        'train_time': train_time
    }
    print(f'{name:40s} | Acc: {acc*100:6.2f}% | BAcc: {bacc*100:6.2f}% | F1: {f1_m:.4f} | F1_0: {f1_cls[0]:.4f} | F1_1: {f1_cls[1]:.4f} | Time: {train_time}s')

    # Save Scheme A trained model
    safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_').replace(':', '').replace(',', '')
    model_filename = MODELS_2CLASS_DIR / f'stai_2class_schemeA_{safe_name}_model.joblib'
    joblib.dump({'model': m, 'scaler': sc, 'features': feature_columns}, model_filename)

# Also save the standard names used in notebook cell 13
notebook_standard_models = {
    'lda': 'LDA',
    'random_forest': 'Random Forest (max_depth=25, updated)',
    'random_forest_max_depth_4': 'Random Forest (max_depth=4, Alkurdi)',
    'svm_linear': 'SVM Linear (C=1.0)',
    'svm_linear_c_1.0': 'SVM Linear (C=1.0)',
    'svm_linear_c_0.9': 'SVM Linear (C=0.9)',
    'svm': 'SVM (RBF)',
    'svm_rbf': 'SVM (RBF)',
    'xgboost': 'XGBoost',
    'knn': 'KNN (k=5)',
}

# Validation Scheme B (LOSO CV)
print('\n==================== VALIDATION SCHEME B (LEAVE-ONE-SUBJECT-OUT CV) ====================')
logo = LeaveOneGroupOut()
scheme_b_results = {}
confusion_matrices = {}

for name, m_proto in models.items():
    t0 = time.time()
    preds = np.zeros(len(y), dtype=int)
    subject_accs = []
    subject_baccs = []
    subject_f1s = []

    for train_idx, test_idx in logo.split(X, y, groups=groups):
        m = type(m_proto)(**m_proto.get_params())
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_te = scaler.transform(X.iloc[test_idx])
        m.fit(X_tr, y.iloc[train_idx])
        fold_pred = m.predict(X_te)
        preds[test_idx] = fold_pred

        y_true_fold = y.iloc[test_idx]
        subject_accs.append(accuracy_score(y_true_fold, fold_pred))
        subject_baccs.append(balanced_accuracy_score(y_true_fold, fold_pred))
        subject_f1s.append(f1_score(y_true_fold, fold_pred, average='macro', zero_division=0))

    acc = accuracy_score(y, preds)
    bacc = balanced_accuracy_score(y, preds)
    f1_m = f1_score(y, preds, average='macro')
    f1_cls = f1_score(y, preds, average=None, labels=[0, 1])
    cm = confusion_matrix(y, preds, labels=[0, 1])
    confusion_matrices[name] = cm.tolist()
    eval_time = round(time.time() - t0, 2)

    scheme_b_results[name] = {
        'pooled_accuracy': round(float(acc), 4),
        'pooled_balanced_accuracy': round(float(bacc), 4),
        'pooled_macro_f1': round(float(f1_m), 4),
        'class_0_f1': round(float(f1_cls[0]), 4),
        'class_1_f1': round(float(f1_cls[1]), 4),
        'macro_subject_accuracy': round(float(np.mean(subject_accs)), 4),
        'macro_subject_balanced_accuracy': round(float(np.mean(subject_baccs)), 4),
        'macro_subject_macro_f1': round(float(np.mean(subject_f1s)), 4),
        'eval_time': eval_time
    }
    print(f'{name:40s} | Acc: {acc*100:6.2f}% | BAcc: {bacc*100:6.2f}% | F1: {f1_m:.4f} | F1_0: {f1_cls[0]:.4f} | F1_1: {f1_cls[1]:.4f}')
    print(f'Confusion Matrix:\n{cm}')

summary_out = {
    'validation_scheme_a': scheme_a_results,
    'validation_scheme_b': scheme_b_results,
    'confusion_matrices': confusion_matrices
}

with open(MODELS_2CLASS_DIR / 'evaluation_summary.json', 'w') as f:
    json.dump(summary_out, f, indent=2)
print('\nSaved 2-class evaluation summary to Models/6th try/2class/evaluation_summary.json!')
