import sys
sys.path.append(r'e:\Rizqy\Kuliah\Skripsi\repo-skripsi\skripsi-venv\Lib\site-packages')

import numpy as np
import pandas as pd
from pathlib import Path
import json, time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

WESAD_DIR = Path('WESAD')
DATA_DIR = Path('Data/6th try')
MODELS_DIR = Path('Models/6th try')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
STAI_REVERSE = {0, 3, 5}

def score_stai_responses(raw):
    return raw, [(5.0 - v) if i in STAI_REVERSE else v for i, v in enumerate(raw)]

def parse_clock_mmss(val):
    if not val or pd.isna(val): return np.nan
    val = str(val).strip()
    if '.' in val:
        m, s = val.split('.', 1)
        return int(m)*60 + int(s)
    return int(val)*60

def _parse_numeric_fields(parts):
    res = []
    for p in parts:
        p = p.strip()
        if not p: res.append(np.nan)
        else:
            try: res.append(float(p))
            except ValueError: res.append(np.nan)
    return res

def parse_quest_file(subject_id):
    quest_path = WESAD_DIR / f'S{subject_id}' / f'S{subject_id}_quest.csv'
    order = start_raw = end_raw = None
    stai_rows = []
    with open(quest_path, 'r', encoding='latin1') as f:
        for line in f:
            parts = [x.strip() for x in line.rstrip('\r\n').split(';')]
            tag = parts[0].strip() if parts else ''
            if tag.startswith('# ORDER'):
                order = [x.strip() for x in parts[1:] if x.strip()]
            elif tag.startswith('# START'):
                start_raw = [x.strip() for x in parts[1:]]
            elif tag.startswith('# END'):
                end_raw = [x.strip() for x in parts[1:]]
            elif tag.startswith('# STAI'):
                stai_rows.append(_parse_numeric_fields(parts[1:]))

    conditions = []
    for index in range(min(len(order), len(stai_rows))):
        condition = order[index].strip()
        if not condition: continue
        start_text, end_text = start_raw[index].strip(), end_raw[index].strip()
        start_seconds = parse_clock_mmss(start_text) if start_text else np.nan
        end_seconds = parse_clock_mmss(end_text) if end_text else np.nan
        responses = [float(v) for v in stai_rows[index] if pd.notna(v)]
        raw_responses, scored_responses = score_stai_responses(responses)
        valid_raw = [float(v) for v in raw_responses if pd.notna(v)]
        valid_scored = [float(v) for v in scored_responses if pd.notna(v)]
        conditions.append({
            'condition': condition,
            'start_seconds': start_seconds,
            'end_seconds': end_seconds,
            'stai_sum': float(np.sum(valid_scored)),
            'stai_mean': float(np.mean(valid_scored)),
        })
    return conditions

# 1. Parse questionnaires and compute labels
cond_labels = {}
for sid in SUBJECTS:
    conds = parse_quest_file(sid)
    base = next(c for c in conds if c['condition'].strip().lower() == 'base')
    base_sum = base['stai_sum']
    base_mean = base['stai_mean']
    cond_labels[sid] = {}
    for c in conds:
        cname = c['condition']
        if cname.strip().lower() == 'base': continue
        d_sum = c['stai_sum'] - base_sum
        d_norm = d_sum / 24.0
        d_mean = c['stai_mean'] - base_mean
        if d_sum > 1: lbl = 2
        elif d_sum >= -1: lbl = 1
        else: lbl = 0
        cond_labels[sid][cname] = {
            'label': lbl,
            'delta_stai': round(d_norm, 4),
            'delta_stai_sum': d_sum,
            'delta_stai_mean': round(d_mean, 4),
            'condition_stai_mean': round(c['stai_mean'], 4),
            'baseline_stai_mean': round(base_mean, 4),
            'condition_stai_sum': c['stai_sum'],
            'baseline_stai_sum': base_sum
        }

# 2. Update files in Data/6th try/
dfs = []
for sid in SUBJECTS:
    f = DATA_DIR / f'S{sid}_stai_3class_60s_shift6s.pkl'
    df = pd.read_pickle(f)
    # keep only the 4 protocol conditions, dropping bRead / sRead
    df = df[df['condition'].isin(['Fun', 'Medi 1', 'TSST', 'Medi 2'])].copy()
    for k in ['anxiety_label', 'delta_stai', 'delta_stai_sum', 'delta_stai_mean', 'condition_stai_mean', 'baseline_stai_mean', 'condition_stai_sum', 'baseline_stai_sum']:
        df[k] = [cond_labels[sid][c][k if k in cond_labels[sid][c] else ('label' if k == 'anxiety_label' else k)] for c in df['condition']]
    # Save back to pickle
    df.to_pickle(f)
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)
metadata_columns = {
    'anxiety_label', 'binary_anxiety', 'three_class_anxiety', 'subject_id',
    'condition', 'window_start_seconds', 'window_end_seconds',
    'condition_stai_mean', 'baseline_stai_mean', 'delta_stai',
    'condition_stai_sum', 'baseline_stai_sum', 'delta_stai_sum', 'delta_stai_mean',
    'ACC_subwindow_count'
}
feature_columns = [col for col in combined_df.columns if col not in metadata_columns]
X = combined_df[feature_columns].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0)
y = combined_df['anxiety_label'].astype(int)
groups = combined_df['subject_id'].astype(int)

print(f'Combined dataset shape: {combined_df.shape}')
print(f'Feature columns count: {len(feature_columns)}')
print(f'Class distribution: {y.value_counts().to_dict()}')

# 3. Validation Scheme A (Naive Random Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

models = {
    'XGBoost': XGBClassifier(n_estimators=100, eval_metric='mlogloss', random_state=0),
    'Random Forest (max_depth=25, updated)': RandomForestClassifier(max_depth=25, random_state=0),
    'Random Forest (max_depth=4, Alkurdi)': RandomForestClassifier(max_depth=4, random_state=0),
    'SVM (RBF)': SVC(kernel='rbf', random_state=0),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'SVM Linear (C=1.0)': SVC(kernel='linear', C=1.0, random_state=0),
    'SVM Linear (C=0.9)': SVC(kernel='linear', C=0.9, random_state=0),
    'LDA': LinearDiscriminantAnalysis(),
}

print('\n==================== VALIDATION SCHEME A (NAIVE RANDOM SPLIT) ====================')
scheme_a_results = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(X_train_sc, y_train)
    y_pred = m.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    f1_m = f1_score(y_test, y_pred, average='macro')
    f1_cls = f1_score(y_test, y_pred, average=None, labels=[0, 1, 2])
    train_time = round(time.time() - t0, 2)
    scheme_a_results[name] = {
        'accuracy': round(float(acc), 4),
        'balanced_accuracy': round(float(bacc), 4),
        'macro_f1': round(float(f1_m), 4),
        'class_0_f1': round(float(f1_cls[0]), 4),
        'class_1_f1': round(float(f1_cls[1]), 4),
        'class_2_f1': round(float(f1_cls[2]), 4),
        'train_time': train_time
    }
    print(f'{name:40s} | Acc: {acc*100:6.2f}% | BAcc: {bacc*100:6.2f}% | F1: {f1_m:.4f} | F1_0: {f1_cls[0]:.4f} | F1_1: {f1_cls[1]:.4f} | F1_2: {f1_cls[2]:.4f} | Time: {train_time}s')

# 4. Validation Scheme B (LOSO CV)
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
    f1_cls = f1_score(y, preds, average=None, labels=[0, 1, 2])
    cm = confusion_matrix(y, preds, labels=[0, 1, 2])
    confusion_matrices[name] = cm.tolist()
    eval_time = round(time.time() - t0, 2)

    scheme_b_results[name] = {
        'pooled_accuracy': round(float(acc), 4),
        'pooled_balanced_accuracy': round(float(bacc), 4),
        'pooled_macro_f1': round(float(f1_m), 4),
        'class_0_f1': round(float(f1_cls[0]), 4),
        'class_1_f1': round(float(f1_cls[1]), 4),
        'class_2_f1': round(float(f1_cls[2]), 4),
        'macro_subject_accuracy': round(float(np.mean(subject_accs)), 4),
        'macro_subject_balanced_accuracy': round(float(np.mean(subject_baccs)), 4),
        'macro_subject_macro_f1': round(float(np.mean(subject_f1s)), 4),
        'eval_time': eval_time
    }
    print(f'{name:40s} | Acc: {acc*100:6.2f}% | BAcc: {bacc*100:6.2f}% | F1: {f1_m:.4f} | F1_0: {f1_cls[0]:.4f} | F1_1: {f1_cls[1]:.4f} | F1_2: {f1_cls[2]:.4f}')
    print(f'Confusion Matrix:\n{cm}')

summary_out = {
    'validation_scheme_a': scheme_a_results,
    'validation_scheme_b': scheme_b_results,
    'confusion_matrices': confusion_matrices
}

with open(MODELS_DIR / 'evaluation_summary.json', 'w') as f:
    json.dump(summary_out, f, indent=2)
print('\nSaved updated evaluation summary to Models/6th try/evaluation_summary.json!')
