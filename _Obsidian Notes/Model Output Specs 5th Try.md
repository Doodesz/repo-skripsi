# Model Output Specifications — `stai_6s_reference_models_stdized`

> **Source Notebook:** `Scripts/5th try/stai_6s_reference_models_stdized.ipynb`
> **Dataset:** WESAD (Wearable Stress and Affect Detection)
> **Task:** Binary Anxiety Classification (0 = No Anxiety, 1 = Anxiety)
> **Labeling Strategy:** Baseline-relative STAI self-report (not protocol-derived)
> **Normalization:** StandardScaler → MinMaxScaler pipeline

---

## 1. Dataset & Participants

| Parameter | Value |
|---|---|
| Dataset | WESAD |
| Number of Participants | 15 |
| Subject IDs | S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17 |
| Excluded | S1 (non-existent), S12 (non-existent) |

---

## 2. STAI Questionnaire & Labeling Specifications

### 2.1 Questionnaire

| Parameter | Value |
|---|---|
| Instrument | Shortened State-Trait Anxiety Inventory (STAI), 6 items |
| Scale | 1–4 Likert scale |
| Items | feel_at_ease, feel_nervous, jittery, relaxed, worried, pleasant |
| Reverse-scored items | feel_at_ease (idx 0), relaxed (idx 3), pleasant (idx 5) |
| Reverse-scoring formula | `reversed = 5.0 - raw` (so 1↔4, 2↔3) |
| Scoring convention | After reverse-scoring, higher values = higher anxiety |

### 2.2 Labeling Logic

| Parameter          | Value                                                                    |
| ------------------ | ------------------------------------------------------------------------ |
| Label type         | Baseline-relative (per-participant)                                      |
| Formula            | `Δ = condition_STAI_mean − baseline_STAI_mean`                           |
| Threshold          | Δ ≥ 0.2 → Anxiety (1); Δ < 0.2 → No Anxiety (0)                          |
| Baseline condition | "Base" condition from WESAD protocol                                     |
| Labeled conditions | All non-Base conditions (TSST, Fun, Medi 1, Medi 2, bRead, sRead, fRead) |

### 2.3 Per-Subject STAI Scores & Assigned Labels

| Subject | Baseline STAI | Conditions & Labels                                                                                    |
| ------- | ------------- | ------------------------------------------------------------------------------------------------------ |
| S2      | 2.60          | Fun (2.40, **0**); Medi 1 (2.40, **0**); Medi 2 (2.60, **0**); TSST (2.40, **0**); sRead (2.40, **0**) |
| S3      | 2.40          | Fun (1.80, **0**); Medi 1 (2.60, **1**); Medi 2 (2.20, **0**); TSST (2.80, **1**); bRead (1.40, **0**) |
| S4      | 2.40          | Fun (1.80, **0**); Medi 1 (2.00, **0**); Medi 2 (2.60, **1**); TSST (2.60, **1**); bRead (2.40, **0**) |
| S5      | 3.20          | Fun (1.60, **0**); Medi 1 (2.20, **0**); Medi 2 (2.00, **0**); TSST (3.20, **0**); bRead (2.40, **0**) |
| S6      | 2.40          | Fun (2.40, **0**); Medi 1 (2.20, **0**); Medi 2 (2.00, **0**); TSST (2.40, **0**); bRead (2.60, **1**) |
| S7      | 2.80          | Fun (1.80, **0**); Medi 1 (2.00, **0**); Medi 2 (2.20, **0**); TSST (3.00, **1**); bRead (3.00, **1**) |
| S8      | 3.00          | Fun (2.00, **0**); Medi 1 (2.20, **0**); Medi 2 (2.20, **0**); TSST (2.80, **0**); bRead (2.00, **0**) |
| S9      | 2.60          | Fun (2.60, **0**); Medi 1 (2.40, **0**); Medi 2 (2.20, **0**); TSST (2.20, **0**); bRead (2.80, **0**) |
| S10     | 3.20          | Fun (1.60, **0**); Medi 1 (2.00, **0**); Medi 2 (2.00, **0**); TSST (3.00, **0**)                      |
| S11     | 2.40          | Fun (2.20, **0**); Medi 1 (2.40, **0**); Medi 2 (2.40, **0**); TSST (2.60, **1**); bRead (1.80, **0**) |
| S13     | 3.40          | Fun (1.60, **0**); Medi 1 (1.80, **0**); Medi 2 (2.40, **0**); TSST (3.00, **0**); bRead (3.00, **0**) |
| S14     | 2.20          | Fun (2.40, **0**); Medi 1 (2.20, **0**); Medi 2 (2.20, **0**); TSST (2.80, **1**); bRead (1.80, **0**) |
| S15     | 3.60          | Fun (1.80, **0**); Medi 1 (2.20, **0**); Medi 2 (2.00, **0**); TSST (3.40, **0**); bRead (2.60, **0**) |
| S16     | 2.60          | Fun (2.60, **0**); Medi 1 (2.20, **0**); Medi 2 (2.60, **0**); TSST (2.60, **0**); bRead (2.40, **0**) |
| S17     | 3.00          | Fun (1.80, **0**); Medi 1 (1.80, **0**); Medi 2 (1.60, **0**); TSST (3.20, **1**); bRead (2.80, **0**) |

> **Note:** 8 of 15 subjects (S2, S5, S8, S9, S10, S13, S15, S16) have **zero Anxiety windows** — all their conditions scored below the 0.2 threshold above their personal baseline. Only 7 subjects (S3, S4, S6, S7, S11, S14, S17) contribute Anxiety-labeled data.

---

## 3. Signal Preprocessing Specifications

### 3.1 Sensor Modalities & Sampling Rates

| Modality | Device | Raw Sampling Rate | Processing Sampling Rate |
|---|---|---|---|
| ACC (3-axis) | Wrist (Empatica E4) | 32 Hz | **32 Hz** (kept native) |
| BVP | Wrist (Empatica E4) | 64 Hz | 70 Hz |
| EDA | Wrist (Empatica E4) | 4 Hz | 70 Hz |
| TEMP | Wrist (Empatica E4) | 4 Hz | 70 Hz |
| ACC (3-axis) | Chest (RespiBAN) | 700 Hz | **700 Hz** (kept native) |
| ECG | Chest (RespiBAN) | 700 Hz | 70 Hz |
| EDA | Chest (RespiBAN) | 700 Hz | 70 Hz |
| EMG | Chest (RespiBAN) | 700 Hz | 70 Hz |
| Resp | Chest (RespiBAN) | 700 Hz | 70 Hz |
| Temp | Chest (RespiBAN) | 700 Hz | 70 Hz |
| Label clock | WESAD protocol | 700 Hz | 700 Hz (master timeline) |

### 3.2 Signal Processing Pipeline

| Step | Detail |
|---|---|
| EDA filtering | 1.0 Hz 6th-order Butterworth low-pass filter |
| EDA decomposition | NeuroKit2 `nk.eda_phasic()` after `nk.standardize()` → SCR (Phasic) + SCL (Tonic) |
| ACC filtering | FIR filter (`filterSignalFIR`) on all 6 ACC axes |
| Resampling method | `scipy.signal.resample_poly` (polyphase) for non-ACC signals |
| Master alignment | All signals mapped onto 700 Hz WESAD label clock via index projection |
| ACC alignment | Wrist ACC at native 32 Hz, Chest ACC at native 700 Hz (no resampling) |

### 3.3 Sliding Window Specifications

| Parameter | Value |
|---|---|
| Main window length | **60 seconds** |
| Main window shift (stride) | **6 seconds** |
| Main window overlap | 90% |
| ACC sub-window length | **5 seconds** |
| ACC sub-window shift | **0.25 seconds** |
| Window retention rule | Only windows fully contained within a single labeled condition |
| Discarded windows | Windows crossing condition boundaries or in unlabeled periods |

---

## 4. Feature Extraction Specifications

### 4.1 Feature Summary

| Parameter | Value |
|---|---|
| Total features per window | **154** (used for modeling; 161 columns total minus 7 metadata columns) |
| Total columns per subject | 161 |
| Metadata columns excluded from features | `binary_anxiety`, `subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `condition_stai_mean`, `baseline_stai_mean` |

### 4.2 Feature Categories

| Category | Features | Source |
|---|---|---|
| Window statistics | mean, std, min, max for each signal channel | `get_window_stats()` |
| Net acceleration (wrist) | `net_acc` = √(ACC_x² + ACC_y² + ACC_z²) — mean, std, min, max | `get_net_accel(window, 'wrist')` |
| Net acceleration (chest) | `net_acc_C` = √(ACC_x_C² + ACC_y_C² + ACC_z_C²) — mean, std, min, max | `get_net_accel(window, 'chest')` |
| ACC sub-window aggregations | For each of 6 ACC axes × 3 features (std, abs_integral, peak_freq) × 2 aggregations (mean, std) = **36 features** + `ACC_subwindow_count` | `aggregate_acc_subwindows()` |
| BVP peak frequency | `BVP_peak_freq` | `get_peak_freq()` |
| ACC peak frequencies & integrals | `{axis}_peak_freq`, `{axis}_abs_integral` for 6 axes + `net_acc_abs_integral` | `get_peak_freq()`, `get_absolute_integral()` |
| ECG / HRV | `ECG_bpm`, `ECG_ibi`, `ECG_sdnn`, `ECG_sdsd`, `ECG_rmssd`, `ECG_pnn20`, `ECG_pnn50` | HeartPy `hp.process()` |
| EDA dynamics | `EDA_slope`, `EDA_drange`, `EDA_C_slope`, `EDA_C_drange` | `get_slope()`, `get_dynamic_range()` |
| EMG dynamics | `EMG_drange`, `EMG_abs_integral` | `get_dynamic_range()`, `get_absolute_integral()` |
| Respiration | `Resp_C_rate`, `Resp_C_Inhal_mean`, `Resp_C_Inhal_std`, `Resp_C_Exhal_mean`, `Resp_C_Exhal_std`, `Resp_C_I/E` | `get_resp_features()` |
| Temperature dynamics | `TEMP_drange`, `TEMP_slope`, `TEMP_C_drange`, `TEMP_C_slope` | `get_slope()`, `get_dynamic_range()` |

### 4.3 Data Cleaning (Pre-Training)

| Step | Detail |
|---|---|
| Type conversion | `pd.to_numeric(errors='coerce')` |
| Infinite values | Replaced with NaN |
| Missing values | NaN filled with 0 |

---

## 5. Per-Subject Window Distribution

| Subject | Total Windows | No Anxiety (0) | Anxiety (1) |
|---|---|---|---|
| S2 | 262 | 262 | 0 |
| S3 | 266 | 110 | 156 |
| S4 | 274 | 119 | 155 |
| S5 | 275 | 275 | 0 |
| S6 | 272 | 269 | 3 |
| S7 | 277 | 175 | 102 |
| S8 | 265 | 265 | 0 |
| S9 | 266 | 266 | 0 |
| S10 | 289 | 289 | 0 |
| S11 | 286 | 179 | 107 |
| S13 | 282 | 282 | 0 |
| S14 | 286 | 180 | 106 |
| S15 | 292 | 292 | 0 |
| S16 | 277 | 277 | 0 |
| S17 | 278 | 164 | 114 |
| **Total** | **4147** | **3404 (82.1%)** | **743 (17.9%)** |

---

## 6. Model Training Specifications

### 6.1 Feature Normalization Pipeline

All models use a two-stage scaling pipeline applied before the estimator:

1. **StandardScaler** — standardize features to mean=0, variance=1
2. **MinMaxScaler** — normalize to [0, 1] range

### 6.2 Model Hyperparameters

| Model | Hyperparameters |
|---|---|
| Random Forest | `n_estimators=100`, `random_state=100` |
| SVM | `kernel='rbf'`, `random_state=100` |
| XGBoost | `n_estimators=100`, `eval_metric='logloss'`, `random_state=100` |
| KNN | `n_neighbors=5` |

### 6.3 Saved Model Files

| Model | File Path |
|---|---|
| Random Forest | `Models/stai_6s_standardized/stai_baseline_relative_random_forest_model.joblib` |
| SVM | `Models/stai_6s_standardized/stai_baseline_relative_svm_model.joblib` |
| XGBoost | `Models/stai_6s_standardized/stai_baseline_relative_xgboost_model.joblib` |
| KNN | `Models/stai_6s_standardized/stai_baseline_relative_knn_model.joblib` |

---

## 7. Evaluation Results — GroupShuffleSplit (Random Window Split)

### 7.1 Split Configuration

| Parameter | Value |
|---|---|
| Split method | `GroupShuffleSplit` (grouped by `subject_id`) |
| Test size | 20% |
| Random state | 100 |
| Train shape | (3303, 154) |
| Test shape | (844, 154) |
| Train subjects | S2, S4, S5, S6, S7, S8, S9, S10, S13, S14, S16, S17 |
| Test subjects | S3, S11, S15 |
| Test class distribution | No Anxiety: 581; Anxiety: 263 |

### 7.2 Classification Results

#### Random Forest

| Metric | No Anxiety | Anxiety | Overall |
|---|---|---|---|
| Precision | 0.83 | 1.00 | — |
| Recall | 1.00 | 0.55 | — |
| F1-score | 0.91 | 0.71 | — |
| **Accuracy** | — | — | **0.8602** |
| Macro avg | 0.92 | 0.78 | 0.81 |
| Weighted avg | 0.88 | 0.86 | 0.85 |

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 581 | 0 |
| **True: Anxiety** | 118 | 145 |

#### SVM

| Metric | No Anxiety | Anxiety | Overall |
|---|---|---|---|
| Precision | 0.82 | 1.00 | — |
| Recall | 1.00 | 0.52 | — |
| F1-score | 0.90 | 0.68 | — |
| **Accuracy** | — | — | **0.8507** |
| Macro avg | 0.91 | 0.76 | 0.79 |
| Weighted avg | 0.88 | 0.85 | 0.83 |

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 581 | 0 |
| **True: Anxiety** | 126 | 137 |

#### XGBoost

| Metric | No Anxiety | Anxiety | Overall |
|---|---|---|---|
| Precision | 0.85 | 1.00 | — |
| Recall | 1.00 | 0.62 | — |
| F1-score | 0.92 | 0.77 | — |
| **Accuracy** | — | — | **0.8815** |
| Macro avg | 0.93 | 0.81 | 0.84 |
| Weighted avg | 0.90 | 0.88 | 0.87 |

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 581 | 0 |
| **True: Anxiety** | 100 | 163 |

#### KNN

| Metric | No Anxiety | Anxiety | Overall |
|---|---|---|---|
| Precision | 0.86 | 1.00 | — |
| Recall | 1.00 | 0.65 | — |
| F1-score | 0.93 | 0.79 | — |
| **Accuracy** | — | — | **0.8922** |
| Macro avg | 0.93 | 0.83 | 0.86 |
| Weighted avg | 0.91 | 0.89 | 0.88 |

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 581 | 0 |
| **True: Anxiety** | 91 | 172 |

### 7.3 GroupShuffleSplit Summary

| Model | Accuracy |
|---|---|
| Random Forest | 0.8602 |
| SVM | 0.8507 |
| XGBoost | 0.8815 |
| **KNN** | **0.8922** |

> **Note:** All models achieved perfect precision (1.00) for the Anxiety class — zero false positives — but moderate recall (0.52–0.65), meaning they miss a significant portion of true Anxiety windows. Test set contains only 3 subjects (S3, S11, S15), of which S15 has zero Anxiety windows.

---

## 8. Evaluation Results — Leave-One-Subject-Out (LOSO)

### 8.1 LOSO Configuration

| Parameter | Value |
|---|---|
| Method | `LeaveOneGroupOut` (grouped by `subject_id`) |
| Number of folds | 15 (one per subject) |
| Total test samples | 4147 |
| Ground truth: No Anxiety | 3404 |
| Ground truth: Anxiety | 743 |

### 8.2 LOSO Pooled Classification Reports

#### Random Forest (LOSO)

| Metric | No Anxiety | Anxiety |
|---|---|---|
| Precision | 0.88 | 0.58 |
| Recall | 0.94 | 0.41 |
| F1-score | 0.91 | 0.48 |
| Support | 3404 | 743 |

- **Accuracy:** 0.84
- **Macro avg:** P=0.73, R=0.67, F1=0.69
- **Weighted avg:** P=0.82, R=0.84, F1=0.83

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 3184 | 220 |
| **True: Anxiety** | 440 | 303 |

#### SVM (LOSO)

| Metric | No Anxiety | Anxiety |
|---|---|---|
| Precision | 0.89 | 0.75 |
| Recall | 0.97 | 0.44 |
| F1-score | 0.93 | 0.56 |
| Support | 3404 | 743 |

- **Accuracy:** 0.87
- **Macro avg:** P=0.82, R=0.71, F1=0.74
- **Weighted avg:** P=0.86, R=0.87, F1=0.86

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 3295 | 109 |
| **True: Anxiety** | 413 | 330 |

#### XGBoost (LOSO)

| Metric | No Anxiety | Anxiety |
|---|---|---|
| Precision | 0.89 | 0.58 |
| Recall | 0.92 | 0.49 |
| F1-score | 0.91 | 0.53 |
| Support | 3404 | 743 |

- **Accuracy:** 0.84
- **Macro avg:** P=0.74, R=0.71, F1=0.72
- **Weighted avg:** P=0.84, R=0.84, F1=0.84

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 3137 | 267 |
| **True: Anxiety** | 376 | 367 |

#### KNN (LOSO)

| Metric | No Anxiety | Anxiety |
|---|---|---|
| Precision | 0.89 | 0.68 |
| Recall | 0.95 | 0.45 |
| F1-score | 0.92 | 0.54 |
| Support | 3404 | 743 |

- **Accuracy:** 0.86
- **Macro avg:** P=0.78, R=0.70, F1=0.73
- **Weighted avg:** P=0.85, R=0.86, F1=0.85

Confusion Matrix:

|  | Pred: No Anxiety | Pred: Anxiety |
|---|---|---|
| **True: No Anxiety** | 3246 | 158 |
| **True: Anxiety** | 408 | 335 |

### 8.3 LOSO Summary — Pooled & Macro-Subject Metrics

| Model | Pooled Accuracy | Pooled Balanced Accuracy | Pooled Anxiety F1 | Macro Subject Accuracy | Macro Subject Balanced Accuracy | Macro Subject Anxiety F1 |
|---|---|---|---|---|---|---|
| Random Forest | 0.8408 | 0.6716 | 0.4787 | 0.8415 | 0.5962 | 0.1587 |
| **SVM** | **0.8741** | 0.7061 | **0.5584** | **0.8727** | **0.6836** | **0.2103** |
| XGBoost | 0.8449 | **0.7078** | 0.5330 | 0.8459 | 0.6216 | 0.1775 |
| KNN | 0.8635 | 0.7022 | 0.5421 | 0.8621 | 0.6681 | 0.2011 |

### 8.4 LOSO Per-Subject Results — Random Forest

| Subject | Windows | Accuracy | Anxiety F1 | Balanced Accuracy |
|---|---|---|---|---|
| S2 | 262 | 1.0000 | 0.0000 | NaN |
| S3 | 266 | 0.6692 | 0.6071 | 0.7179 |
| S4 | 274 | 0.4307 | 0.0000 | 0.4958 |
| S5 | 275 | 1.0000 | 0.0000 | NaN |
| S6 | 272 | 0.9890 | 0.0000 | 0.5000 |
| S7 | 277 | 0.6318 | 0.0000 | 0.5000 |
| S8 | 265 | 1.0000 | 0.0000 | NaN |
| S9 | 266 | 1.0000 | 0.0000 | NaN |
| S10 | 289 | 1.0000 | 0.0000 | NaN |
| S11 | 286 | 0.6259 | 0.6667 | 0.7011 |
| S13 | 282 | 1.0000 | 0.0000 | NaN |
| S14 | 286 | 0.6399 | 0.4926 | 0.6053 |
| S15 | 292 | 1.0000 | 0.0000 | NaN |
| S16 | 277 | 0.9892 | 0.0000 | NaN |
| S17 | 278 | 0.6475 | 0.6142 | 0.6531 |

### 8.5 LOSO Per-Subject Results — SVM

| Subject | Windows | Accuracy | Anxiety F1 | Balanced Accuracy |
|---|---|---|---|---|
| S2 | 262 | 0.9313 | 0.0000 | NaN |
| S3 | 266 | 0.5414 | 0.3579 | 0.6090 |
| S4 | 274 | 0.4562 | 0.0745 | 0.5194 |
| S5 | 275 | 1.0000 | 0.0000 | NaN |
| S6 | 272 | 0.9743 | 0.0000 | 0.4926 |
| S7 | 277 | 0.6823 | 0.2414 | 0.5686 |
| S8 | 265 | 0.8868 | 0.0000 | NaN |
| S9 | 266 | 1.0000 | 0.0000 | NaN |
| S10 | 289 | 1.0000 | 0.0000 | NaN |
| S11 | 286 | 0.9406 | 0.9264 | 0.9525 |
| S13 | 282 | 1.0000 | 0.0000 | NaN |
| S14 | 286 | 0.8147 | 0.7072 | 0.7713 |
| S15 | 292 | 1.0000 | 0.0000 | NaN |
| S16 | 277 | 1.0000 | 0.0000 | NaN |
| S17 | 278 | 0.8633 | 0.8468 | 0.8721 |

### 8.6 LOSO Per-Subject Results — XGBoost

| Subject | Windows | Accuracy | Anxiety F1 | Balanced Accuracy |
|---|---|---|---|---|
| S2 | 262 | 1.0000 | 0.0000 | NaN |
| S3 | 266 | 0.6504 | 0.5753 | 0.7019 |
| S4 | 274 | 0.4343 | 0.0000 | 0.5000 |
| S5 | 275 | 1.0000 | 0.0000 | NaN |
| S6 | 272 | 0.9890 | 0.0000 | 0.5000 |
| S7 | 277 | 0.6318 | 0.0000 | 0.5000 |
| S8 | 265 | 1.0000 | 0.0000 | NaN |
| S9 | 266 | 1.0000 | 0.0000 | NaN |
| S10 | 289 | 1.0000 | 0.0000 | NaN |
| S11 | 286 | 0.5594 | 0.6294 | 0.6480 |
| S13 | 282 | 1.0000 | 0.0000 | NaN |
| S14 | 286 | 0.5455 | 0.6084 | 0.6292 |
| S15 | 292 | 1.0000 | 0.0000 | NaN |
| S16 | 277 | 1.0000 | 0.0000 | NaN |
| S17 | 278 | 0.8777 | 0.8496 | 0.8723 |

### 8.7 LOSO Per-Subject Results — KNN

| Subject | Windows | Accuracy | Anxiety F1 | Balanced Accuracy |
|---|---|---|---|---|
| S2 | 262 | 0.9733 | 0.0000 | NaN |
| S3 | 266 | 0.6617 | 0.5946 | 0.7115 |
| S4 | 274 | 0.4526 | 0.0964 | 0.5132 |
| S5 | 275 | 1.0000 | 0.0000 | NaN |
| S6 | 272 | 0.9890 | 0.0000 | 0.5000 |
| S7 | 277 | 0.6318 | 0.0000 | 0.5000 |
| S8 | 265 | 0.7811 | 0.0000 | NaN |
| S9 | 266 | 1.0000 | 0.0000 | NaN |
| S10 | 289 | 1.0000 | 0.0000 | NaN |
| S11 | 286 | 0.9930 | 0.9906 | 0.9907 |
| S13 | 282 | 0.9894 | 0.0000 | NaN |
| S14 | 286 | 0.8776 | 0.8536 | 0.8950 |
| S15 | 292 | 1.0000 | 0.0000 | NaN |
| S16 | 277 | 1.0000 | 0.0000 | NaN |
| S17 | 278 | 0.5827 | 0.4821 | 0.5661 |

> **Note on NaN balanced accuracy:** Subjects with only one ground-truth class (all No Anxiety) cannot produce a meaningful balanced accuracy, so it is excluded (NaN) from the macro average.

---

## 9. Key Observations

1. **Severe class imbalance:** Only 17.9% of windows are labeled Anxiety (743 / 4147). Eight subjects contribute zero Anxiety windows.
2. **Macro-subject Anxiety F1 is very low** (0.16–0.21 across models) because subjects with zero Anxiety windows necessarily get F1 = 0.00, dragging down the average.
3. **SVM is the best LOSO model** across most metrics: highest pooled accuracy (0.8741), highest macro-subject balanced accuracy (0.6836), and highest macro-subject Anxiety F1 (0.2103).
4. **XGBoost achieves the highest pooled balanced accuracy** (0.7078), indicating the best overall trade-off between class-level recalls when pooling all subjects.
5. **Per-subject performance is highly variable:** For subjects with Anxiety data, LOSO Anxiety F1 ranges from 0.00 (S7 in RF/XGB/KNN) to 0.99 (S11 in KNN). S4 is consistently the hardest subject across all models.
6. **GroupShuffleSplit results are optimistic** compared to LOSO because the test set (S3, S11, S15) happened to include subjects whose patterns are more learnable from the training data.

---

## 10. Output File Locations

| Output | Path |
|---|---|
| Preprocessed feature pickles | `Data/Preprocessed n STAI-labeled 6s std-norm-ed/S{id}_stai_60s_shift6s.pkl` |
| Trained model files | `Models/stai_6s_standardized/stai_baseline_relative_{model}_model.joblib` |
| Source notebook | `Scripts/5th try/stai_6s_reference_models_stdized.ipynb` |

---

## 11. Architectural & Output Comparison: User's Model vs. Reference Model (Alkurdi's `_forked`)

> **Reference Repository:** `_forked/` (Upstream codebase *anxietyFB* by Alkurdi et al.)  
> **Reference Notebook:** `_forked/Feature Based Model WESAD.ipynb`  
> **Reference Feature Extraction:** `_forked/fb_code/feature_extraction.py`  
> **Reference Data Files:** `_forked/data/oct5_feats4.csv`, `_forked/data/subject_feats/S{id}_feats.csv`

---

### 11.1 High-Level Side-by-Side Comparison

| Parameter / Dimension           | User's Model (`stai_6s_reference_models_stdized`)                                                                                                       | Alkurdi's Reference Model (`_forked`)                                                                                       |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Target Ground Truth**         | **Subjective Self-Report Anxiety (STAI)**<br>Baseline-relative score delta: $\Delta \ge 0.2$                                                            | **Laboratory Protocol Condition**<br>Baseline (1), Stress/TSST (2), Amusement (0)                                           |
| **Classification Task**         | **Binary Classification**<br>0: No Anxiety, 1: Anxiety                                                                                                  | **3-Class Multi-class Classification**<br>0: Amusement, 1: Baseline, 2: Stress                                              |
| **Research Question**           | *"Did this individual experience a conscious increase in subjective anxiety relative to their own resting baseline?"*                                   | *"Which controlled experimental protocol condition was the subject participating in?"*                                      |
| **Main Window Length**          | **60 seconds** ($42,000$ samples at $700\text{ Hz}$)                                                                                                    | **60 seconds** ($42,000$ samples at $700\text{ Hz}$)                                                                        |
| **Window Shift (Stride)**       | **6 seconds** (90% overlap, sliding window)                                                                                                             | **60 seconds** (0% overlap, tumbling/non-overlapping)                                                                       |
| **ACC Sub-Window Extraction**   | **Yes:** 5-second sub-windows shifted by 0.25 seconds ($36$ kinetic statistical moments + count)                                                        | **None** (ACC treated as a standard 60-second modality)                                                                     |
| **Window Retention Rule**       | **Strict Temporal Enclosure** (windows crossing condition boundaries or in unmeasured periods are dropped)                                              | **Label Grouping** (`.groupby('label')` chunks entire condition into tumbling segments)                                     |
| **Total Windows (15 Subjects)** | **4,147 windows** (~$262$–$292$ windows/subject)                                                                                                        | **577 windows** (~$37$–$40$ windows/subject)                                                                                |
| **Class Distribution**          | **Severe Imbalance (82.1% / 17.9%)**<br>No Anxiety: 3,404 (82.1%)<br>Anxiety: 743 (17.9%)<br>*8 subjects have 0 Anxiety windows*                        | **Moderate Imbalance**<br>Baseline (1): 312 (54.1%)<br>Stress (2): 175 (30.3%)<br>Amusement (0): 90 (15.6%)                 |
| **Sensor Sampling Alignment**   | Polyphase resampling (`resample_poly`) to **70 Hz** for non-ACC; ACC kept at **native 32 Hz (wrist)** and **700 Hz (chest)**; mapped to 700 Hz timeline | `pd.to_datetime` with `outer` join across all sensor clocks, generating extensive `NaN` gaps followed by `bfill` and dropna |
| **Total Features per Window**   | **154 predictive features** (+ 7 metadata columns)                                                                                                      | **120 predictive features** (121 columns with target)                                                                       |
| **Feature Overlap**             | **117 features in common**; 37 unique to User (all ACC sub-window); 1 feature unique to Forked (`net_acc_C_abs_integral`)                               | 117 features in common; 3 one-hot targets (`0`, `1`, `2`)                                                                   |
| **Data Scaling Pipeline**       | Leakage-safe `Pipeline([('standardize', StandardScaler()), ('normalize', MinMaxScaler()), ('model', estimator)])` fitted strictly inside training folds | Ad-hoc `StandardScaler().fit_transform(X_train)` or pre-windowing global normalization                                      |
| **Model Architectures**         | Random Forest, SVM (RBF kernel), XGBoost, KNN ($k=5$)                                                                                                   | Linear Discriminant Analysis (LDA), Random Forest (`max_depth=4`), SVM (Linear kernel, $C=1.0$ and $C=0.9$)                 |
| **Validation Scheme A**         | **`GroupShuffleSplit` (80/20 grouped by `subject_id`)**<br>Zero subject overlap between train and test                                                  | **`train_test_split(test_size=0.2, random_state=0)`**<br>Naive random split; subject data leaked across splits              |
| **Validation Scheme B**         | **Leave-One-Subject-Out (LOSO, 15 folds)**<br>Full subject-independent generalization testing                                                           | **None** (Markdown headers for K-Fold and LOO exist in notebook but contain zero lines of code)                             |
| **Model Artifact Storage**      | Serialized `.joblib` pipelines stored in `Models/stai_6s_standardized/`                                                                                 | Not saved to disk (transient in-memory execution)                                                                           |

---

### 11.2 Detailed Methodological Differences

#### 1. Ground Truth & Construct Validity
- **Alkurdi's Model (`_forked`):** Predicts experimental stimulus conditions (TSST vs. Baseline vs. Amusement). It assumes a deterministic, one-size-fits-all response: every participant exposed to the TSST stressor is labeled as "Stress". However, psychological and clinical literature confirms that physiological stress reactivity varies widely across individuals.
- **User's Model (`5th try`):** Classifies self-reported anxiety grounded in the participant's State-Trait Anxiety Inventory (STAI). A condition is labeled Anxiety (`1`) only if the participant's post-condition STAI score was $\ge 0.2$ points higher than their own baseline score. This directly measures conscious, subjective anxiety. For example, Participant S10 underwent the TSST protocol, but their STAI score was $3.0$ vs. a Baseline of $3.2$ ($\Delta = -0.2$); the User's pipeline correctly assigns `0` (No Anxiety), whereas Alkurdi's pipeline blindly labels those same windows as `Stress (2)`.

#### 2. Temporal Resolution & Windowing Density
- **Alkurdi's Model:** Uses 60-second non-overlapping windows (stride = 60s). This produces only 37 to 40 data points per participant, totaling 577 windows across the entire WESAD cohort. While tumbling windows avoid autocorrelation, 577 samples is very small for training complex machine learning models across 120 features.
- **User's Model:** Uses 60-second sliding windows with a 6-second stride (90% overlap), generating 4,147 high-density windows. This captures fine-grained temporal dynamics and state transitions. To counteract the autocorrelation introduced by overlapping windows, the User strictly enforces subject-grouped partitioning (`GroupShuffleSplit` and LOSO).

#### 3. Accelerometer Sub-Window Dynamics
- **Alkurdi's Model:** Calculates ACC statistics (mean, std, min, max, peak frequency, absolute integral) once over the entire 60-second window. Long 60-second averages dilute short, sharp micro-movements, kinetic bursts, or fidgeting behaviors typical of anxious states.
- **User's Model:** Introduces a multi-scale hierarchy: within each 60-second window, it slices ACC into 5-second sub-windows sliding every 0.25 seconds. It computes standard deviation, absolute integral, and peak frequency for all 6 axes within each sub-window, then aggregates these distributions using mean and standard deviation. This yields **37 unique kinetic features** that capture transient movement variability.

#### 4. Signal Alignment & Numerical Cleanliness
- **Alkurdi's Model:** Converted sensor index timestamps to Python `datetime` objects and executed sequential pandas `.join(how='outer')`. Because modalities sample at disparate rates (4 Hz, 32 Hz, 64 Hz, 700 Hz), the outer join produced millions of empty rows (`NaN`) interleaved between timestamps. Labels were back-filled (`bfill`), and missing values were dropped ad-hoc inside feature functions (`.dropna()`).
- **User's Model:** Replaced outer-join padding with mathematically rigorous polyphase rational resampling (`scipy.signal.resample_poly`) to a uniform 70 Hz for non-ACC modalities, while preserving wrist ACC at native 32 Hz and chest ACC at 700 Hz. All arrays are indexed onto a clean integer master timeline at 700 Hz using bounded projection and `.groupby(level=0).mean()`.

---

### 11.3 Feature Set Comparison (161 Columns vs. 121 Columns)

```
User's Dataset (161 cols):
├── Metadata Columns (7 cols, excluded from X):
│   └── binary_anxiety, subject_id, condition, window_start_seconds,
│       window_end_seconds, condition_stai_mean, baseline_stai_mean
├── Common Predictive Features (117 cols):
│   ├── Window statistics (mean, std, min, max for raw and filtered channels)
│   ├── Net acceleration moments (net_acc, net_acc_C)
│   ├── ECG/HRV parameters (bpm, ibi, sdnn, sdsd, rmssd, pnn20, pnn50)
│   ├── EDA dynamics (EDA_slope, EDA_drange, EDA_C_slope, EDA_C_drange)
│   ├── EMG parameters (EMG_drange, EMG_abs_integral)
│   ├── Respiration features (Resp_C_rate, Inhalation/Exhalation stats, I/E ratio)
│   └── Temperature dynamics (TEMP_drange, TEMP_slope, Temp_C_drange, Temp_C_slope)
└── Unique User Features (37 cols):
    ├── ACC sub-window moments (6 axes × 3 stats × 2 aggregations = 36 cols)
    └── ACC_subwindow_count (1 col)

Alkurdi's Dataset (121 cols):
├── Target Column(s) (1 or 3 cols):
│   └── 'label' (or one-hot '0', '1', '2')
├── Common Predictive Features (117 cols)
└── Unique Forked Feature (1 col):
    └── net_acc_C_abs_integral (Chest net acceleration absolute integral)
```

---

### 11.4 Output Model Performance: Side-by-Side Comparison

#### A. Alkurdi's Reference Results (`_forked/Feature Based Model WESAD.ipynb`)
*Evaluation setup: 3-class protocol condition classification, naive 80/20 train/test random window split ($N_{\text{test}} = 116$ to $236$ windows).*

| Model Architecture | Hyperparameters | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|---|
| **Linear Discriminant Analysis (LDA)** | Default | **93.64%** (0.9364) | — | — | — |
| **Random Forest** | `max_depth=4`, `random_state=0` | **94.49%** (0.9449) | — | — | — |
| **Support Vector Machine (Linear)** | `kernel='linear'`, $C=1.0$ | **97.46%** (0.9746) | 0.9762 | 0.9586 | 0.9666 |
| **Support Vector Machine (Linear)** | `kernel='linear'`, $C=0.9$ | **97.88%** (0.9788) | 0.9790 | 0.9663 | 0.9722 |

*Confusion Matrix (SVM Linear, C=0.9, Test N=236):*
- Amusement (0): 39 correct, 4 confused with Baseline
- Baseline (1): 125 correct, 1 confused with Stress
- Stress (2): 67 correct, 0 confused

#### B. User's Results: GroupShuffleSplit (`Scripts/5th try/stai_6s_reference_models_stdized.ipynb`)
*Evaluation setup: Binary STAI anxiety classification, 80/20 subject-grouped split (Train: 12 subjects / 3,303 windows; Test: 3 subjects [S3, S11, S15] / 844 windows).*

| Model Architecture | Hyperparameters | Accuracy | Anxiety Precision | Anxiety Recall | Anxiety F1 | Macro Avg F1 |
|---|---|---|---|---|---|---|
| **Random Forest** | `n_estimators=100`, `random_state=100` | **86.02%** (0.8602) | 1.0000 | 0.5513 | 0.7108 | 0.8080 |
| **SVM (RBF Kernel)** | `kernel='rbf'`, `random_state=100` | **85.07%** (0.8507) | 1.0000 | 0.5209 | 0.6850 | 0.7925 |
| **XGBoost** | `n_estimators=100`, `eval_metric='logloss'` | **88.15%** (0.8815) | 1.0000 | 0.6198 | 0.7653 | 0.8415 |
| **KNN** | `n_neighbors=5` | **89.22%** (0.8922) | 1.0000 | 0.6540 | 0.7908 | 0.8587 |

#### C. User's Results: Leave-One-Subject-Out (LOSO) Cross-Validation
*Evaluation setup: Binary STAI anxiety classification, 15 folds (each subject held out once), evaluated across all 4,147 windows.*
*Note: Alkurdi's pipeline has NO corresponding LOSO evaluation.*

| Model Architecture | Pooled Accuracy | Pooled Balanced Acc | Pooled Anxiety F1 | Macro-Subject Accuracy | Macro-Subject Balanced Acc | Macro-Subject Anxiety F1 |
|---|---|---|---|---|---|---|
| **Random Forest** | **84.08%** (0.8408) | 0.6716 | 0.4787 | 0.8415 | 0.5962 | 0.1587 |
| **SVM (RBF)** | **87.41%** (0.8741) | 0.7061 | **0.5584** | **0.8727** | **0.6836** | **0.2103** |
| **XGBoost** | **84.49%** (0.8449) | **0.7078** | 0.5330 | 0.8459 | 0.6216 | 0.1775 |
| **KNN** | **86.35%** (0.8635) | 0.7022 | 0.5421 | 0.8621 | 0.6681 | 0.2011 |

---

### 11.5 Scientific Explanation of the Performance Discrepancy

A casual comparison might suggest that Alkurdi's model performed better (~$94$–$98\%$ accuracy) than the User's model (~$84$–$87\%$ accuracy with moderate Anxiety F1). However, this difference stems entirely from methodology and experimental design, not algorithmic superiority:

1. **Protocol Classification vs. Subjective Affect Prediction:**
   - Alkurdi's model solves a much simpler problem: detecting whether a participant is sitting quietly in Baseline vs. speaking/calculating under social evaluation in TSST. The physical demands of public speaking (speaking aloud, elevated breathing, arm movement) create pronounced physiological shifts across almost all participants.
   - The User's model tackles actual subjective anxiety. Because human psychology is non-uniform, some participants remain calm under public speaking, while others experience high anticipatory or lingering anxiety. Predicting internal affect across people is inherently harder than classifying lab tasks.

2. **Severe Class Imbalance in Psychological Labels:**
   - In Alkurdi's setup, the minority class (Amusement) still comprises $15.6\%$ of the data, and Stress comprises $30.3\%$.
   - In the User's setup, only $17.9\%$ of windows represent Anxiety, and **8 out of 15 participants never experienced anxiety above baseline** during any condition. This zero-inflation heavily penalizes macro-subject metrics (macro Anxiety F1 is $\approx 0.16$–$0.21$ because 8 subjects automatically receive F1 = $0.00$).

3. **Data Leakage & Evaluation Rigor:**
   - Alkurdi used a standard `train_test_split(test_size=0.2, random_state=0)`. Because windows were split randomly across the combined dataset, windows from the *same participant* appeared simultaneously in both the training set and the test set. The model memorized individual physiological baselines (e.g. participant S2's idiosyncratic resting heart rate or skin conductance level), artificially inflating test accuracy to $97\%+$.
   - The User strictly utilized `GroupShuffleSplit` and full Leave-One-Subject-Out (LOSO) cross-validation. In LOSO, the model is trained on $N-1$ subjects and tested on an individual whose physiological signals the model has never encountered before. This provides a realistic, leak-free benchmark for real-world clinical or wearable deployment.
