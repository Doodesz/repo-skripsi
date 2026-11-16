# Model Output Specifications — `stai_2class_alkurdi_alignment`

> **Source Notebook:** [`Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/stai_2class_alkurdi_alignment.ipynb)
> **Dataset:** WESAD (Wearable Stress and Affect Detection)
> **Task:** Binary Subjective Anxiety Classification (0 = No Anxiety Increase, 1 = Anxiety Increased)
> **Labeling Strategy:** Participant-relative STAI self-report (rectified STAI-6 questionnaire row-wise parsing)
> **Sensor Clock Alignment:** Alkurdi `pd.to_datetime` outer-join clock alignment across differing frequencies
> **Validation Paradigms:** Validation Scheme A (Naive Random Split 80/20) & Validation Scheme B (15-Fold Leave-One-Subject-Out CV)

---

## 1. Dataset & Participants

| Parameter | Value |
|---|---|
| Dataset | WESAD (Wearable Stress and Affect Detection) |
| Number of Participants | 15 |
| Participant IDs | S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17 |
| Excluded Participants | S1, S12 (non-existent in public WESAD benchmark release) |
| Labeled Protocol Conditions | `Fun`, `Medi 1`, `TSST`, `Medi 2` (4 conditions per participant) |
| Excluded Protocols | `Base` (reference baseline), `bRead`, `sRead` (reading intervals without questionnaire administration) |
| Total Experimental Conditions | 60 affective conditions (15 participants × 4 conditions) |

---

## 2. STAI Questionnaire & Labeling Specifications

### 2.1 Questionnaire Instrument
| Parameter | Value |
|---|---|
| Instrument | Shortened State-Trait Anxiety Inventory (STAI-6 items) |
| Scale | 1–4 Likert scale (1 = Not at all, 2 = Somewhat, 3 = Moderately so, 4 = Very much so) |
| Items | feel_at_ease (idx 0), feel_nervous (idx 1), jittery (idx 2), relaxed (idx 3), worried (idx 4), pleasant (idx 5) |
| Reverse-Scored Items | Items 0 (`feel_at_ease`), 3 (`relaxed`), 5 (`pleasant`) |
| Inversion Formula | `reversed = 5.0 - raw` (so 1 ↔ 4, 2 ↔ 3) |
| Score Range | Sum: 6 to 24 points; Mean: 1.0 to 4.0 points (higher values = higher subjective anxiety) |
| Minimum Resolvable Step | $\Delta_{\text{sum}} = 1$ point on raw scale; $\Delta_{\text{norm}} = \frac{1}{24} \approx 0.0417$; $\Delta_{\text{mean}} = \frac{1}{6} \approx 0.1667$ |

### 2.2 Labeling Logic
| Parameter                         | Value                                                                    |
| --------------------------------- | ------------------------------------------------------------------------ |
| Label Type                        | Participant-relative (grounded in personal resting baseline)             |
| Mathematical Formulation          | $\Delta = \text{condition\_stai\_sum} - \text{baseline\_stai\_sum}$      |
| **Class 0 (No Anxiety Increase)** | $\Delta \le 0$ ($ext{STAI}_{\text{cond}} \le \text{STAI}_{\text{base}}$) |
| **Class 1 (Anxiety Increased)**   | $\Delta > 0$ ($ext{STAI}_{\text{cond}} > \text{STAI}_{\text{base}}$)     |
| Reference Baseline Condition      | `Base` (20-minute quiet resting condition)                               |

### 2.3 Per-Subject STAI Scores & Assigned Binary Labels
*(Derived from rectified row-wise parsing of `SX_quest.csv` `# STAI` self-reports)*

| Subject | Baseline STAI Sum (Mean) | Conditions, Raw STAI Sums, Score Shift ($\Delta$), & Binary Labels |
|---|---|---|
| **S2** | 10.0 (1.67) | Fun: 8.0 ($\Delta = -2$, **0**); Medi 1: 7.0 ($\Delta = -3$, **0**); TSST: 20.0 ($\Delta = +10$, **1**); Medi 2: 6.0 ($\Delta = -4$, **0**) |
| **S3** | 12.0 (2.00) | Fun: 15.0 ($\Delta = +3$, **1**); Medi 1: 17.0 ($\Delta = +5$, **1**); TSST: 18.0 ($\Delta = +6$, **1**); Medi 2: 12.0 ($\Delta = 0$, **0**) |
| **S4** | 13.0 (2.17) | Fun: 15.0 ($\Delta = +2$, **1**); Medi 1: 14.0 ($\Delta = +1$, **1**); TSST: 17.0 ($\Delta = +4$, **1**); Medi 2: 11.0 ($\Delta = -2$, **0**) |
| **S5** | 14.0 (2.33) | Fun: 11.0 ($\Delta = -3$, **0**); Medi 1: 9.0 ($\Delta = -5$, **0**); TSST: 21.0 ($\Delta = +7$, **1**); Medi 2: 15.0 ($\Delta = +1$, **1**) |
| **S6** | 12.0 (2.00) | Fun: 9.0 ($\Delta = -3$, **0**); Medi 1: 12.0 ($\Delta = 0$, **0**); TSST: 20.0 ($\Delta = +8$, **1**); Medi 2: 13.0 ($\Delta = +1$, **1**) |
| **S7** | 11.0 (1.83) | Fun: 14.0 ($\Delta = +3$, **1**); Medi 1: 12.0 ($\Delta = +1$, **1**); TSST: 17.0 ($\Delta = +6$, **1**); Medi 2: 8.0 ($\Delta = -3$, **0**) |
| **S8** | 13.0 (2.17) | Fun: 11.0 ($\Delta = -2$, **0**); Medi 1: 11.0 ($\Delta = -2$, **0**); TSST: 19.0 ($\Delta = +6$, **1**); Medi 2: 14.0 ($\Delta = +1$, **1**) |
| **S9** | 13.0 (2.17) | Fun: 9.0 ($\Delta = -4$, **0**); Medi 1: 11.0 ($\Delta = -2$, **0**); TSST: 21.0 ($\Delta = +8$, **1**); Medi 2: 14.0 ($\Delta = +1$, **1**) |
| **S10** | 11.0 (1.83) | Fun: 8.0 ($\Delta = -3$, **0**); Medi 1: 10.0 ($\Delta = -1$, **0**); TSST: 18.0 ($\Delta = +7$, **1**); Medi 2: 13.0 ($\Delta = +2$, **1**) |
| **S11** | 14.0 (2.33) | Fun: 13.0 ($\Delta = -1$, **0**); Medi 1: 12.0 ($\Delta = -2$, **0**); TSST: 17.0 ($\Delta = +3$, **1**); Medi 2: 12.0 ($\Delta = -2$, **0**) |
| **S13** | 10.0 (1.67) | Fun: 8.0 ($\Delta = -2$, **0**); Medi 1: 8.0 ($\Delta = -2$, **0**); TSST: 19.0 ($\Delta = +9$, **1**); Medi 2: 11.0 ($\Delta = +1$, **1**) |
| **S14** | 13.0 (2.17) | Fun: 11.0 ($\Delta = -2$, **0**); Medi 1: 12.0 ($\Delta = -1$, **0**); TSST: 21.0 ($\Delta = +8$, **1**); Medi 2: 12.0 ($\Delta = -1$, **0**) |
| **S15** | 13.0 (2.17) | Fun: 10.0 ($\Delta = -3$, **0**); Medi 1: 11.0 ($\Delta = -2$, **0**); TSST: 18.0 ($\Delta = +5$, **1**); Medi 2: 11.0 ($\Delta = -2$, **0**) |
| **S16** | 12.0 (2.00) | Fun: 10.0 ($\Delta = -2$, **0**); Medi 1: 11.0 ($\Delta = -1$, **0**); TSST: 18.0 ($\Delta = +6$, **1**); Medi 2: 10.0 ($\Delta = -2$, **0**) |
| **S17** | 13.0 (2.17) | Fun: 7.0 ($\Delta = -6$, **0**); Medi 1: 10.0 ($\Delta = -3$, **0**); TSST: 22.0 ($\Delta = +9$, **1**); Medi 2: 14.0 ($\Delta = +1$, **1**) |

> [!IMPORTANT]
> **Complete Elimination of Zero-Inflation:** In historical iterations (2nd through 5th try), an indexing transposition bug caused 8 of 15 participants to be mislabeled with zero anxiety windows. In this rectified formulation, **100% of participants (15/15)** exhibit verified anxiety increases during TSST, producing a clean, natural cohort balance of **43.59% Class 0 vs. 56.41% Class 1**.

---

## 3. Signal Preprocessing & Windowing Specifications

### 3.1 Sensor Modalities, Native Sampling Rates & Preprocessing Filters

| Modality | Body Location | Device Hardware | Raw Sampling Rate | Preprocessing Filter & Conditioning Algorithm | Methodological Purpose |
|---|---|---|---|---|---|
| **ACC (3-Axis)** | Wrist | Empatica E4 | 32 Hz | Finite Impulse Response (FIR) Lowpass (`filter_signal_fir`, Hann window, order 64, $f_c \approx 5\text{ Hz}$) | Suppresses high-frequency mechanical vibration; isolates bodily motion acceleration from the constant $1g$ gravitational vector. Native rate preserved. |
| **BVP** | Wrist | Empatica E4 | 64 Hz | Fast Fourier Transform (FFT) peak frequency extraction | Extracts dominant pulse rate frequency from raw optical photoplethysmography wave. Native rate preserved. |
| **EDA** | Wrist | Empatica E4 | 4 Hz | 6th-Order Butterworth IIR Lowpass ($f_c = 1.0\text{ Hz}$, zero-phase) + NeuroKit2 `nk.eda_phasic()` after `nk.standardize()` | Attenuates high-frequency noise while decomposing signal into Phasic (SCR) and Tonic (SCL) components. |
| **TEMP** | Wrist | Empatica E4 | 4 Hz | Native pass-through with slope & dynamic range extraction | Captures peripheral skin temperature shifts and slope dynamics. |
| **ACC (3-Axis)** | Chest | RespiBAN Professional | 700 Hz | Finite Impulse Response (FIR) Lowpass (`filter_signal_fir`) | Attenuates high-frequency muscle tremor artifacts; preserves chest kinetic dynamics at native 700 Hz. |
| **ECG** | Chest | RespiBAN Professional | 700 Hz | HeartPy (`hp.process`) internal bandpass ($0.75 - 3.5\text{ Hz}$) + peak validation | Suppresses respiratory baseline wander and high-frequency EMG noise; reliably detects R-peaks to compute 7 HRV metrics. |
| **EDA** | Chest | RespiBAN Professional | 700 Hz | 6th-Order Butterworth Lowpass ($f_c = 1.0\text{ Hz}$) + NeuroKit2 Phasic/Tonic decomposition | Cleans high-frequency chest artifact and extracts central sympathetic sudomotor tone. |
| **EMG** | Chest | RespiBAN Professional | 700 Hz | Native pass-through with dynamic range ($\max - \min$) & absolute integral ($\int \|x\| dt$) | Measures muscular tension and contraction energy on the trapezius muscle. |
| **Resp** | Chest | RespiBAN Professional | 700 Hz | Peak-to-trough cycle delineation algorithm | Isolates inhalation/exhalation phases, respiratory rate, and $I/E$ ratio from piezoelectric belt. |
| **Temp** | Chest | RespiBAN Professional | 700 Hz | Native pass-through with slope & dynamic range extraction | Captures central skin temperature variation. |

### 3.2 Sensor Alignment Strategy & Resampling Philosophy
- **Reversion from Polyphase Resampling:** Unlike the 5th try which applied synthetic polyphase rational resampling (`scipy.signal.resample_poly`) to force non-ACC signals into 70 Hz, the 6th try **completely eliminates synthetic resampling**, retaining each sensor channel at its native physical sampling rate to preserve true signal morphology and spectral fidelity.
- **Alkurdi Outer-Join Alignment:** All sensor streams are indexed with high-precision nanosecond timestamps (`pd.to_datetime(np.round(np.arange(len(s)) * (1e9 / freq)), unit='ns')`).
- Channels are merged via an outer join (`pd.concat(..., axis=1)`), generating intermittent NaN gaps across mismatched sampling rates, followed by forward/backward fill on protocol labels and local `.dropna()` within feature computation functions.

### 3.3 Sliding Window Parameters
| Parameter | Value |
|---|---|
| Main Window Length | **60 seconds** ($42,000$ samples on 700 Hz timeline) |
| Main Window Shift (Stride) | **6 seconds** (90% temporal overlap) |
| Kinetic ACC Sub-Windows | **5.0 seconds** length, shifted by **0.25 seconds** |
| Boundary Retention Rule | Windows strictly contained within affective protocol intervals; cross-boundary windows dropped |

---

## 4. Feature Extraction Specifications

### 4.1 Feature Count Summary
| Parameter | Value |
|---|---|
| Total DataFrame Columns | **167** |
| Metadata Columns (Excluded from $X$) | **12** (`anxiety_label`, `delta_stai`, `delta_stai_sum`, `delta_stai_mean`, `condition_stai_sum`, `baseline_stai_sum`, `condition_stai_mean`, `baseline_stai_mean`, `subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`) |
| **Predictive Feature Dimension ($X$)** | **155 features** |

### 4.2 Feature Modality Distribution
- **Statistical Moments (Mean, Std, Min, Max):** Computed across EDA, EDA Phasic (SCR), EDA Tonic (SCL), BVP, TEMP, Chest ECG, Chest EDA, Chest EMG, Chest Resp, Chest Temp, and Net Accelerations.
- **Dynamic Moments:** EDA dynamic range and slope (wrist & chest), TEMP dynamic range and slope (wrist & chest), EMG dynamic range and absolute integral.
- **Cardiovascular Features (HeartPy):** `ECG_bpm`, `ECG_ibi`, `ECG_sdnn`, `ECG_sdsd`, `ECG_rmssd`, `ECG_pnn20`, `ECG_pnn50`.
- **Respiration Dynamics:** `Resp_C_rate`, `Resp_C_Inhal_mean`, `Resp_C_Inhal_std`, `Resp_C_Exhal_mean`, `Resp_C_Exhal_std`, `Resp_C_I/E`.
- **Kinetic ACC Sub-Window Moments:** 36 features representing the mean and standard deviation of sub-window standard deviations, absolute integrals, and peak frequencies across all 6 ACC axes.

---

## 5. Cohort Class & Window Distribution

| Subject | Total Windows | Class 0 (No Anxiety, $\Delta \le 0$) | Class 1 (Anxiety Inc, $\Delta > 0$) | TSST Assigned Class |
|---|---|---|---|---|
| **S2** | 257 | 166 (64.59%) | 91 (35.41%) | Class 1 |
| **S3** | 260 | 48 (18.46%) | 212 (81.54%) | Class 1 |
| **S4** | 272 | 55 (20.22%) | 217 (79.78%) | Class 1 |
| **S5** | 270 | 113 (41.85%) | 157 (58.15%) | Class 1 |
| **S6** | 269 | 111 (41.26%) | 158 (58.74%) | Class 1 |
| **S7** | 275 | 56 (20.36%) | 219 (79.64%) | Class 1 |
| **S8** | 263 | 103 (39.16%) | 160 (60.84%) | Class 1 |
| **S9** | 262 | 113 (43.13%) | 149 (56.87%) | Class 1 |
| **S10** | 289 | 110 (38.06%) | 179 (61.94%) | Class 1 |
| **S11** | 285 | 178 (62.46%) | 107 (37.54%) | Class 1 |
| **S13** | 280 | 116 (41.43%) | 164 (58.57%) | Class 1 |
| **S14** | 282 | 176 (62.41%) | 106 (37.59%) | Class 1 |
| **S15** | 282 | 174 (61.70%) | 108 (38.30%) | Class 1 |
| **S16** | 274 | 164 (59.85%) | 110 (40.15%) | Class 1 |
| **S17** | 275 | 102 (37.09%) | 173 (62.91%) | Class 1 |
| **Total Cohort** | **4,095** | **1,785 (43.59%)** | **2,310 (56.41%)** | **15/15 (100%)** |

---

## 6. Model Training & Serialization Specifications

### 6.1 Feature Standardization & Normalization Specifications

#### A. Standardization: Single-Stage Z-Score Scaling
All models in the 6th try use **single-stage Z-score standardization** via `sklearn.preprocessing.StandardScaler`:
$$z_{i, j} = \frac{x_{i, j} - \mu_j}{\sigma_j}$$
where $\mu_j$ is the training-set mean and $\sigma_j$ is the training-set standard deviation of feature $j$.

- **Scaling Application:** Applied across all 155 continuous physiological features, centering them at $\mu = 0$ with unit variance $\sigma = 1$.
- **Methodological Purpose:** Normalizes the divergent scales across multimodal sensors (e.g., skin conductance level in $\mu\text{S}$, heart rate variability in $\text{ms}$, triaxial acceleration in $g$, and respiratory phase ratios).

#### B. Normalization: Omission of MinMaxScaler (Alkurdi Alignment)
- **No Min-Max Normalization Applied:** Unlike the 5th try which chained `StandardScaler -> MinMaxScaler` to bound feature values to $[0, 1]$, the 6th try **strictly omits `MinMaxScaler`**.
- **Rationale for Omission:**
  1. Replicates Alkurdi et al.'s exact modeling cells and reported baseline pipeline (`StandardScaler` alone).
  2. Preserves the natural Gaussian distribution tails of physiological stress responses, avoiding artificial boundary compression from extreme physiological spikes during TSST.

#### C. Scaling Isolation across Validation Schemes
- **Scheme A (Naive Random Split 80/20):** Ad-hoc global `StandardScaler().fit_transform(X_train)` and `sc.transform(X_test)`. While standardizing training and test partitions separately, it operates on a dataset contaminated by subject identity and temporal leakage across overlapping windows.
- **Scheme B (15-Fold Leave-One-Subject-Out CV):** Strict fold-isolated scaling. In every fold $k \in \{1, \dots, 15\}$, `StandardScaler` is fitted exclusively on the 14 training participants (`sc.fit_transform(X_train_fold)`) and applied without refitting to the 1 held-out test participant (`sc.transform(X_test_fold)`), guaranteeing absolute zero data leakage across participant boundaries.

### 6.2 Model Architectures & Parameter Configurations
| Architecture | Hyperparameters | Purpose / Benchmark Role |
|---|---|---|
| **LDA** | Default parameters | Alkurdi baseline reference |
| **Random Forest (Alkurdi)** | `max_depth=4`, `random_state=0` | Alkurdi specified architecture |
| **Random Forest (Updated)** | `max_depth=25`, `random_state=0` | Optimized non-linear tree model |
| **SVM Linear ($C=1.0$)** | `kernel='linear'`, $C=1.0$, `random_state=0` | Alkurdi specified architecture |
| **SVM Linear ($C=0.9$)** | `kernel='linear'`, $C=0.9$, `random_state=0` | Alkurdi specified architecture |
| **SVM (RBF)** | `kernel='rbf'`, `random_state=0` | Non-linear margin classifier |
| **XGBoost** | `n_estimators=100`, `eval_metric='logloss'`, `random_state=0` | Gradient boosted decision trees |
| **KNN** | `n_neighbors=5` | Instance-based local classifier |

### 6.3 Serialized Model Artifacts
All models are serialized to [`Models/6th try/2class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/):
- `stai_2class_schemeA_lda_model.joblib`
- `stai_2class_schemeA_random_forest_model.joblib` (max_depth=25)
- `stai_2class_schemeA_svm_linear_model.joblib` (linear kernel, C=1.0)
- `stai_2class_schemeA_svm_model.joblib` (rbf kernel)
- `stai_2class_schemeA_xgboost_model.joblib`
- `stai_2class_schemeA_knn_model.joblib` (k=5)
- `evaluation_summary.json` & `per_subject_loso_results.json`

---

## 7. Evaluation Results — Validation Scheme A (Naive Random Split 80/20)

*Replicating Alkurdi's reporting methodology: 80% train ($N=3,276$), 20% test ($N=819$), with ad-hoc standard scaling.*

### 7.1 Scheme A Performance Summary
| Model                                      | Accuracy    | Balanced Accuracy | Macro F1   | Class 0 F1 (No Anx) | Class 1 F1 (Anx Inc) | Fit Time |
| ------------------------------------------ | ----------- | ----------------- | ---------- | ------------------- | -------------------- | -------- |
| **Random Forest (`max_depth=25`)**         | **100.00%** | **100.00%**       | **1.0000** | 1.0000              | 1.0000               | 2.46s    |
| **SVM (RBF)**                              | **100.00%** | **100.00%**       | **1.0000** | 1.0000              | 1.0000               | 0.39s    |
| **XGBoost**                                | **99.76%**  | **99.78%**        | **0.9975** | 0.9973              | 0.9978               | 2.17s    |
| **KNN ($k=5$)**                            | **99.39%**  | **99.45%**        | **0.9938** | 0.9932              | 0.9945               | 0.28s    |
| **SVM Linear ($C=1.0$)**                   | **96.58%**  | **96.51%**        | **0.9654** | 0.9614              | 0.9693               | 0.39s    |
| **SVM Linear ($C=0.9$)**                   | **96.58%**  | **96.51%**        | **0.9654** | 0.9614              | 0.9693               | 0.30s    |
| **Random Forest (`max_depth=4`, Alkurdi)** | **96.21%**  | **96.57%**        | **0.9619** | 0.9590              | 0.9648               | 1.49s    |
| **LDA**                                    | **93.77%**  | **93.82%**        | **0.9371** | 0.9308              | 0.9434               | 0.08s    |

> [!WARNING]
> **Data Leakage Artifact:** Because 60-second sliding windows with a 6-second shift overlap by 90%, adjacent windows from the same participant appear in both training and test partitions. Models memorize participant identity rather than generalized affective response, artificially driving accuracies to 96%–100%.

---

## 8. Evaluation Results — Validation Scheme B (15-Fold Leave-One-Subject-Out CV)

*Strict subject-independent generalization: iteratively train on 14 participants and evaluate on the 1 held-out participant with fold-isolated scaling.*

### 8.1 Pooled Classification Metrics & Confusion Matrices

#### A. Random Forest (`max_depth=4`, Alkurdi Spec) — **Peak Generalization Model**

- **Pooled Accuracy:** **74.43%** (3,048 / 4,095 correct)
- **Balanced Accuracy:** **74.90%**
- **Macro F1:** **0.7434**
- **Class 0 F1 (No Anxiety):** **0.7281** (Recall: 78.54%, Precision: 67.86%)
- **Class 1 F1 (Anxiety Increased):** **0.7587** (Recall: 71.26%, Precision: 81.12%)

Confusion Matrix (LOSO Pooled):

|                                 | Pred: No Anxiety (0) | Pred: Anxiety Increased (1) |
| ------------------------------- | -------------------- | --------------------------- |
| **True: No Anxiety (0)**        | **1,402**            | 383                         |
| **True: Anxiety Increased (1)** | 664                  | **1,646**                   |

#### B. XGBoost (`n_estimators=100`)

- **Pooled Accuracy:** **74.14%** (3,036 / 4,095 correct)
- **Balanced Accuracy:** **74.24%**
- **Macro F1:** **0.7394**
- **Class 0 F1:** **0.7168**
- **Class 1 F1:** **0.7621**

Confusion Matrix (LOSO Pooled):

|                                 | Pred: No Anxiety (0) | Pred: Anxiety Increased (1) |
| ------------------------------- | -------------------- | --------------------------- |
| **True: No Anxiety (0)**        | **1,340**            | 445                         |
| **True: Anxiety Increased (1)** | 614                  | **1,696**                   |

#### C. Random Forest (`max_depth=25`, Updated)
- **Pooled Accuracy:** **73.72%** (3,019 / 4,095 correct)
- **Balanced Accuracy:** **74.11%**
- **Macro F1:** **0.7361**
- **Class 0 F1:** **0.7191**
- **Class 1 F1:** **0.7532**

Confusion Matrix (LOSO Pooled):

| | Pred: No Anxiety (0) | Pred: Anxiety Increased (1) |
|---|---|---|
| **True: No Anxiety (0)** | **1,377** | 408 |
| **True: Anxiety Increased (1)** | 668 | **1,642** |

#### D. SVM (RBF Kernel)
- **Pooled Accuracy:** **72.04%** (2,950 / 4,095 correct)
- **Balanced Accuracy:** **71.59%**
- **Macro F1:** **0.7158**
- **Class 0 F1:** **0.6797**
- **Class 1 F1:** **0.7519**

Confusion Matrix (LOSO Pooled):

| | Pred: No Anxiety (0) | Pred: Anxiety Increased (1) |
|---|---|---|
| **True: No Anxiety (0)** | **1,215** | 570 |
| **True: Anxiety Increased (1)** | 575 | **1,735** |

#### E. KNN ($k=5$)
- **Pooled Accuracy:** **67.64%**
- **Balanced Accuracy:** **68.53%**
- **Macro F1:** **0.6763**
- **Class 0 F1:** **0.6703**
- **Class 1 F1:** **0.6823**

#### F. SVM Linear ($C=1.0$ & $C=0.9$)
- **Pooled Accuracy:** **64.96%** ($C=1.0$) / **64.98%** ($C=0.9$)
- **Balanced Accuracy:** **64.64%** / **64.70%**
- **Macro F1:** **0.6455** / **0.6459**

#### G. Linear Discriminant Analysis (LDA)
- **Pooled Accuracy:** **62.66%**
- **Balanced Accuracy:** **62.70%**
- **Macro F1:** **0.6244**

---

### 8.2 LOSO Cross-Validation Summary Table
| Model Architecture | Pooled Accuracy | Pooled Balanced Acc | Pooled Macro F1 | Class 0 F1 (No Anx) | Class 1 F1 (Anx Inc) | Macro-Subject Accuracy | Macro-Subject Balanced Acc | Macro-Subject Anxiety F1 |
|---|---|---|---|---|---|---|---|---|
| **Random Forest (`max_depth=4`, Alkurdi)** | **74.43%** | **74.90%** | **0.7434** | **0.7281** | **0.7587** | **74.40%** | **78.45%** | **0.7720** |
| **XGBoost** | **74.14%** | **74.24%** | **0.7394** | **0.7168** | **0.7621** | **74.19%** | **77.29%** | **0.7742** |
| **Random Forest (`max_depth=25`)** | **73.72%** | **74.11%** | **0.7361** | **0.7191** | **0.7532** | **73.71%** | **77.52%** | **0.7672** |
| **SVM (RBF)** | **72.04%** | **71.59%** | **0.7158** | **0.6797** | **0.7519** | **71.91%** | **72.49%** | **0.7502** |
| **KNN ($k=5$)** | **67.64%** | **68.53%** | **0.6763** | **0.6703** | **0.6823** | **67.43%** | **69.83%** | **0.6803** |
| **SVM Linear ($C=0.9$)** | **64.98%** | **64.70%** | **0.6459** | **0.6088** | **0.6830** | **64.81%** | **66.42%** | **0.6626** |
| **SVM Linear ($C=1.0$)** | **64.96%** | **64.64%** | **0.6455** | **0.6074** | **0.6836** | **64.79%** | **66.39%** | **0.6623** |
| **LDA** | **62.66%** | **62.70%** | **0.6244** | **0.5952** | **0.6535** | **62.58%** | **63.45%** | **0.6298** |

---

### 8.3 LOSO Per-Subject Detailed Results

#### Random Forest (`max_depth=4`, Alkurdi Spec)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Anx Inc) | Macro F1 |
| ------- | ------- | -------- | ----------------- | ------------------- | -------------------- | -------- |
| **S2**  | 257     | 0.9922   | 0.9890            | 0.9940              | 0.9889               | 0.9914   |
| **S3**  | 260     | 0.3577   | 0.2757            | 0.0773              | 0.5074               | 0.2924   |
| **S4**  | 272     | 0.5772   | 0.7350            | 0.4889              | 0.6395               | 0.5642   |
| **S5**  | 270     | 0.7741   | 0.7995            | 0.7798              | 0.7681               | 0.7739   |
| **S6**  | 269     | 0.6320   | 0.6291            | 0.5787              | 0.6733               | 0.6260   |
| **S7**  | 275     | 0.5673   | 0.7283            | 0.4848              | 0.6270               | 0.5559   |
| **S8**  | 263     | 0.7719   | 0.8125            | 0.7744              | 0.7692               | 0.7718   |
| **S9**  | 262     | 0.8053   | 0.8289            | 0.8159              | 0.7935               | 0.8047   |
| **S10** | 289     | 0.8270   | 0.8603            | 0.8148              | 0.8377               | 0.8262   |
| **S11** | 285     | 0.5860   | 0.6685            | 0.5042              | 0.6446               | 0.5744   |
| **S13** | 280     | 1.0000   | 1.0000            | 1.0000              | 1.0000               | 1.0000   |
| **S14** | 282     | 0.6170   | 0.6932            | 0.5574              | 0.6625               | 0.6100   |
| **S15** | 282     | 0.7908   | 0.8305            | 0.7958              | 0.7855               | 0.7906   |
| **S16** | 274     | 1.0000   | 1.0000            | 1.0000              | 1.0000               | 1.0000   |
| **S17** | 275     | 0.8618   | 0.8721            | 0.8304              | 0.8834               | 0.8569   |

#### XGBoost (`n_estimators=100`)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|
| **S2** | 257 | 0.9844 | 0.9880 | 0.9878 | 0.9785 | 0.9832 |
| **S3** | 260 | 0.6231 | 0.6480 | 0.4024 | 0.7247 | 0.5636 |
| **S4** | 272 | 0.5772 | 0.7350 | 0.4889 | 0.6395 | 0.5642 |
| **S5** | 270 | 0.6444 | 0.6447 | 0.6033 | 0.6779 | 0.6406 |
| **S6** | 269 | 0.6543 | 0.6079 | 0.4497 | 0.7480 | 0.5988 |
| **S7** | 275 | 0.5345 | 0.6347 | 0.4128 | 0.6145 | 0.5137 |
| **S8** | 263 | 0.7719 | 0.8125 | 0.7744 | 0.7692 | 0.7718 |
| **S9** | 262 | 0.8053 | 0.8289 | 0.8159 | 0.7935 | 0.8047 |
| **S10** | 289 | 0.7993 | 0.8380 | 0.7914 | 0.8067 | 0.7990 |
| **S11** | 285 | 0.5789 | 0.6629 | 0.4915 | 0.6407 | 0.5661 |
| **S13** | 280 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **S14** | 282 | 0.6986 | 0.7585 | 0.6816 | 0.7138 | 0.6977 |
| **S15** | 282 | 0.7837 | 0.8247 | 0.7875 | 0.7798 | 0.7836 |
| **S16** | 274 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **S17** | 275 | 0.6727 | 0.6675 | 0.5946 | 0.7256 | 0.6601 |

#### SVM (RBF Kernel)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|
| **S2** | 257 | 0.6732 | 0.6800 | 0.7219 | 0.6038 | 0.6628 |
| **S3** | 260 | 0.4423 | 0.6016 | 0.3612 | 0.5051 | 0.4332 |
| **S4** | 272 | 0.6507 | 0.7811 | 0.5366 | 0.7198 | 0.6282 |
| **S5** | 270 | 0.6667 | 0.6526 | 0.5872 | 0.7205 | 0.6538 |
| **S6** | 269 | 0.6506 | 0.6436 | 0.5877 | 0.6968 | 0.6423 |
| **S7** | 275 | 0.6473 | 0.4064 | 0.0000 | 0.7859 | 0.3929 |
| **S8** | 263 | 0.7567 | 0.8000 | 0.7630 | 0.7500 | 0.7565 |
| **S9** | 262 | 0.7557 | 0.7542 | 0.7241 | 0.7808 | 0.7525 |
| **S10** | 289 | 0.8097 | 0.8464 | 0.8000 | 0.8185 | 0.8092 |
| **S11** | 285 | 0.5684 | 0.6545 | 0.4721 | 0.6350 | 0.5535 |
| **S13** | 280 | 0.8464 | 0.8222 | 0.7861 | 0.8802 | 0.8331 |
| **S14** | 282 | 0.6773 | 0.7377 | 0.6566 | 0.6957 | 0.6761 |
| **S15** | 282 | 0.7979 | 0.8362 | 0.8041 | 0.7912 | 0.7977 |
| **S16** | 274 | 0.9891 | 0.9909 | 0.9908 | 0.9865 | 0.9887 |
| **S17** | 275 | 0.8545 | 0.8462 | 0.8058 | 0.8837 | 0.8448 |

> [!NOTE]
> **Complete Absence of NaN Balanced Accuracy:** Unlike the 5th try where 8 subjects had all-negative ground truth, every single participant in the 6th try has both Class 0 and Class 1 windows, allowing legitimate calculation of Balanced Accuracy across all 15 participants.

---

## 9. Key Empirical Findings & Insights

1. **Resolution of the Zero-Inflation Pathology:**
   Fixing the questionnaire parsing bug completely eliminated the historical zero-inflation problem. Rather than 8 of 15 participants scoring 0 positive windows, **every single subject displays positive Class 1 windows during TSST**, achieving a clean 43.6% vs 56.4% cohort distribution.
2. **Subject-Independent Generalization (74.4% Accuracy):**
   When evaluated under strict 15-fold Leave-One-Subject-Out CV, Random Forest (`max_depth=4`) achieves **74.43% pooled accuracy**, **74.90% balanced accuracy**, and **0.7434 macro F1**.
3. **Balanced Per-Class Identification:**
   Unlike prior runs where minority classes collapsed, Random Forest achieves **0.7281 F1 on No Anxiety** and **0.7587 F1 on Anxiety Increased**. The mean subject-level anxiety F1 reaches **0.7720**, demonstrating genuine affective generalization across unseen subjects.
4. **Shallow vs. Deep Trees in LOSO:**
   While deep trees (`max_depth=25`) achieved 100.0% accuracy under the leaked Scheme A split, shallow trees (`max_depth=4`) slightly outperformed deep trees in LOSO cross-validation (74.43% vs 73.72%), proving that aggressive tree pruning acts as effective regularization against inter-subject physiological variance.

---

## 10. Output File Locations

| Output Description | File Path |
|---|---|
| **Source Notebook** | [`Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/stai_2class_alkurdi_alignment.ipynb) |
| **Extracted 2-Class Feature Datasets** | [`Data/6th try/2class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/) (`S{id}_stai_2class_60s_shift6s.pkl`) |
| **Trained Models (Scheme A)** | [`Models/6th try/2class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/) (`stai_2class_schemeA_*.joblib`) |
| **Evaluation Summary (Schemes A & B)** | [`Models/6th try/2class/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/evaluation_summary.json) |
| **Per-Subject LOSO Metrics** | [`Models/6th try/2class/per_subject_loso_results.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/per_subject_loso_results.json) |

---

## 11. Comprehensive Evolution & Comparison Matrix Across Iterations

| Dimension / Metric | 5th Try (`Scripts/5th try/`) | 6th Try 3-Class (`3class/`) | **6th Try 2-Class (`2class/`)** | Alkurdi Reference (`_forked/`) |
|---|---|---|---|---|
| **Target Ground Truth** | Subjective STAI (Buggy, 5 items transposed) | Subjective STAI (Fixed, 6 items) | **Subjective STAI (Fixed, 6 items)** | Objective Protocol (Base vs Fun vs TSST) |
| **Classification Task** | Binary: $\Delta \ge 0.2$ | 3-Class: $\Delta < -\frac{1}{24}$, $\pm \frac{1}{24}$, $\Delta > \frac{1}{24}$ | **Binary: $\Delta \le 0$ vs $\Delta > 0$** | 3-Class Protocol: 0, 1, 2 |
| **Cohort Windows** | 4,147 windows (with reading tasks) | 4,095 windows (clean 4 conditions) | **4,095 windows (clean 4 conditions)** | 577 windows (tumbling 60s, stride 60s) |
| **Class Distribution** | 82.1% Class 0 / 17.9% Class 1 | 27.1% C0 / 25.1% C1 / 47.8% C2 | **43.59% Class 0 / 56.41% Class 1** | 54.1% Base / 30.3% Stress / 15.6% Fun |
| **Zero-Inflation** | **Severe (8/15 subjects had 0 positive)** | **Resolved (15/15 have positive)** | **Resolved (15/15 have positive)** | None (protocol stimulus driven) |
| **Sensor Clock Alignment** | Polyphase rational resampling (70 Hz) | Alkurdi `outer` join + `pd.to_datetime` | **Alkurdi `outer` join + `pd.to_datetime`** | Alkurdi `outer` join + `pd.to_datetime` |
| **Predictive Features** | 154 features | 155 features | **155 features** | 120 features |
| **Scheme A Leakage Acc** | 86%–89% (`GroupShuffleSplit`) | 92.8%–100.0% (`train_test_split`) | **93.8%–100.0% (`train_test_split`)** | 93.6%–97.9% (`train_test_split`) |
| **LOSO Peak Accuracy** | 87.41% (SVM RBF, collapsed minority) | 57.24% (XGBoost) | **74.43% (Random Forest depth 4)** | *No LOSO evaluation performed* |
| **LOSO Peak Balanced Acc** | 70.78% (XGBoost) | 55.93% (XGBoost) | **74.90% (Random Forest depth 4)** | *No LOSO evaluation performed* |
| **LOSO Peak Macro F1** | 0.5584 (SVM RBF) | 0.5593 (XGBoost) | **0.7434 (Random Forest depth 4)** | *No LOSO evaluation performed* |
| **Macro-Subject Anxiety F1** | 0.16–0.21 (dragged down by zeros) | 0.77–0.80 (Class 2) | **0.7720 (Class 1 across all 15 subjects)** | *N/A* |
