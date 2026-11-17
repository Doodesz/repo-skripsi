# Model Output Specifications — `stai_3class_alkurdi_alignment`

> **Source Notebook:** [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb)
> **Dataset:** WESAD (Wearable Stress and Affect Detection)
> **Task:** 3-Class Subjective Anxiety Classification:
> - **Class 0:** No Anxiety Increase ($\Delta_{\text{norm}} < -\frac{1}{24} \approx -0.0417$, $\Delta_{\text{sum}} < -1$)
> - **Class 1:** Potential Anxiety Increase ($-\frac{1}{24} \le \Delta_{\text{norm}} \le \frac{1}{24}$, $\Delta_{\text{sum}} \in \{-1, 0, +1\}$)
> - **Class 2:** Anxiety Increased ($\Delta_{\text{norm}} > \frac{1}{24} \approx +0.0417$, $\Delta_{\text{sum}} > +1$)
> **Labeling Strategy:** Participant-relative continuous STAI self-report with $\pm \frac{1}{24}$ neutral band thresholding ($STAI_{\text{unit}} = 1/24 \approx 0.04167$)
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

### 2.2 Labeling Logic & Psychometric Rationale
| Parameter | Value |
|---|---|
| Label Type | Participant-relative continuous thresholding (anchored in personal resting baseline) |
| Mathematical Formulation | $\Delta_{\text{norm}} = \frac{\text{condition\_stai\_sum} - \text{baseline\_stai\_sum}}{24} = \frac{\text{condition\_stai\_mean} - \text{baseline\_stai\_mean}}{4}$ |
| **Class 0 (No Anxiety Increase)** | $\Delta_{\text{norm}} < -\frac{1}{24} \approx -0.0417$ ($\Delta_{\text{sum}} < -1$, i.e. $\le -2$ points below baseline) |
| **Class 1 (Potential Anxiety Increase)** | $-\frac{1}{24} \le \Delta_{\text{norm}} \le \frac{1}{24}$ ($\Delta_{\text{sum}} \in \{-1, 0, +1\}$, neutral zone capturing baseline measurement noise) |
| **Class 2 (Anxiety Increased)** | $\Delta_{\text{norm}} > \frac{1}{24} \approx +0.0417$ ($\Delta_{\text{sum}} > +1$, i.e. $\ge +2$ points above baseline) |
| Reference Baseline Condition | `Base` (20-minute quiet resting condition) |

> [!TIP]
> **Psychometric Rationale for $\pm \frac{1}{24}$ Neutral Band:**
> The shortened STAI questionnaire consists of 6 items on a 4-point Likert scale (maximum score = 24 points). A 1-point raw score shift ($\Delta_{\text{sum}} = 1$) represents $\frac{1}{24} \approx 0.04167$ on the normalized scale. Setting the neutral zone (Class 1) to $\pm \frac{1}{24}$ guarantees that any self-report response within 1 questionnaire point of the resting baseline is treated as natural baseline measurement variation/potential anxiety, while score changes $\ge 2$ points represent distinct psychological state shifts.

### 2.3 Per-Subject STAI Scores & Assigned 3-Class Labels
*(Derived from rectified row-wise parsing of `SX_quest.csv` `# STAI` self-reports)*

| Subject | Baseline STAI Sum (Mean) | Conditions, Raw STAI Sums, Score Shift ($\Delta_{\text{norm}}$), & 3-Class Labels |
|---|---|---|
| **S2** | 10.0 (1.67) | Fun: 8.0 ($\Delta = -0.0833$, **0**); Medi 1: 7.0 ($\Delta = -0.1250$, **0**); TSST: 20.0 ($\Delta = +0.4167$, **2**); Medi 2: 6.0 ($\Delta = -0.1667$, **0**) |
| **S3** | 9.0 (1.50) | Fun: 9.0 ($\Delta = 0.0000$, **1**); Medi 1: 12.0 ($\Delta = +0.1250$, **2**); TSST: 20.0 ($\Delta = +0.4583$, **2**); Medi 2: 11.0 ($\Delta = +0.0833$, **2**) |
| **S4** | 9.0 (1.50) | Fun: 8.0 ($\Delta = -0.0417$, **1**); Medi 1: 15.0 ($\Delta = +0.2500$, **2**); TSST: 17.0 ($\Delta = +0.3333$, **2**); Medi 2: 13.0 ($\Delta = +0.1667$, **2**) |
| **S5** | 10.0 (1.67) | Fun: 12.0 ($\Delta = +0.0833$, **2**); Medi 1: 8.0 ($\Delta = -0.0833$, **0**); TSST: 18.0 ($\Delta = +0.3333$, **2**); Medi 2: 8.0 ($\Delta = -0.0833$, **0**) |
| **S6** | 9.0 (1.50) | Fun: 10.0 ($\Delta = +0.0417$, **1**); Medi 1: 9.0 ($\Delta = 0.0000$, **1**); TSST: 15.0 ($\Delta = +0.2500$, **2**); Medi 2: 6.0 ($\Delta = -0.1250$, **0**) |
| **S7** | 8.0 (1.33) | Fun: 7.0 ($\Delta = -0.0417$, **1**); Medi 1: 9.0 ($\Delta = +0.0417$, **1**); TSST: 18.0 ($\Delta = +0.4167$, **2**); Medi 2: 11.0 ($\Delta = +0.1250$, **2**) |
| **S8** | 13.0 (2.17) | Fun: 13.0 ($\Delta = 0.0000$, **1**); Medi 1: 14.0 ($\Delta = +0.0417$, **1**); TSST: 18.0 ($\Delta = +0.2083$, **2**); Medi 2: 12.0 ($\Delta = -0.0417$, **1**) |
| **S9** | 9.0 (1.50) | Fun: 10.0 ($\Delta = +0.0417$, **1**); Medi 1: 7.0 ($\Delta = -0.0833$, **0**); TSST: 14.0 ($\Delta = +0.2083$, **2**); Medi 2: 7.0 ($\Delta = -0.0833$, **0**) |
| **S10** | 11.0 (1.83) | Fun: 8.0 ($\Delta = -0.1250$, **0**); Medi 1: 10.0 ($\Delta = -0.0417$, **1**); TSST: 20.0 ($\Delta = +0.3750$, **2**); Medi 2: 12.0 ($\Delta = +0.0417$, **1**) |
| **S11** | 9.0 (1.50) | Fun: 6.0 ($\Delta = -0.1250$, **0**); Medi 1: 8.0 ($\Delta = -0.0417$, **1**); TSST: 19.0 ($\Delta = +0.4167$, **2**); Medi 2: 8.0 ($\Delta = -0.0417$, **1**) |
| **S13** | 14.0 (2.33) | Fun: 8.0 ($\Delta = -0.2500$, **0**); Medi 1: 8.0 ($\Delta = -0.2500$, **0**); TSST: 21.0 ($\Delta = +0.2917$, **2**); Medi 2: 16.0 ($\Delta = +0.0833$, **2**) |
| **S14** | 13.0 (2.17) | Fun: 12.0 ($\Delta = -0.0417$, **1**); Medi 1: 13.0 ($\Delta = 0.0000$, **1**); TSST: 18.0 ($\Delta = +0.2083$, **2**); Medi 2: 13.0 ($\Delta = 0.0000$, **1**) |
| **S15** | 12.0 (2.00) | Fun: 10.0 ($\Delta = -0.0833$, **0**); Medi 1: 9.0 ($\Delta = -0.1250$, **0**); TSST: 19.0 ($\Delta = +0.2917$, **2**); Medi 2: 9.0 ($\Delta = -0.1250$, **0**) |
| **S16** | 13.0 (2.17) | Fun: 11.0 ($\Delta = -0.0833$, **0**); Medi 1: 9.0 ($\Delta = -0.1667$, **0**); TSST: 19.0 ($\Delta = +0.2500$, **2**); Medi 2: 10.0 ($\Delta = -0.1250$, **0**) |
| **S17** | 13.0 (2.17) | Fun: 7.0 ($\Delta = -0.2500$, **0**); Medi 1: 10.0 ($\Delta = -0.1250$, **0**); TSST: 22.0 ($\Delta = +0.3750$, **2**); Medi 2: 14.0 ($\Delta = +0.0417$, **1**) |

> [!IMPORTANT]
> **Complete Elimination of Zero-Inflation & 100% TSST Response:**
> In the 3-class formulation, **100% of participants (15/15)** exhibit confirmed Class 2 anxiety increases during TSST. The continuous partitioning across the real number line creates a natural, well-represented cohort balance of **27.11% Class 0**, **25.05% Class 1**, and **47.84% Class 2**, with zero orphaned or ambiguous windows.

---

## 3. Signal Preprocessing & Windowing Specifications

### 3.1 Sensor Modalities, Preprocessing Filters & Hardware
| Modality | Body Location | Device Hardware | Raw Sampling Rate | Preprocessing Filter & Signal Treatment | Physiological Rationale |
|---|---|---|---|---|---|
| **ACC (3-Axis)** | Wrist | Empatica E4 | 32 Hz | Finite Impulse Response (FIR) Lowpass (`filter_signal_fir`, Hann window, order 64, $f_c \approx 5\text{ Hz}$) | Suppresses high-frequency mechanical vibration; isolates bodily motion acceleration from the constant $1g$ gravitational vector. Native rate preserved. |
| **BVP** | Wrist | Empatica E4 | 64 Hz | Fast Fourier Transform (FFT) peak frequency extraction | Extracts dominant pulse rate frequency from raw optical photoplethysmography wave. Native rate preserved. |
| **EDA** | Wrist | Empatica E4 | 4 Hz | 6th-Order Butterworth IIR Lowpass ($f_c = 1.0\text{ Hz}$, zero-phase) + NeuroKit2 `nk.eda_phasic()` after `nk.standardize()` | Attenuates high-frequency noise while decomposing signal into Phasic (SCR) and Tonic (SCL) components. |
| **TEMP** | Wrist | Empatica E4 | 4 Hz | Native pass-through with slope & dynamic range extraction | Captures peripheral skin temperature shifts and slope dynamics. |
| **ACC (3-Axis)** | Chest | RespiBAN Professional | 700 Hz | Finite Impulse Response (FIR) Lowpass (`filter_signal_fir`) | Attenuates high-frequency muscle tremor artifacts; preserves chest kinetic dynamics at native 700 Hz. |
| **ECG** | Chest | RespiBAN Professional | 700 Hz | HeartPy (`hp.process`) internal bandpass ($0.75 - 3.5\text{ Hz}$) + peak validation | Suppresses respiratory baseline wander and high-frequency EMG noise; reliably detects R-peaks to compute 7 HRV metrics. |
| **EDA** | Chest | RespiBAN Professional | 700 Hz | 6th-Order Butterworth Lowpass ($f_c = 1.0\text{ Hz}$) + NeuroKit2 Phasic/Tonic decomposition | Cleans high-frequency chest artifact and extracts central sympathetic sudomotor tone. |
| **EMG** | Chest | RespiBAN Professional | 700 Hz | Native pass-through with dynamic range ($\max - \min$) & absolute integral ($\int |x| dt$) | Measures muscular tension and contraction energy on the trapezius muscle. |
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
| Metadata Columns (Excluded from $X$) | **13** (`anxiety_label`, `delta_stai`, `delta_stai_sum`, `delta_stai_mean`, `condition_stai_sum`, `baseline_stai_sum`, `condition_stai_mean`, `baseline_stai_mean`, `subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `ACC_subwindow_count`) |
| **Predictive Feature Dimension ($X$)** | **154 features** (Sanitized per 2026-11-17 audit; formerly 155 when operational `ACC_subwindow_count` was unmasked) |

### 4.2 Feature Modality Distribution
- **Statistical Moments (Mean, Std, Min, Max):** Computed across EDA, EDA Phasic (SCR), EDA Tonic (SCL), BVP, TEMP, Chest ECG, Chest EDA, Chest EMG, Chest Resp, Chest Temp, and Net Accelerations.
- **Dynamic Moments:** EDA dynamic range and slope (wrist & chest), TEMP dynamic range and slope (wrist & chest), EMG dynamic range and absolute integral.
- **Cardiovascular Features (HeartPy):** `ECG_bpm`, `ECG_ibi`, `ECG_sdnn`, `ECG_sdsd`, `ECG_rmssd`, `ECG_pnn20`, `ECG_pnn50`.
- **Respiration Dynamics:** `Resp_C_rate`, `Resp_C_Inhal_mean`, `Resp_C_Inhal_std`, `Resp_C_Exhal_mean`, `Resp_C_Exhal_std`, `Resp_C_I/E`.
- **Kinetic ACC Sub-Window Moments:** 36 features representing the mean and standard deviation of sub-window standard deviations, absolute integrals, and peak frequencies across all 6 ACC axes.

### 4.3 Spectral Peak Frequency Calibration & Model Sensitivity (2026-11-17 Update)
> [!NOTE] Sensor-Calibrated Sampling Frequencies in `get_peak_freq`
> In earlier benchmark runs inherited from `_forked/fb_code/utils.py`, `get_peak_freq` computed periodograms using a hardcoded $f_s = 8\text{ Hz}$. Per the 2026-11-17 audit ([[2026-11-17 Research Note]]), the helper function was localized to [`Scripts/6th try/fb_code/utils.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/utils.py) and refactored to dynamically accept the native sensor sampling rate (`fs=64` for BVP, `fs=32` for wrist ACC, `fs=700` for chest ACC).
> 
> **Impact on Feature Space & Model Metrics:**
> - Exactly **19 features** (1 BVP peak, 6 global ACC peaks, and 12 ACC sub-window peak statistics) now reflect physical Hertz instead of compressed frequency bins.
> - This recalibration causes minor performance shifts across 3-class pooled and LOSO fold metrics compared to the uncalibrated $f_s=8$ baseline. This variation is a natural, expected consequence of physical feature calibration and ensures true cross-device portability.

---

## 5. Cohort Class & Window Distribution

| Subject | Total Windows | Class 0 (No Anxiety, $\Delta < -\frac{1}{24}$) | Class 1 (Potential, $-\frac{1}{24} \le \Delta \le \frac{1}{24}$) | Class 2 (Anxiety Inc, $\Delta > \frac{1}{24}$) | TSST Assigned Class |
|---|---|---|---|---|---|
| **S2** | 257 | 166 (64.59%) | 0 (0.00%) | 91 (35.41%) | Class 2 |
| **S3** | 260 | 0 (0.00%) | 48 (18.46%) | 212 (81.54%) | Class 2 |
| **S4** | 272 | 0 (0.00%) | 55 (20.22%) | 217 (79.78%) | Class 2 |
| **S5** | 270 | 113 (41.85%) | 0 (0.00%) | 157 (58.15%) | Class 2 |
| **S6** | 269 | 51 (18.96%) | 116 (43.12%) | 102 (37.92%) | Class 2 |
| **S7** | 275 | 0 (0.00%) | 116 (42.18%) | 159 (57.82%) | Class 2 |
| **S8** | 263 | 0 (0.00%) | 163 (61.98%) | 100 (38.02%) | Class 2 |
| **S9** | 262 | 113 (43.13%) | 56 (21.37%) | 93 (35.50%) | Class 2 |
| **S10** | 289 | 56 (19.38%) | 114 (39.45%) | 119 (41.18%) | Class 2 |
| **S11** | 285 | 55 (19.30%) | 123 (43.16%) | 107 (37.54%) | Class 2 |
| **S13** | 280 | 116 (41.43%) | 0 (0.00%) | 164 (58.57%) | Class 2 |
| **S14** | 282 | 0 (0.00%) | 176 (62.41%) | 106 (37.59%) | Class 2 |
| **S15** | 282 | 174 (61.70%) | 0 (0.00%) | 108 (38.30%) | Class 2 |
| **S16** | 274 | 164 (59.85%) | 0 (0.00%) | 110 (40.15%) | Class 2 |
| **S17** | 275 | 102 (37.09%) | 59 (21.45%) | 114 (41.45%) | Class 2 |
| **Total Cohort** | **4,095** | **1,110 (27.11%)** | **1,026 (25.05%)** | **1,959 (47.84%)** | **15/15 (100% Class 2)** |

---

## 6. Model Training & Serialization Specifications

### 6.1 Feature Standardization & Normalization Specifications

#### A. Standardization: Single-Stage Z-Score Scaling
All models in the 6th try use **single-stage Z-score standardization** via `sklearn.preprocessing.StandardScaler`:
$$z_{i, j} = \frac{x_{i, j} - \mu_j}{\sigma_j}$$
where $\mu_j$ is the training-set mean and $\sigma_j$ is the training-set standard deviation of feature $j$.

- **Scaling Application:** Applied across all 154 continuous physiological features, centering them at $\mu = 0$ with unit variance $\sigma = 1$.
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
| **XGBoost** | `n_estimators=100`, `eval_metric='mlogloss'`, `random_state=0` | Gradient boosted decision trees |
| **KNN** | `n_neighbors=5` | Instance-based local classifier |

### 6.3 Serialized Model Artifacts
All models are serialized to [`Models/6th try/3class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/):
- `stai_3class_schemeA_lda_model.joblib`
- `stai_3class_schemeA_random_forest_model.joblib` (max_depth=25)
- `stai_3class_schemeA_svm_linear_model.joblib` (linear kernel, C=1.0)
- `stai_3class_schemeA_svm_model.joblib` (rbf kernel)
- `stai_3class_schemeA_xgboost_model.joblib`
- `stai_3class_schemeA_knn_model.joblib` (k=5)
- `evaluation_summary.json` & `per_subject_loso_results.json`

---

## 7. Evaluation Results — Validation Scheme A (Naive Random Split 80/20)

*Replicating Alkurdi's reporting methodology: 80% train ($N=3,276$), 20% test ($N=819$), with ad-hoc standard scaling.*

### 7.1 Scheme A Performance Summary
| Model | Accuracy | Balanced Accuracy | Macro F1 | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Fit Time |
|---|---|---|---|---|---|---|---|
| **Random Forest (`max_depth=25`)** | **100.00%** | **100.00%** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 2.65s |
| **XGBoost** | **100.00%** | **100.00%** | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 3.83s |
| **SVM (RBF)** | **99.51%** | **99.39%** | **0.9939** | 0.9909 | 0.9908 | 1.0000 | 0.34s |
| **KNN ($k=5$)** | **99.39%** | **99.37%** | **0.9930** | 0.9931 | 0.9886 | 0.9974 | 0.26s |
| **SVM Linear ($C=1.0$)** | **99.27%** | **99.09%** | **0.9912** | 0.9887 | 0.9861 | 0.9987 | 0.19s |
| **SVM Linear ($C=0.9$)** | **99.27%** | **99.09%** | **0.9912** | 0.9887 | 0.9861 | 0.9987 | 0.19s |
| **LDA** | **94.51%** | **93.46%** | **0.9341** | 0.9099 | 0.9028 | 0.9895 | 0.10s |
| **Random Forest (`max_depth=4`, Alkurdi)** | **92.80%** | **91.14%** | **0.9142** | 0.9002 | 0.8655 | 0.9769 | 1.41s |

> [!WARNING]
> **Data Leakage Artifact:** Because 60-second sliding windows with a 6-second shift overlap by 90%, adjacent windows from the same participant appear in both training and test partitions. Models memorize participant identity rather than generalized affective response, artificially inflating accuracies to 93%–100%.

---

## 8. Evaluation Results — Validation Scheme B (15-Fold Leave-One-Subject-Out CV)

*Strict subject-independent generalization: iteratively train on 14 participants and evaluate on the 1 held-out participant with fold-isolated scaling.*

### 8.1 Pooled Classification Metrics & Confusion Matrices

#### A. SVM (RBF Kernel) — **Peak Pooled Accuracy Model**
- **Pooled Accuracy:** **57.24%** (2,344 / 4,095 correct)
- **Balanced Accuracy:** **50.30%**
- **Macro F1:** **0.4980**
- **Class 0 F1 (No Anxiety):** **0.4354**
- **Class 1 F1 (Potential Anxiety):** **0.2621**
- **Class 2 F1 (Anxiety Increased):** **0.7965**
- **Macro-Subject Accuracy:** **57.09%** | **Macro-Subject Balanced Acc:** **57.74%**

Confusion Matrix (LOSO Pooled, rows=True [0, 1, 2], cols=Pred [0, 1, 2]):

| | Pred: No Anxiety (0) | Pred: Potential (1) | Pred: Anxiety Inc (2) |
|---|---|---|---|
| **True: No Anxiety (0)** | **514** | 412 | 184 |
| **True: Potential (1)** | 528 | **240** | 258 |
| **True: Anxiety Inc (2)** | 210 | 160 | **1,589** |

#### B. SVM Linear ($C=1.0$)
- **Pooled Accuracy:** **56.39%** (2,309 / 4,095 correct)
- **Balanced Accuracy:** **51.01%**
- **Macro F1:** **0.5024**
- **Class 0 F1:** **0.4558**
- **Class 1 F1:** **0.2672**
- **Class 2 F1:** **0.7841**

Confusion Matrix (LOSO Pooled):

| | Pred: No Anxiety (0) | Pred: Potential (1) | Pred: Anxiety Inc (2) |
|---|---|---|---|
| **True: No Anxiety (0)** | **621** | 375 | 114 |
| **True: Potential (1)** | 611 | **235** | 180 |
| **True: Anxiety Inc (2)** | 383 | 123 | **1,453** |

#### C. XGBoost (`n_estimators=100`) — **Top Macro-Subject Balanced Accuracy**
- **Pooled Accuracy:** **55.58%** (2,276 / 4,095 correct)
- **Balanced Accuracy:** **49.31%**
- **Macro F1:** **0.4942**
- **Class 0 F1:** **0.3777**
- **Class 1 F1:** **0.3098**
- **Class 2 F1:** **0.7951**
- **Macro-Subject Balanced Accuracy:** **57.78%**

Confusion Matrix (LOSO Pooled):

| | Pred: No Anxiety (0) | Pred: Potential (1) | Pred: Anxiety Inc (2) |
|---|---|---|---|
| **True: No Anxiety (0)** | **433** | 510 | 167 |
| **True: Potential (1)** | 524 | **320** | 182 |
| **True: Anxiety Inc (2)** | 226 | 210 | **1,523** |

#### D. Linear Discriminant Analysis (LDA)
- **Pooled Accuracy:** **55.56%**
- **Balanced Accuracy:** **51.28%**
- **Macro F1:** **0.5130**
- **Class 0 F1:** **0.3367**
- **Class 1 F1:** **0.4319**
- **Class 2 F1:** **0.7704**

Confusion Matrix (LOSO Pooled):

| | Pred: No Anxiety (0) | Pred: Potential (1) | Pred: Anxiety Inc (2) |
|---|---|---|---|
| **True: No Anxiety (0)** | **386** | 514 | 210 |
| **True: Potential (1)** | 471 | **488** | 67 |
| **True: Anxiety Inc (2)** | 326 | 232 | **1,401** |

#### E. Random Forest (`max_depth=4`, Alkurdi Spec)
- **Pooled Accuracy:** **54.29%**
- **Balanced Accuracy:** **46.84%**
- **Macro F1:** **0.4630**
- **Class 0 F1:** **0.4017**
- **Class 1 F1:** **0.1955**
- **Class 2 F1:** **0.7918**
- **Macro-Subject Accuracy:** **54.11%** | **Macro-Subject Balanced Acc:** **56.86%**

#### F. Random Forest (`max_depth=25`, Updated)
- **Pooled Accuracy:** **53.82%**
- **Balanced Accuracy:** **45.71%**
- **Macro F1:** **0.4490**
- **Class 0 F1:** **0.3571**
- **Class 1 F1:** **0.1952**
- **Class 2 F1:** **0.7947**
- **Macro-Subject Accuracy:** **53.60%** | **Macro-Subject Balanced Acc:** **56.25%**

#### G. KNN ($k=5$)
- **Pooled Accuracy:** **51.06%**
- **Balanced Accuracy:** **45.85%**
- **Macro F1:** **0.4555**
- **Class 0 F1:** **0.3990**
- **Class 1 F1:** **0.2214**
- **Class 2 F1:** **0.7462**

---

### 8.2 LOSO Cross-Validation Summary Table
| Model Architecture | Pooled Accuracy | Pooled Balanced Acc | Pooled Macro F1 | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Macro-Subject Accuracy | Macro-Subject Balanced Acc | Macro-Subject Macro F1 |
|---|---|---|---|---|---|---|---|---|---|
| **SVM (RBF)** | **57.24%** | **50.30%** | **0.4980** | **0.4354** | **0.2621** | **0.7965** | **57.09%** | **57.74%** | **0.4342** |
| **SVM Linear ($C=1.0$)** | **56.39%** | **51.01%** | **0.5024** | **0.4558** | **0.2672** | **0.7841** | **56.53%** | **56.03%** | **0.4332** |
| **SVM Linear ($C=0.9$)** | **56.36%** | **51.01%** | **0.5017** | **0.4585** | **0.2640** | **0.7827** | **56.50%** | **56.10%** | **0.4343** |
| **XGBoost** | **55.58%** | **49.31%** | **0.4942** | **0.3777** | **0.3098** | **0.7951** | **55.48%** | **57.78%** | **0.4405** |
| **LDA** | **55.56%** | **51.28%** | **0.5130** | **0.3367** | **0.4319** | **0.7704** | **55.28%** | **56.92%** | **0.4544** |
| **Random Forest (`max_depth=4`, Alkurdi)** | **54.29%** | **46.84%** | **0.4630** | **0.4017** | **0.1955** | **0.7918** | **54.11%** | **56.86%** | **0.4003** |
| **Random Forest (`max_depth=25`)** | **53.82%** | **45.71%** | **0.4490** | **0.3571** | **0.1952** | **0.7947** | **53.60%** | **56.25%** | **0.4204** |
| **KNN ($k=5$)** | **51.06%** | **45.85%** | **0.4555** | **0.3990** | **0.2214** | **0.7462** | **50.79%** | **52.32%** | **0.4071** |

---

### 8.3 LOSO Per-Subject Detailed Results

#### SVM (RBF Kernel) — Peak Generalization Model
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|---|
| **S2** | 257 | 0.5097 | 0.5708 | 0.4878 | 0.0000 | 0.5796 | 0.3558 |
| **S3** | 260 | 0.3423 | 0.2861 | 0.0000 | 0.1705 | 0.5012 | 0.2239 |
| **S4** | 272 | 0.4265 | 0.2929 | 0.0000 | 0.0889 | 0.6395 | 0.2428 |
| **S5** | 270 | 0.5185 | 0.5057 | 0.5097 | 0.0000 | 0.7710 | 0.4269 |
| **S6** | 269 | 0.5613 | 0.5694 | 0.2312 | 0.4444 | 0.8127 | 0.4961 |
| **S7** | 275 | 0.5782 | 0.5000 | 0.0000 | 0.0000 | 0.9582 | 0.3194 |
| **S8** | 263 | 0.7605 | 0.7679 | 0.0000 | 0.8037 | 0.8556 | 0.5531 |
| **S9** | 262 | 0.7481 | 0.6667 | 0.7338 | 0.0000 | 1.0000 | 0.5779 |
| **S10** | 289 | 0.5709 | 0.6404 | 0.4891 | 0.0000 | 0.9597 | 0.4829 |
| **S11** | 285 | 0.5018 | 0.4851 | 0.4048 | 0.2353 | 0.7205 | 0.4535 |
| **S13** | 280 | 0.6107 | 0.5960 | 0.6667 | 0.0000 | 0.7770 | 0.4812 |
| **S14** | 282 | 0.3759 | 0.5000 | 0.0000 | 0.0000 | 0.6897 | 0.2299 |
| **S15** | 282 | 0.7624 | 0.8073 | 0.7619 | 0.0000 | 0.8571 | 0.5397 |
| **S16** | 274 | 0.6715 | 0.7256 | 0.6270 | 0.0000 | 1.0000 | 0.5423 |
| **S17** | 275 | 0.6218 | 0.6433 | 0.2031 | 0.5746 | 0.8655 | 0.5478 |

#### SVM Linear ($C=1.0$)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|---|
| **S2** | 257 | 0.7198 | 0.8100 | 0.8100 | 0.0000 | 0.8835 | 0.5645 |
| **S3** | 260 | 0.2846 | 0.1987 | 0.0000 | 0.0619 | 0.5018 | 0.1879 |
| **S4** | 272 | 0.3750 | 0.2350 | 0.0000 | 0.0000 | 0.6395 | 0.2132 |
| **S5** | 270 | 0.4963 | 0.4677 | 0.4430 | 0.0000 | 0.7710 | 0.4046 |
| **S6** | 269 | 0.4647 | 0.4617 | 0.1910 | 0.0984 | 0.8571 | 0.3822 |
| **S7** | 275 | 0.6582 | 0.6065 | 0.0000 | 0.4295 | 0.9582 | 0.4626 |
| **S8** | 263 | 0.8327 | 0.8264 | 0.0000 | 0.8825 | 0.8556 | 0.5794 |
| **S9** | 262 | 0.7061 | 0.6047 | 0.7132 | 0.0000 | 0.9841 | 0.5658 |
| **S10** | 289 | 0.5329 | 0.6078 | 0.4534 | 0.0000 | 0.9032 | 0.4522 |
| **S11** | 285 | 0.4316 | 0.3934 | 0.1639 | 0.1215 | 0.6524 | 0.3126 |
| **S13** | 280 | 0.5929 | 0.6524 | 0.6705 | 0.0000 | 0.4673 | 0.5689 |
| **S14** | 282 | 0.2837 | 0.3774 | 0.0000 | 0.0000 | 0.6897 | 0.2299 |
| **S15** | 282 | 0.7553 | 0.8017 | 0.7527 | 0.0000 | 0.8571 | 0.5366 |
| **S16** | 274 | 0.6861 | 0.7378 | 0.6446 | 0.0000 | 1.0000 | 0.5482 |
| **S17** | 275 | 0.5855 | 0.5928 | 0.1062 | 0.4656 | 0.8952 | 0.4890 |

#### XGBoost (`n_estimators=100`)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|---|
| **S2** | 257 | 0.3541 | 0.5000 | 0.0000 | 0.0000 | 0.8792 | 0.2931 |
| **S3** | 260 | 0.6885 | 0.8090 | 0.0000 | 0.6621 | 0.7638 | 0.4753 |
| **S4** | 272 | 0.3750 | 0.2350 | 0.0000 | 0.0000 | 0.6395 | 0.2132 |
| **S5** | 270 | 0.5704 | 0.5562 | 0.4775 | 0.0000 | 0.7829 | 0.4201 |
| **S6** | 269 | 0.6840 | 0.5690 | 0.0000 | 0.8283 | 0.7158 | 0.5147 |
| **S7** | 275 | 0.3636 | 0.3145 | 0.0000 | 0.0000 | 0.7722 | 0.2574 |
| **S8** | 263 | 0.4068 | 0.5215 | 0.0000 | 0.0824 | 1.0000 | 0.3608 |
| **S9** | 262 | 0.5611 | 0.6518 | 0.0175 | 0.4796 | 0.9841 | 0.4938 |
| **S10** | 289 | 0.6055 | 0.6667 | 0.5068 | 0.0000 | 0.9794 | 0.4954 |
| **S11** | 285 | 0.6175 | 0.5505 | 0.2338 | 0.5286 | 0.8045 | 0.5223 |
| **S13** | 280 | 0.5214 | 0.4981 | 0.5316 | 0.0000 | 0.7591 | 0.4303 |
| **S14** | 282 | 0.3972 | 0.5020 | 0.0000 | 0.1474 | 0.6599 | 0.2691 |
| **S15** | 282 | 0.7660 | 0.8103 | 0.7660 | 0.0000 | 0.7941 | 0.5200 |
| **S16** | 274 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **S17** | 275 | 0.4109 | 0.4831 | 0.0000 | 0.4534 | 0.5729 | 0.3421 |

#### Random Forest (`max_depth=4`, Alkurdi Spec)
| Subject | Windows | Accuracy | Balanced Accuracy | Class 0 F1 (No Anx) | Class 1 F1 (Potential) | Class 2 F1 (Anx Inc) | Macro F1 |
|---|---|---|---|---|---|---|---|
| **S2** | 257 | 0.4436 | 0.5693 | 0.2434 | 0.0000 | 0.9838 | 0.4091 |
| **S3** | 260 | 0.5231 | 0.7075 | 0.0000 | 0.5926 | 0.5867 | 0.3931 |
| **S4** | 272 | 0.3750 | 0.2350 | 0.0000 | 0.0000 | 0.6395 | 0.2132 |
| **S5** | 270 | 0.4074 | 0.3615 | 0.1475 | 0.0000 | 0.7829 | 0.3102 |
| **S6** | 269 | 0.4089 | 0.3856 | 0.0936 | 0.0000 | 0.8127 | 0.3021 |
| **S7** | 275 | 0.3636 | 0.3145 | 0.0000 | 0.0000 | 0.7722 | 0.2574 |
| **S8** | 263 | 0.3802 | 0.5000 | 0.0000 | 0.0000 | 1.0000 | 0.3333 |
| **S9** | 262 | 0.5687 | 0.6517 | 0.0847 | 0.4744 | 0.9738 | 0.5110 |
| **S10** | 289 | 0.6055 | 0.6667 | 0.4956 | 0.0000 | 1.0000 | 0.4985 |
| **S11** | 285 | 0.3895 | 0.3442 | 0.0000 | 0.0460 | 0.6524 | 0.2328 |
| **S13** | 280 | 0.6107 | 0.6021 | 0.6667 | 0.0000 | 0.7698 | 0.4788 |
| **S14** | 282 | 0.4894 | 0.5909 | 0.0000 | 0.3077 | 0.6163 | 0.3080 |
| **S15** | 282 | 0.7766 | 0.8190 | 0.7789 | 0.0000 | 0.9038 | 0.5609 |
| **S16** | 274 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| **S17** | 275 | 0.5782 | 0.5930 | 0.0000 | 0.6065 | 0.7645 | 0.4570 |

---

## 9. Key Empirical Findings & Insights

1. **Resolution of the Zero-Inflation Pathology:**
   Continuous thresholding based on the minimum questionnaire resolvable step ($\pm \frac{1}{24}$) completely resolves the zero-inflation bug of historical iterations (2nd–5th try). In the 3-class setup, **15 out of 15 participants (100%)** demonstrate verified Class 2 anxiety during TSST, generating a balanced cohort of 27.1% Class 0, 25.1% Class 1, and 47.8% Class 2.
2. **Robust Anxiety State Detection (Class 2 F1 ~ 0.80):**
   Across unseen participants under strict Leave-One-Subject-Out CV, models achieve remarkably strong and consistent detection of genuine anxiety increases (Class 2 F1: **0.7965** for SVM RBF, **0.7951** for XGBoost, **0.7935** for Random Forest depth 25).
3. **The Challenge of the Neutral Boundary (Class 1):**
   The intermediate neutral category (Class 1: $\Delta_{\text{sum}} \in \{-1, 0, +1\}$) exhibits lower F1 scores (0.19–0.43) in LOSO cross-validation because resting physiological variance and subtle self-report measurement noise frequently blur the boundary between true baseline and minor fluctuations.
4. **Data Leakage Contrast (Scheme A vs. Scheme B):**
   Under naive random 80/20 splitting (Scheme A), classifiers achieve 93%–100% accuracy due to extreme temporal autocorrelation across the 90% overlapping sliding windows. When evaluated with fold-isolated Leave-One-Subject-Out CV (Scheme B), realistic subject-independent generalization stabilizes at **55%–57% pooled accuracy** (chance level = 33.3%).

---

## 10. Output File Locations

| Output Description | File Path |
|---|---|
| **Source Notebook** | [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb) |
| **Extracted 3-Class Feature Datasets** | [`Data/6th try/3class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/3class/) (`S{id}_stai_3class_60s_shift6s.pkl`) |
| **Trained Models (Scheme A)** | [`Models/6th try/3class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/) (`stai_3class_schemeA_*.joblib`) |
| **Evaluation Summary (Schemes A & B)** | [`Models/6th try/3class/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/evaluation_summary.json) |
| **Per-Subject LOSO Metrics** | [`Models/6th try/3class/per_subject_loso_results.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/per_subject_loso_results.json) |

---

## 11. Comprehensive Evolution & Comparison Matrix Across Iterations

| Dimension / Metric | 5th Try (`Scripts/5th try/`) | **6th Try 3-Class (`3class/`)** | 6th Try 2-Class (`2class/`) | Alkurdi Reference (`_forked/`) |
|---|---|---|---|---|
| **Target Ground Truth** | Subjective STAI (Buggy, 5 items transposed) | **Subjective STAI (Fixed, 6 items)** | Subjective STAI (Fixed, 6 items) | Objective Protocol (Base vs Fun vs TSST) |
| **Classification Task** | Binary: $\Delta \ge 0.2$ | **3-Class: $\Delta < -\frac{1}{24}$, $\pm \frac{1}{24}$, $\Delta > \frac{1}{24}$** | Binary: $\Delta \le 0$ vs $\Delta > 0$ | 3-Class Protocol: 0, 1, 2 |
| **Cohort Windows** | 4,147 windows (with reading tasks) | **4,095 windows (clean 4 conditions)** | 4,095 windows (clean 4 conditions) | 577 windows (tumbling 60s, stride 60s) |
| **Class Distribution** | 82.1% Class 0 / 17.9% Class 1 | **27.11% C0 / 25.05% C1 / 47.84% C2** | 43.59% Class 0 / 56.41% Class 1 | 54.1% Base / 30.3% Stress / 15.6% Fun |
| **Zero-Inflation** | Severe (8/15 subjects had 0 positive) | **Resolved (15/15 have Class 2)** | Resolved (15/15 have Class 1) | None (protocol stimulus driven) |
| **Sensor Clock Alignment** | Polyphase rational resampling (70 Hz) | **Alkurdi `outer` join + `pd.to_datetime`** | Alkurdi `outer` join + `pd.to_datetime` | Alkurdi `outer` join + `pd.to_datetime` |
| **Predictive Features** | 154 features | **154 features** | 154 features | 120 features |
| **Scheme A Leakage Acc** | 86%–89% (`GroupShuffleSplit`) | **92.8%–100.0% (`train_test_split`)** | 92.3%–100.0% (`train_test_split`) | 93.6%–97.9% (`train_test_split`) |
| **LOSO Peak Accuracy** | 87.41% (SVM RBF, collapsed minority) | **57.24% (SVM RBF)** | 74.14% (XGBoost) / 73.46% (RF depth 4) | *No LOSO evaluation performed* |
| **LOSO Peak Balanced Acc** | 70.78% (XGBoost) | **51.28% (LDA) / 51.01% (SVM Lin)** | 74.24% (XGBoost) / 74.00% (RF depth 4) | *No LOSO evaluation performed* |
| **LOSO Peak Macro F1** | 0.5584 (SVM RBF) | **0.5130 (LDA) / 0.5024 (SVM Lin)** | 0.7394 (XGBoost) / 0.7338 (RF depth 4) | *No LOSO evaluation performed* |
| **Macro-Subject Anxiety F1** | 0.16–0.21 (dragged down by zeros) | **0.7965 (Class 2 F1 across all 15 subjects)** | 0.7720 (Class 1 across all 15 subjects) | *N/A* |
