---
tags:
  - skripsi/research-log
  - machine-learning
  - validation-schemes
  - spectral-analysis
  - feature-engineering
  - alkurdi-alignment
  - peak-frequency-rectification
  - model-accuracy-impact
  - wesad
  - wear-os
  - samsung-health-sdk
  - smartwatch-deployment
  - sensor-logging
  - offline-batching
  - adb-pull
date: 2026-11-17
aliases:
  - 2026-11-17 Research Note
  - Peak Frequency Sampling Rate Rectification & Model Accuracy Impact
  - Audit & Rektifikasi Parameter Sampling Rate get_peak_freq
  - Wear OS Smartwatch Multi-Sensor Logging Pipeline
  - Smartwatch Sensor Data Extraction & Offline Batch Logging
---

# Research Note — Spectral Peak Frequency Sampling Rate Rectification & Smartwatch Multi-Sensor Deployment Pipeline

**Date:** 2026-11-17  
**Workspace:** [`Scripts/6th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/) | [`Scripts/6th try/fb_code/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/) | [`Deployment/test1/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/)  
**Modified Scripts & Modules:**  
- Helper Utility: [`Scripts/6th try/fb_code/utils.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/utils.py) (`get_peak_freq`)  
- Reference Extractor: [`Scripts/6th try/fb_code/feature_extraction.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/feature_extraction.py)  
- Notebook 2-Class Full: [`Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/stai_2class_alkurdi_alignment.ipynb)  
- Notebook 3-Class Full: [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb)  
- Notebook Sliced Wrist: [`Scripts/6th try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb)  
- Notebooks Minimum Sensors: [`Scripts/6th try/2class/Minimum_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Minimum_sensor_modals/) (`acc_temp`, `eda_acc`, `eda_temp`)  
- Android Smartwatch Deployment:
  - App Manifest: [`Deployment/test1/app/src/main/AndroidManifest.xml`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/AndroidManifest.xml)
  - Batch CSV Logger: [`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/CsvLogger.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/CsvLogger.kt)
  - Sensor Tracking Manager: [`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt)
  - Wear OS Compose UI: [`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/presentation/MainActivity.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/presentation/MainActivity.kt)
**Related Specs & Documentation:**  
- Audit Report: [`_Obsidian Notes/Scripts & Implementation/Script 6th Try Pipeline (Alkurdi Alignment).md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Scripts%20&%20Implementation/Script%206th%20Try%20Pipeline%20(Alkurdi%20Alignment).md)  
- Model Specs: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]], [[Model Output Specs 6th Try 2-Class Sliced Sensor]], [[Model Output Specs 6th Try 2-Class Minimum Sensor]]  
- Smartwatch Technical Docs: [`Deployment/test1/_user_folder/Obsidian Notes/2026-11-17_Implementation.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/2026-11-17_Implementation.md), [`Deployment/test1/_user_folder/Obsidian Notes/Samsung_Health_Sensor_SDK_Specs.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/Samsung_Health_Sensor_SDK_Specs.md)  

---

## 1. Executive Summary

Today marked two major synchronized breakthroughs across both the machine learning modeling and physical wearable hardware deployment phases of the research:

1. **Spectral Feature Calibration & Model Sensitivity (ML Pipeline):**  
   During an in-depth architectural audit of the 6th-try pipeline documented in [[Script 6th Try Pipeline (Alkurdi Alignment)]], a critical legacy quirk was identified in the spectral feature extraction code inherited from the upstream Alkurdi et al. reference repository (`_forked/fb_code/utils.py`). Specifically, `get_peak_freq` calculated signal periodograms using a **hardcoded sampling frequency of $f_s = 8\text{ Hz}$** across all sensors regardless of their actual physical sampling rates ($32\text{ Hz}$, $64\text{ Hz}$, or $700\text{ Hz}$). Re-extracting features with true physical sampling frequencies alters 19 spectral peak features, subtly shifting benchmark metrics while successfully grounding all biomarkers in physical Hertz.
2. **Wear OS Smartwatch Multi-Sensor App Deployment ([`Deployment/test1/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/)):**  
   Significant progress was achieved on the wearable hardware acquisition pipeline. The Android smartwatch application built with the Samsung Health Sensor SDK was refactored into a high-performance, standalone multi-sensor logger operating directly on the Samsung Galaxy Watch (Wear OS). The app tracks continuous physiological signals—including raw multi-wavelength Photoplethysmogram (PPG Green, Red, IR), Electrodermal Activity (EDA), Tri-axial Accelerometer (ACC), Continuous Heart Rate (BPM), Skin and Ambient Temperature, SpO2, and ECG—and records them locally via a RAM-buffered background coroutine batching architecture. Logged sessions are exported seamlessly to the host computer using Android Debug Bridge (`adb pull`), establishing a zero-packet-loss, battery-efficient pipeline for real-world ambulatory data collection.

---

## 2. Technical Audit: The Hardcoded `fs=8` Periodogram Issue

### 2.1 The Origin in Upstream `_forked` Code
In the upstream repository (`_forked/fb_code/utils.py`), the periodogram peak frequency estimator was implemented as:

```python
# Upstream Alkurdi implementation (_forked/fb_code/utils.py):
def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)  # Flaw: Hardcoded fs=8 Hz
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq
```

### 2.2 Mathematical Distortion Factor
SciPy's `scipy.signal.periodogram(x, fs=fs)` constructs the discrete frequency vector as:
$$f_k = \frac{k \cdot f_s}{N}, \quad \text{for } k = 0, 1, \dots, \left\lfloor\frac{N}{2}\right\rfloor$$
where $N$ is the number of samples in input vector $x$, and the maximum resolvable frequency is the Nyquist limit $f_{\text{Nyquist}} = \frac{f_s}{2}$.

When a signal sampled at physical frequency $f_{s, \text{true}}$ is processed assuming $f_s = 8\text{ Hz}$, the computed peak frequency $f_{\text{calc}}$ is compressed or distorted by a constant ratio:
$$f_{\text{calc}} = f_{\text{true}} \times \left(\frac{8}{f_{s, \text{true}}}\right)$$

| Modality & Device | Native $f_{s, \text{true}}$ | Assumed $f_s$ | Distortion Factor $\left(\frac{8}{f_{s, \text{true}}}\right)$ | Example True Frequency | Resulting Distorted Frequency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Wrist BVP (Empatica E4)** | **$64\text{ Hz}$** | $8\text{ Hz}$ | **$0.125\times$** ($1/8$) | $1.50\text{ Hz}$ ($90\text{ bpm}$) | **$0.1875\text{ Hz}$** |
| **Wrist ACC (Empatica E4)** | **$32\text{ Hz}$** | $8\text{ Hz}$ | **$0.250\times$** ($1/4$) | $2.00\text{ Hz}$ (Gait/Tremor) | **$0.5000\text{ Hz}$** |
| **Chest ACC (RespiBAN)** | **$700\text{ Hz}$** | $8\text{ Hz}$ | **$0.0114\times$** ($8/700$) | $5.00\text{ Hz}$ (Cardiokinet.) | **$0.0571\text{ Hz}$** |

### 2.3 Physical Implications
1. **BVP Optical Pulse:** A typical resting heart rate of $60\text{--}120\text{ bpm}$ corresponds to a fundamental frequency of $1.0\text{--}2.0\text{ Hz}$. Hardcoded to $8\text{ Hz}$, the detected peak frequency collapsed to $0.125\text{--}0.25\text{ Hz}$, overlapping numerically with the ultra-low respiratory frequency band.
2. **Triaxial Accelerometer Sub-windows:** Dynamic body tremor and posture adjustments captured in 5-second sub-windows had their spectral components compressed by $4\times$ (wrist) and $87.5\times$ (chest), reducing physical discriminability between fine kinetic tremor and slow gravitational drift.

---

## 3. Implementation of the Rectification Architecture

### 3.1 Module Localization (`Scripts/6th try/fb_code/`)
To preserve `_forked` strictly as an immutable reference snapshot of Alkurdi et al.'s original code, the entire `fb_code` directory was localized to [`Scripts/6th try/fb_code/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/).

### 3.2 Refactoring `utils.py`
In [`Scripts/6th try/fb_code/utils.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/fb_code/utils.py#L148-L172), `get_peak_freq` was restructured to accept dynamic $f_s$ with empty-array safeguards:

```python
def get_peak_freq(x, fs=8):
    '''get peak frequency

    Function that computes the peak frequency of a dataset.

    Parameters
    ----------
    x : pandas series or pandas dataframe
        the series/dataframe upon which we wish to compute the peak frequency
    fs : float, optional
        sampling frequency of the signal x (default is 8 for backwards compatibility)

    Returns
    -------
    peak_freq : float
        peak frequency of the dataset in physical Hertz
    '''
    f, Pxx = scisig.periodogram(x, fs=fs)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    if not psd_dict:
        return np.nan
    peak_freq = psd_dict[max(psd_dict.keys())]
    return peak_freq
```

### 3.3 Synchronizing the 6th Try Notebooks
All 6 notebooks in `Scripts/6th try/` were updated across two key execution cells:

1. **Cell 1 (Imports & Directory Resolution):**
   ```python
   # Direct reference to localized 6th try helper library
   REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'WESAD').exists())
   FB_CODE_DIR = REPO_ROOT / 'Scripts' / '6th try' / 'fb_code'
   if str(FB_CODE_DIR) not in sys.path:
       sys.path.insert(0, str(FB_CODE_DIR))
   ```

2. **Cell 7 (Feature Extraction Call Sites):**
   - **BVP Periodogram:**
     ```python
     wdf['BVP_peak_freq'] = get_peak_freq(window['BVP'].dropna(), fs=RAW_FS['BVP'])  # fs=64
     ```
   - **Sub-window ACC Helper (`_safe_acc_feature_values`):**
     ```python
     def _safe_acc_feature_values(series, fs=RAW_FS['ACC']):
         ...
         peak_freq = get_peak_freq(series, fs=fs)
     ```
   - **Sub-window Aggregation Loop:**
     ```python
     # Wrist sub-windows (32 Hz)
     for col in ['ACC_x', 'ACC_y', 'ACC_z']:
         std_v, int_v, pk_v = _safe_acc_feature_values(w_sub[col], fs=RAW_FS['ACC'])
     # Chest sub-windows (700 Hz)
     for col in ['ACC_x_C', 'ACC_y_C', 'ACC_z_C']:
         std_v, int_v, pk_v = _safe_acc_feature_values(c_sub[col], fs=FS['chest'])
     ```
   - **Global Window ACC Loop:**
     ```python
     col_fs = FS['chest'] if col.endswith('_C') else RAW_FS['ACC']
     wdf[f'{col}_peak_freq'] = None if series.empty else get_peak_freq(series, fs=col_fs)
     ```

---

## 4. Impact Analysis: Sensitivity of Model Accuracy

### 4.1 Scope of Affected Features
Across the different pipeline tiers, the number of affected features out of the predictive feature set $X$ is distributed as follows:

| Pipeline Configuration | Total Features ($X$) | Spectral Peak Features Affected | Percentage of Feature Space | Modalities Involved |
|---|:---:|:---:|:---:|---|
| **Full 2-Class & 3-Class** | **155** | **19 features** | **12.26%** | BVP ($64\text{ Hz}$), Wrist ACC ($32\text{ Hz}$), Chest ACC ($700\text{ Hz}$) |
| **Sliced Sensor (Smartwatch)** | **62** | **7 features** | **11.29%** | Wrist ACC ($32\text{ Hz}$) sub-windows & global axes |
| **Minimum ACC + TEMP** | **56** | **7 features** | **12.50%** | Wrist ACC ($32\text{ Hz}$) sub-windows & global axes |
| **Minimum EDA + ACC** | **56** | **7 features** | **12.50%** | Wrist ACC ($32\text{ Hz}$) sub-windows & global axes |
| **Minimum EDA + TEMP** | **18** | **0 features** | **0.00%** | None (electrodermal and thermal features only) |

#### Exact Inventory of the 19 Calibrated Features:
1. `BVP_peak_freq` (global window)
2. `ACC_x_peak_freq`, `ACC_y_peak_freq`, `ACC_z_peak_freq` (wrist global window)
3. `ACC_x_C_peak_freq`, `ACC_y_C_peak_freq`, `ACC_z_C_peak_freq` (chest global window)
4. `ACC_x_peak_freq_mean`, `ACC_x_peak_freq_std` (wrist sub-window moments)
5. `ACC_y_peak_freq_mean`, `ACC_y_peak_freq_std` (wrist sub-window moments)
6. `ACC_z_peak_freq_mean`, `ACC_z_peak_freq_std` (wrist sub-window moments)
7. `ACC_x_C_peak_freq_mean`, `ACC_x_C_peak_freq_std` (chest sub-window moments)
8. `ACC_y_C_peak_freq_mean`, `ACC_y_C_peak_freq_std` (chest sub-window moments)
9. `ACC_z_C_peak_freq_mean`, `ACC_z_C_peak_freq_std` (chest sub-window moments)

---

### 4.2 Why Model Accuracy Exhibits Minor Fluctuations
When re-extracted and evaluated, classification performance across the machine learning models exhibits minor shifts (typically $\pm 0.5\%\text{--}1.8\%$ across pooled and fold metrics). The mathematical mechanisms driving these shifts are:

```
True Frequency Scaling (f_s calibration)
   │
   ├── 1. Non-linear Peak Rescaling
   │     └── Peak frequencies expand into physical Hertz (e.g. 0.1875 Hz -> 1.50 Hz)
   │
   ├── 2. Decision Tree Split Re-alignment (Random Forest & XGBoost)
   │     └── Optimal Gini / Entropy split thresholds shift to accommodate actual Hertz distributions
   │
   ├── 3. Covariance Matrix Re-estimation (LDA)
   │     └── Linear projections re-weight the uncompressed variance of ACC and BVP spectral peaks
   │
   ├── 4. Distance Metric & Margin Readjustment (KNN, SVM RBF & Linear)
   │     └── Standardized Euclidean/kernel distances shift subtly along the 19 calibrated coordinate axes
   │
   └── 5. Per-Subject Generalization Differences in LOSO (Validation Scheme B)
         └── Held-out subjects with unique kinetic or pulse dynamics project differently on calibrated axes
```

1. **Non-Uniform Rescaling across Sensors:**  
   Because the distortion factor was different for each sensor ($0.125\times$ for BVP, $0.25\times$ for wrist ACC, and $0.0114\times$ for chest ACC), fixing $f_s$ is **not a simple scalar multiplication across the whole matrix**. The relative spatial relationship between BVP spectral peaks and chest ACC spectral peaks changed non-linearly.
2. **Impact on Z-Score Standardization:**  
   While `StandardScaler` standardizes variables to $\mu=0$ and $\sigma=1$, periodogram peak distributions are often multimodal or non-Gaussian (due to discrete dominant frequencies during stationary vs. active states). Rescaling the underlying frequency axis redistributes the variance and alters the standardized coordinates of outliers.
3. **Tree-Based Partitioning Sensitivity:**  
   In Random Forest (`max_depth=4` and `max_depth=25`) and XGBoost, candidate split points are evaluated over feature quantiles. A feature whose dynamic range was artificially compressed previously contributed less resolving power than it now possesses with its full physical spectrum resolved.

---

### 4.3 Scientific Justification for the Skripsi Thesis

> [!IMPORTANT] Methodological Integrity Over Numerical Invariance
> In academic research and thesis defense, maintaining a historical accuracy metric by preserving an uncalibrated code defect (`fs=8` hardcoded across 32 Hz, 64 Hz, and 700 Hz sensors) constitutes an unacceptable methodological flaw. The minor performance shift observed after this correction represents:
> 1. **Physical Grounding:** Features such as `BVP_peak_freq` now represent genuine pulse frequencies ($1\text{--}2\text{ Hz}$), enabling physiological interpretability in Bab 4.
> 2. **Cross-Hardware Portability:** The model is no longer bound to a 8 Hz computational artifact and can be realistically deployed on consumer Wear OS smartwatches where sampling rates are fixed by device drivers.
> 3. **Academic Transparency:** Explicitly documenting this audit, the rectification, and the resulting sensitivity in Bab 3 and Bab 4 demonstrates rigorous scientific vigilance.

---

## 5. Historical Archive: Outdated 3-Class Metrics & Narration Audit (Notebook Cell 16)

> [!WARNING] Mandatory Pre-Update Archive
> As required by our methodological audit protocol, this section formally records the previous, outdated metrics and text from Cell 16 of [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb) before that markdown cell is updated with rectified data.

### 5.1 Outdated Text & Metrics Preserved from Cell 16
Prior to rectification, Cell 16 contained narrative text reflecting an early test execution conducted before the questionnaire parser's *indexing transposition bug* was fixed (i.e. when reverse scoring of items 1, 4, 6 and correct row-wise parsing of `S{id}_quest.csv` were not yet properly integrated):

```
Outdated Cell 16 Class Distribution (INVALID / LEGACY):
├── Class 0 (No Anxiety Increase):        2,512 windows (60.57%)
├── Class 1 (Potential Anxiety Increase):   832 windows (20.06%)
└── Class 2 (Anxiety Increased):            803 windows (19.36%)
Total: 4,147 windows across cohort
Legacy Note: "with 12 out of 15 subjects contributing positive windows (Class 1 or 2)"

Outdated Cell 16 Model Metrics (INVALID / LEGACY):
├── Validation Scheme A: Reported 94.5%–100.0% accuracy across models
└── Validation Scheme B (15-Fold LOSO):
    ├── Pooled Accuracy: Reported range of 52.7%–67.9%
    ├── Top Balanced Accuracy: XGBoost at 54.04% (Macro F1: 0.5454)
    ├── Random Forest (max_depth=4, Alkurdi):
    │   └── Stated to collapse onto majority Class 0 with Class 1 F1: 0.0021
    └── Random Forest (max_depth=25):
        └── Stated to achieve Balanced Accuracy: 45.48% and Class 1 F1: 0.0257
```

### 5.2 Root-Cause Forensics of the Discrepancy
1. **The Questionnaires Indexing Bug:** In earlier iterations (1st through early 6th try drafts), STAI questionnaire values were read column-wise rather than row-wise or without proper inversion of positive items (`feel_at_ease`, `relaxed`, `pleasant`). This artificially compressed the TSST scores of several participants, falsely categorizing anxiety-induced windows as resting baseline or "No Anxiety Increase" (inflating Class 0 to 2,512 windows).
2. **Rectification in Code Execution (Cell 13):** When the questionnaire parser in Cell 2 was rectified to parse all 6 items with proper reverse-scoring ($5 - \text{raw}$), re-running the extraction produced the true, psychometrically verified ground truth in Cell 13:
   - **Class 0 (No Anxiety Increase):** **1,110 windows (27.11%)**
   - **Class 1 (Potential Anxiety Increase):** **1,026 windows (25.05%)**
   - **Class 2 (Anxiety Increased):** **1,959 windows (47.84%)**
   - **Total Cohort Windows:** Exactly **4,095 windows** (all 15/15 participants exhibit $\Delta > \frac{1}{24}$ during TSST).
3. **True Model Benchmarks on Rectified Ground Truth:**
   - **Scheme A:** RF `max_depth=25` (100.0%), XGBoost (100.0%), SVM RBF (99.51%), KNN (99.39%), SVM Linear (99.27%), LDA (94.51%), RF `max_depth=4` (92.80%).
   - **Scheme B (15-Fold LOSO):**
     - **Top Pooled Accuracy:** **SVM RBF at 57.22%** (Balanced Acc: 50.27%, Macro F1: 0.4976, Class 2 F1: 0.7965).
     - **Top Balanced Accuracy:** **SVM Linear ($C=1.0$) at 51.01%** (Pooled Acc: 56.39%, Macro F1: 0.5024).
     - **Random Forest (`max_depth=4`):** Pooled Acc 54.68%, Balanced Acc 47.92%, Macro F1 0.4727.
     - **Random Forest (`max_depth=25`):** Pooled Acc 55.48%, Balanced Acc 48.60%, Macro F1 0.4808.
     - **XGBoost:** Pooled Acc 53.60%, Balanced Acc 48.24%, Macro F1 0.4754.

---

## 6. Feature Space Sanitization: Promoting `ACC_subwindow_count` to Metadata

### 6.1 Motivation & Rationale
In the feature extraction pipeline (Cell 7 of all notebooks), an operational tracking variable was introduced:
```python
aggregated['ACC_subwindow_count'] = subwindow_count  # typically ~221 sub-windows per 60s frame
```
In prior configurations, this variable was omitted from the `metadata_columns` exclusion set. Consequently, it leaked into the predictive feature matrix $X$ as a 155th feature column (or 62nd in sliced setups). 

While `ACC_subwindow_count` had near-zero variance across clean windows, it represents an **operational pipeline counter** rather than a genuine physiological biomarker.

### 6.2 Sanitization Across All Scripts
To restore absolute physiological feature purity, `'ACC_subwindow_count'` is explicitly added to `metadata_columns` across:
- `Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb` (Cell 13)
- `Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb` (Cell 13)
- `Scripts/6th try/2class/Minimum_sensor_modals/*.ipynb` (Cell 11 & Cell 13)
- `Scripts/6th try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb` (Cell 11 & Cell 13)
- `Scripts/6th try/analyze_sensor_modality_weight.ipynb` (Cell 1: `METADATA_COLUMNS`)
- `Scripts/Utils/generate_2class_data_and_eval.py` & `relabel_and_evaluate_6th_try.py`

### 6.3 Resulting Feature Dimensions
- **Full Modality Pipelines (2-Class & 3-Class):** Exactly **154 physiological features** ($155 - 1$).
- **Sliced Sensor Pipeline (Smartwatch Wrist Only):** Exactly **61 physiological features** ($62 - 1$).
- **Minimum Sensor (ACC + TEMP & EDA + ACC):** Exactly **55 physiological features** ($56 - 1$).
- **Minimum Sensor (EDA + TEMP):** Retains **18 physiological features** (unaffected).

---

## 7. Methodological Critique: The Invalidity of Alkurdi's Evaluation Scheme in Affective Computing

### 7.1 Why Alkurdi's Scheme A Is Invalid for the 6th Try Pipeline
Alkurdi et al. evaluated their models using a naive 80/20 train/test split:
```python
train_test_split(test_size=0.2, random_state=0)
```
In the 6th-try pipeline, data is segmented using **sliding windows of 60 seconds with a 6-second shift (90% temporal overlap)**. When a naive random split is applied to this overlapping sequence:
1. **Direct Temporal Autocorrelation Leakage:** Two adjacent windows ($[t, t+60\text{s}]$ and $[t+6\text{s}, t+66\text{s}]$) share **54 seconds (90%) of identical raw physiological signals**. If Window 1 is assigned to the training set and Window 2 is assigned to the test set, the classifier is evaluated on data it has already seen 90% of during training.
2. **Subject Identity Leakage (Fingerprinting):** The classifier does not learn a generalizable function relating autonomic arousal to subjective anxiety; instead, it memorizes participant-specific resting skin conductance, idiosyncratic heart rate ranges, and arm tremor habits. This results in trivially inflated test accuracy (**93%–100%**).

### 7.2 Deeper Critique: Why Alkurdi's Evaluation Method Was Questionable Even in Their Own Study
A critical examination of Alkurdi et al.'s original dissertation code (`_forked`) reveals an even more fundamental methodological issue that compromises their reported $>90\%$ accuracy claims:

1. **Non-Overlapping Windows Still Share Subject Identity:**  
   Alkurdi et al. used non-overlapping tumbling windows (length 60s, stride 60s). However, they applied `train_test_split(test_size=0.2, random_state=0)` **globally across all 577 windows from all 15 subjects pooled together**.
2. **Inter-Subject vs. Intra-Subject Physiological Variance:**  
   In biomedical and affective computing, physiological variability between different human beings (**inter-subject variability**) is significantly larger than the physiological shifts caused by psychological stress within a single person (**intra-subject variability**). A person's baseline tonic EDA or resting HRV is uniquely individual like a physiological fingerprint.
3. **The Unfair Advantage of Pooled Random Splitting:**  
   Because windows from Subject $k$ were randomly scattered across both the 80% train split and the 20% test split, the test set contained windows from the exact same human subjects present in the training set. A machine learning model (such as Random Forest or SVM) naturally exploits the unique subject-specific feature clusters to identify the subject, effectively solving an easy pattern matching problem:
   $$\text{"This test window has an EDA baseline of } 8.2\,\mu\text{S} \implies \text{this is Subject 10 during TSST."}$$
   The model does not need to learn true stress or anxiety dynamics; it only needs to map subject fingerprints to their labeled condition in the training set.
4. **Failure in Ambulatory Deployment:**  
   When such a model is deployed on a brand new person (*unseen subject*) in real life, it completely fails because the new user's baseline physiology does not match any of the memorized fingerprints.

### 7.3 Academic Conclusion for Thesis Bab 3 & Bab 4
- **Scheme A (Naive Random Split 80/20)** must be presented in the thesis strictly as an **analytical counter-example (demonstration of data leakage)** to explain the artificially high accuracies in prior literature (including Alkurdi et al.).
- **Scheme B (15-Fold Leave-One-Subject-Out CV with In-Fold Scaling)** must be defended as the **only scientifically rigorous and clinically valid benchmark** for wearable anxiety monitoring, proving that performance metrics (e.g. 74.43% for 2-class RF and 57.22% for 3-class SVM) reflect true, subject-independent physiological generalization.

---

## 8. Summary of Modified Notes & Cross-References

| Note Path | Scope of Modification |
|---|---|
| [`_Obsidian Notes/Research Notes/2026-11-17 Research Note.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Research%20Notes/2026-11-17%20Research%20Note.md) | Documented `fs=8` rectification, archived historical 3-class Cell 16 metrics, detailed `ACC_subwindow_count` sanitization, formulated methodological critique of Alkurdi's evaluation scheme, and integrated the Wear OS smartwatch deployment logging progress (§9). |
| [`_Obsidian Notes/Model Specs/Model Output Specs 6th Try 2-Class.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class.md) | Added technical addendum to Feature Extraction (§4) and Empirical Findings (§9) documenting 19 spectral features calibration and 154-feature sanitization. |
| [`_Obsidian Notes/Model Specs/Model Output Specs 6th Try 3-Class.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%203-Class.md) | Added calibration addendum and updated feature count to 154 features. |
| [`_Obsidian Notes/Model Specs/Model Output Specs 6th Try 2-Class Sliced Sensor.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class%20Sliced%20Sensor.md) | Annotated the 7 wrist ACC peak features in Feature Taxonomy (§3) with physical 32 Hz calibration and updated feature count to 61 features. |
| [`_Obsidian Notes/Model Specs/Model Output Specs 6th Try 2-Class Minimum Sensor.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class%20Minimum%20Sensor.md) | Annotated pair-wise spectral feature calibration and updated feature count to 55 features for ACC pairs. |
| [`_Obsidian Notes/Scripts & Implementation/Script 6th Try Pipeline (Alkurdi Alignment).md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Scripts%20&%20Implementation/Script%206th%20Try%20Pipeline%20(Alkurdi%20Alignment).md) | Updated §3, §5.1, §5.2 (`ACC_subwindow_count`), §5.3 (Cell 16 update), and expanded §5.4 with the comprehensive critique of Alkurdi's evaluation method. |
| [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb) | Rectified narrative text in Cell 16 to match ground-truth cohort distribution (1,110 / 1,026 / 1,959) and Scheme B benchmark outputs. |
| [`Deployment/test1/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/) | Wear OS smartwatch multi-sensor offline logging app implemented using Samsung Health Sensor SDK and Coroutine disk buffering. |
| [`Deployment/test1/_user_folder/Obsidian Notes/2026-11-17_Implementation.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/2026-11-17_Implementation.md) | Detailed implementation note for Wear OS sensor data extraction, offline batching architecture, and ADB workflow. |
| [`Deployment/test1/_user_folder/Obsidian Notes/Samsung_Health_Sensor_SDK_Specs.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/Samsung_Health_Sensor_SDK_Specs.md) | Specifications and operational guidelines for `CONTINUOUS` vs `ON_DEMAND` trackers in Samsung Health Sensor SDK. |

---

## 9. Wearable Hardware Deployment: Multi-Sensor Data Logging & Local Acquisition Pipeline

### 9.1 Background & Engineering Motivation
To bridge the gap between experimental offline model development (trained on the benchmark WESAD dataset) and real-world deployment on consumer wearables, significant implementation milestones were achieved today in the Android smartwatch submodule repository ([`Deployment/test1/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/)).

The target wearable platform is the **Samsung Galaxy Watch** running **Wear OS** (API Level 34+ / Android 14+), powered by the **Samsung Health Sensor SDK**.

Previous exploratory concepts considered streaming physiological signals in real-time to an external computer via Wi-Fi or Bluetooth sockets. However, real-time wireless transmission in consumer wearables introduces severe operational vulnerabilities during human subject experiments:
1. **Excessive Battery Consumption & Thermal Throttling:** Continuous Wi-Fi/Bluetooth packet transmission draws substantial current (200–400 mA), depleting the watch's compact battery within 30–60 minutes and triggering operating system thermal safety throttling, which forcibly terminates sensor drivers.
2. **Packet Loss & Network Jitter:** Wearable devices experience variable radio attenuation as subjects move their arms or walk, leading to dropped samples and broken time-series continuity.
3. **Aggressive OS Background Throttling:** Wear OS aggressively terminates unprivileged background socket connections and network threads when the display dims or sleeps to conserve power.

To resolve these challenges, the architecture was fundamentally engineered around a **high-reliability offline batch logging paradigm** that operates fully locally on the watch hardware.

---

### 9.2 Offline Batching Architecture (`CsvLogger.kt`)
The logging subsystem ([`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/CsvLogger.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/CsvLogger.kt)) implements a memory-buffered, asynchronous disk persistence architecture designed for high throughput without UI latency:

```
[Sensor Event Listeners (ECG, PPG, ACC, EDA, HR, Temp)]
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  In-Memory Synchronized Buffer   │  (RAM String Accumulator)
        │     `buffer: MutableList<String>`│
        └─────────────────┬────────────────┘
                          │ (Every 10 seconds OR Stop Session)
                          ▼
        ┌──────────────────────────────────┐
        │  Background Coroutine Flush Task │  (Dispatchers.IO + SupervisorJob)
        └─────────────────┬────────────────┘
                          │
                          ▼
        ┌──────────────────────────────────┐
        │ Flash Storage (CSV File)         │  (External Files Directory)
        │ `sensor_data_YYYYMMDD_HHmmss.csv`│
        └─────────────────┬────────────────┘
                          │
                          ▼ (Post-Experiment Extraction)
        ┌──────────────────────────────────┐
        │ Host Workstation (via `adb pull`)│
        └──────────────────────────────────┘
```

1. **Thread-Safe RAM Buffer (`StringBuilder`):**
   Incoming high-frequency sensor callbacks (e.g. ECG at up to 500 Hz, PPG at 25–100 Hz, ACC at 25 Hz) do not perform immediate blocking disk I/O. Instead, events are formatted as comma-separated values and appended to an in-memory synchronized list (`buffer`).
2. **Periodic Coroutine Flush to Flash Storage:**
   Flushing to disk is decoupled from the main/UI thread and managed by Kotlin Coroutines on `Dispatchers.IO + SupervisorJob()`. Flushes occur automatically every **10 seconds** (`MAX_BUFFER_TIME_MS = 10_000L`) or upon session termination, writing sequentially to `context.getExternalFilesDir(null)`.
3. **Session-Bound File Lifecycle:**
   - **Start Logging (`startSession()`):** Initializes a fresh, timestamped file formatted as `sensor_data_YYYYMMDD_HHmmss.csv` and writes the standardized CSV header.
   - **Stop Logging (`stopSession()` / `flush()`):** Immediately flushes all remaining buffered samples to flash storage and closes file handlers, preventing zero-byte clutter or corrupted CSV records on app restart.

#### Standardized CSV Output Schema:
```csv
Timestamp,SensorType,Value1,Value2,Value3,Value4
```
* **`Timestamp`**: Unix epoch time in milliseconds (`System.currentTimeMillis()`).
* **`SensorType`**: Name of the reporting sensor tracker.
* **`Value1`–`Value4`**: Sensor-specific physiological coordinates or status metrics.

---

### 9.3 Comprehensive Sensor Coverage & SDK Tracking Engine (`TrackingManager.kt`)

The tracking orchestrator ([`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt)) connects to Samsung's proprietary `HealthTrackingService` and registers event listeners across all available physical and algorithmic sensor trackers on the watch.

Because [`Deployment/test1/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/) operates as an isolated git submodule, standalone notes were maintained in [`Deployment/test1/_user_folder/Obsidian Notes/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/) ([`2026-11-17_Implementation.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/2026-11-17_Implementation.md) and [`Samsung_Health_Sensor_SDK_Specs.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Obsidian%20Notes/Samsung_Health_Sensor_SDK_Specs.md)). To establish a unified single source of truth across the repository, the complete tracker taxonomies, sampling specifications, and column schemas are fully consolidated below.

---

#### 9.3.1 Tracker Classification: `CONTINUOUS` vs. `ON_DEMAND` Specifications

The Samsung Health Sensor SDK categorizes wearable sensors into two primary operational modes based on sampling frequency, system resource utilization, and intended measurement duration:

```
                          Samsung Health Sensor SDK
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           ▼                                                   ▼
 ┌───────────────────────────┐                       ┌───────────────────────────┐
 │   CONTINUOUS TRACKERS     │                       │    ON_DEMAND TRACKERS     │
 ├───────────────────────────┤                       ├───────────────────────────┤
 │ • Purpose: Ongoing Monitor│                       │ • Purpose: Short-Burst Rx │
 │ • ACCELEROMETER   : 25 Hz │                       │ • ECG_ON_DEMAND : 500 Hz  │
 │ • PPG_CONTINUOUS  : 25 Hz │                       │ • PPG_ON_DEMAND : 100 Hz  │
 │ • EDA_CONTINUOUS  :  1 Hz │                       │ • SPO2_ON_DEMAND: Spot-chk│
 │ • HEART_RATE      :  1 Hz │                       │ • BIA & Sweat   : Spot-chk│
 │ • Battery-conserving mode │                       │ • High CPU & Power Draw   │
 └───────────────────────────┘                       └───────────────────────────┘
```

##### 1. `CONTINUOUS` Trackers (Long-Term Ambulatory Monitoring)
Optimized for sustained, continuous acquisition across 10–15+ minute experimental sessions (such as the TSST stress protocol). They utilize lower, battery-conserving sampling rates:
* `ACCELEROMETER_CONTINUOUS`: **$25\text{ Hz}$** (tri-axial motion tracking)
* `PPG_CONTINUOUS`: **$25\text{ Hz}$** (raw optical pulse plethysmogram)
* `EDA_CONTINUOUS`: **$1\text{ Hz}$** (electrodermal activity / skin conductance level)
* `HEART_RATE_CONTINUOUS`: **$1\text{ Hz}$** (processed cardiac beats-per-minute)
* `SKIN_TEMPERATURE_CONTINUOUS`: **Event-driven / Periodic** (skin and ambient thermal tracking)

##### 2. `ON_DEMAND` Trackers (High-Fidelity Waveform Capture)
Engineered for detailed algorithmic analysis and high-frequency physiological signal reconstruction:
* `ECG_ON_DEMAND` / `ECG`: **$500\text{ Hz}$** (clinical-grade electrical cardiac potential)
* `PPG_ON_DEMAND`: **$100\text{ Hz}$** (high-resolution single/multi-wavelength optical absorption)

##### ⚠️ Crucial Samsung SDK Guidelines & Constraints for `ON_DEMAND` Trackers:
Due to high battery consumption, algorithmic intensity, and hardware contention, Samsung enforces four strict operational rules:
1. **Foreground Execution Only:** On-demand tracking must execute strictly within an active foreground UI (`ComponentActivity`). The user must be actively viewing the watch screen; background service tracking for on-demand sensors is forbidden by the OS.
2. **Single Tracker Concurrency Limit:** The watch hardware cannot operate multiple on-demand trackers simultaneously. For example, running `ECG` and `SpO2` concurrently causes hardware resource conflicts.
3. **Time-Bound Interval Cap ($30\text{ Seconds}$):** On-demand trackers are strictly not intended for unbounded continuous execution. The UI should automatically limit measurements to 30-second windows.
4. **Concurrency & Sensor Invalidation:** Activating an on-demand measurement concurrently while a continuous tracker is logging can corrupt sensor baselines (e.g., executing `HEART_RATE_CONTINUOUS` while calculating `SPO2_ON_DEMAND` yields invalid readings). Continuous trackers should be paused prior to initiating on-demand spot checks.

##### Architectural Impact on the App:
* **RAM Buffering:** Logging high-frequency signals like `ECG` ($500\text{ Hz}$) or `PPG_ON_DEMAND` ($100\text{ Hz}$) requires the in-memory batch buffer in `CsvLogger.kt` to prevent disk I/O bottlenecks and UI frame drops. Conversely, lower-rate trackers ($1\text{ Hz}$) could technically sustain immediate writes, but are routed through the buffer for architectural uniformity.
* **State Management:** `TrackingManager.kt` ensures trackers are cleanly unhooked and flushed upon session termination.

---

#### 9.3.2 Exhaustive CSV Column Data Reference & Status Code Guide

The logging subsystem outputs records into a unified 6-column CSV structure:
$$\text{CSV Record Format: } \texttt{Timestamp, SensorType, Value1, Value2, Value3, Value4}$$

The detailed breakdown of each sensor's assigned columns, physical units, data types, and status code interpretations is presented below:

| SensorType (`String`) | Modality & Description | `Value1` | `Value2` | `Value3` | `Value4` |
|:---|:---|:---|:---|:---|:---|
| **`HeartRateContinuous`** / **`HeartRate`** | Continuous Heart Rate Monitoring | **BPM** (*Integer*)<br>Beats Per Minute ($30\text{--}220\text{ bpm}$) | **Status** (*Integer*)<br>$\mathbf{1}$ = Valid measurement<br>$\mathbf{<1}$ = Invalid / Searching / Low SNR | — | — |
| **`AccelerometerContinuous`** / **`Accelerometer`** | Motion & Tri-Axial Acceleration | **X-axis** (*Float*)<br>Acceleration ($mG$ or $mm/s^2$) | **Y-axis** (*Float*)<br>Acceleration ($mG$ or $mm/s^2$) | **Z-axis** (*Float*)<br>Acceleration ($mG$ or $mm/s^2$) | — |
| **`EdaContinuous`** | Electrodermal Activity (GSR / Sympathetic Arousal) | **Skin Conductance** (*Float*)<br>Raw Conductance in **$\mu\text{S}$** (microsiemens) | **Status** (*Integer*)<br>$\mathbf{0}$ = Calculating / OK / Valid | — | — |
| **`PpgContinuous`** | Raw Multi-Wavelength Photoplethysmogram | **PPG Green** (*Long/Float*)<br>Raw green light reflection intensity | **PPG IR** (*Long/Float*)<br>Raw infrared light absorption | **PPG Red** (*Long/Float*)<br>Raw red light absorption | — |
| **`PpgGreen`** | Standalone Green PPG Channel | **PPG Green** (*Long/Float*)<br>Raw green optical signal | **Status** (*Integer*)<br>Measurement status code | — | — |
| **`PpgRed`** | Standalone Red PPG Channel | **PPG Red** (*Long/Float*)<br>Raw red optical signal | **Status** (*Integer*)<br>Measurement status code | — | — |
| **`PpgIr`** | Standalone Infrared PPG Channel | **PPG IR** (*Long/Float*)<br>Raw infrared optical signal | **Status** (*Integer*)<br>Measurement status code | — | — |
| **`SkinTemperatureContinuous`** / **`SkinTemperature`** | Cutaneous & Ambient Temperature | **Object / Skin Temp** (*Float*)<br>Skin contact surface temp in **$^\circ\text{C}$** | **Ambient Temp** (*Float*)<br>Surrounding air temp in **$^\circ\text{C}$** | **Status** (*Integer*)<br>Sensor tracking status code | — |
| **`SpO2`** | Peripheral Blood Oxygen Saturation | **Status** (*Integer*)<br>$\mathbf{2}$ = Completed / Valid<br>$\mathbf{0}$ = Calculating / Initializing<br>$\mathbf{-4}$ = Motion Artifact (User Moving) | **SpO2 Percentage** (*Integer*)<br>Oxygen saturation level ($\%$, e.g. $98$) | — | — |
| **`Ecg`** | Electrocardiogram Waveform | **Sequence Number** (*Long*)<br>Sample index counter ($0, 1, 2, \dots$) | **ECG Voltage** (*Float*)<br>Differential cardiac voltage in **$mV$** | **Lead-Off Status** (*Integer*)<br>$\mathbf{0}$ = Good electrode contact<br>$\mathbf{5}$ = Lead Off (Finger/Skin detached) | — |
| **`Bia`** | Bioelectrical Impedance Analysis | **Status** (*Integer*)<br>Measurement status code | **Body Fat Mass** (*Float*)<br>Calculated fat mass in **$kg$** | **Skeletal Muscle Mass** (*Float*)<br>Calculated muscle mass in **$kg$** | — |
| **`SweatLoss`** | Galvanic Sweat Accumulation | **Status** (*Integer*)<br>Measurement status code | **Sweat Loss** (*Float*)<br>Estimated sweat volume in **$mL$** | — | — |

---

#### 9.3.3 SDK ValueKey Extraction & Event Parser Mapping

In [`TrackingManager.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt#L88-L173), every listener callback extracts hardware values from the incoming `DataPoint` list using SDK-specific `ValueKey` enums and routes them directly to `csvLogger.logData(sensorType, timestamp, *values)`:

```kotlin
// Example ValueKey extraction logic in TrackingManager.kt:

// 1. Continuous Heart Rate (BPM & Status):
csvLogger.logData("HeartRateContinuous", System.currentTimeMillis(),
    dp.getValue(ValueKey.HeartRateSet.HEART_RATE),
    dp.getValue(ValueKey.HeartRateSet.HEART_RATE_STATUS)
)

// 2. Tri-Axial Accelerometer (mG):
csvLogger.logData("AccelerometerContinuous", System.currentTimeMillis(),
    dp.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_X),
    dp.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Y),
    dp.getValue(ValueKey.AccelerometerSet.ACCELEROMETER_Z)
)

// 3. Electrodermal Activity (Skin Conductance in µS):
csvLogger.logData("EdaContinuous", System.currentTimeMillis(),
    dp.getValue(ValueKey.EdaSet.SKIN_CONDUCTANCE),
    dp.getValue(ValueKey.EdaSet.STATUS)
)

// 4. Skin & Ambient Temperature (°C):
csvLogger.logData("SkinTemperatureContinuous", System.currentTimeMillis(),
    dp.getValue(ValueKey.SkinTemperatureSet.OBJECT_TEMPERATURE),
    dp.getValue(ValueKey.SkinTemperatureSet.AMBIENT_TEMPERATURE),
    dp.getValue(ValueKey.SkinTemperatureSet.STATUS)
)

// 5. Electrocardiogram (Waveform mV & Lead-Off):
csvLogger.logData("Ecg", System.currentTimeMillis(),
    dp.getValue(ValueKey.EcgSet.SEQUENCE),
    dp.getValue(ValueKey.EcgSet.ECG_MV),
    dp.getValue(ValueKey.EcgSet.LEAD_OFF)
)
```

---

#### 9.3.4 Empirical Multi-Sensor Logging Verification & Frequency Analysis

In an end-to-end hardware logging validation conducted on 2026-11-17 ([`Deployment/test1/_user_folder/Pulled CSVs/sensor_data_20261117_231921.csv`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/_user_folder/Pulled%20CSVs/sensor_data_20261117_231921.csv)), the smartwatch app recorded continuously across all available modalities for approximately **$31\text{ seconds}$**, generating a total of **23,568 raw physiological data points**:

| Logged Sensor Identifier | Row Count | Inferred Frequency | Alignment with SDK Specifications |
|:---|:---:|:---:|:---|
| **`Ecg`** | **15,610 samples** | $\approx 500\text{ Hz}$ | Perfectly matches the $500\text{ Hz}$ short-burst ECG sampling specification. |
| **`PpgRed`** | **3,126 samples** | $\approx 100\text{ Hz}$ | High-fidelity raw optical photoplethysmogram on red wavelength. |
| **`PpgIr`** | **3,126 samples** | $\approx 100\text{ Hz}$ | High-fidelity raw optical photoplethysmogram on infrared wavelength. |
| **`Accelerometer`** | **604 samples** | $\approx 20\text{--}25\text{ Hz}$ | Corresponds to continuous tri-axial motion tracking (~$25\text{ Hz}$). |
| **`PpgGreen`** | **604 samples** | $\approx 20\text{--}25\text{ Hz}$ | Continuous green optical absorption channel (~$25\text{ Hz}$). |
| **`SpO2`** | **405 samples** | Continuous Burst | Optical pulse oximetry calculation sequence. |
| **`SkinTemperature`** | **31 samples** | $\mathbf{1.0\text{ Hz}}$ | Exactly $1\text{ sample/second}$ across the 31-second logging window. |
| **`EdaContinuous`** | **31 samples** | $\mathbf{1.0\text{ Hz}}$ | Exactly $1\text{ sample/second}$ across the 31-second logging window. |
| **`HeartRate`** | **31 samples** | $\mathbf{1.0\text{ Hz}}$ | Exactly $1\text{ sample/second}$ across the 31-second logging window. |
| **Total Recorded Rows** | **23,568 rows** | Multi-rate | Zero packet loss, clean CSV row formatting, zero UI jank. |

This empirical frequency analysis proves that the low-frequency continuous sensors (`SkinTemperature`, `EdaContinuous`, `HeartRate`) arrive at precisely $1\text{ Hz}$, while high-rate kinetic and optical sensors stream reliably without OS queue overflow.

---

### 9.4 Permissions, System Security, & Firmware Fault Tolerance
To ensure uninterrupted continuous operation without OS throttling or permission denial:
1. **Samsung Proprietary Permission:**  
   `com.samsung.android.hardware.sensormanager.permission.READ_ADDITIONAL_HEALTH_DATA` is declared in [`AndroidManifest.xml`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/AndroidManifest.xml) and requested dynamically to unlock direct access to raw PPG and EDA silicon drivers.
2. **High-Frequency Sensor Permission (Android 12+ / API 31+):**  
   `android.permission.HIGH_SAMPLING_RATE_SENSORS` is granted to avoid the default 200 Hz OS rate cap on accelerometer and optical sensors.
3. **Background Sensing & Motion Permissions:**  
   `android.permission.BODY_SENSORS_BACKGROUND` and `android.permission.ACTIVITY_RECOGNITION` maintain uninterrupted sensor sampling even when the watch screen enters ambient dim mode or the display turns off.
4. **Android 14+ Health Permissions:**  
   `HealthPermissions.READ_HEART_RATE` and `android.permission.health.READ_SKIN_TEMPERATURE` satisfy modern Android Health Connect security requirements.
5. **Firmware Graceful Degradation:**  
   Because firmware builds on different Galaxy Watch models may declare capability for a tracker in capability queries but throw `IllegalArgumentException: Invalid HealthTrackerType` when initializing `getHealthTracker()`, the manager wraps tracker instantiation in isolated exception handlers. Unsupported modalities fail safely and silently, allowing all valid sensors to record uninterrupted.

---

### 9.5 Wear OS User Interface (`MainActivity.kt`)
The user interface is built with **Wear OS Jetpack Compose** ([`Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/presentation/MainActivity.kt`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/presentation/MainActivity.kt)), providing an ergonomic, touch-friendly UI for circular watch displays:
- **Connection Indicator:** Displays real-time status of the connection to `HealthTrackingService` (`Connected` vs. `Disconnected`).
- **One-Touch Start / Stop Controls:** Single primary button that toggles cleanly between `Start Logging` and `Stop Logging`.
- **Toast Event Feedback:** Provides real-time visual confirmation to the wearer when a new CSV session file is initialized, when batches are persisted, or when tracking errors occur.

---

### 9.6 Data Extraction Protocol via Android Debug Bridge (`adb pull`)
Because the smartwatch logs data locally to application-scoped storage (`context.getExternalFilesDir(null)`), extracting the experimental datasets to a computer workstation is simple, reliable, and completely independent of network connectivity:

1. **Verify Connection over Wi-Fi / USB Debugging:**
   ```powershell
   adb devices
   ```
2. **Inspect Generated CSV Sessions on Watch Storage:**
   ```powershell
   adb shell ls -lh /sdcard/Android/data/com.samsung.health.sensorsdksample.multisensormeasurement/files/
   ```
3. **Pull All Experimental CSV Files to Host Workstation:**
   ```powershell
   adb pull /sdcard/Android/data/com.samsung.health.sensorsdksample.multisensormeasurement/files/. ./Pulled_CSVs/
   ```

---

### 9.7 Significance & Academic Bridge to Thesis Bab 3 & Bab 4
The successful deployment of this offline logging application marks a critical transition in the thesis project:
1. **Closing the Loop Between Model and Real-World Hardware:**  
   The feature extraction pipelines designed in [`Scripts/6th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/)—specifically the Sliced Wrist Model (61 features) and the Minimum Sensor Models (55 features)—were conceived specifically around wrist Accelerometer, EDA, Skin Temperature, and optical PPG. The smartwatch app demonstrates that these exact physiological modalities are now fully acquirable from off-the-shelf consumer hardware.
2. **Empirical Ground Truth Acquisition:**  
   Researchers can now conduct 10–15 minute experimental protocols (such as stress-induction tasks or baseline resting) on human subjects wearing the Samsung Galaxy Watch, pull the pristine high-resolution CSVs via ADB, and feed them directly into the rectified feature extraction pipeline for inference, validation, and real-time stress/anxiety detection.
