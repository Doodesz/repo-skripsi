# Model Output Specifications — `stai_2class_alkurdi_alignment_sliced_sensor`

> **Source Notebook:** [`Scripts/6th try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb)  
> **Dataset:** WESAD (Wearable Stress and Affect Detection)  
> **Task:** Binary Subjective Anxiety Classification (0 = No Anxiety Increase, 1 = Anxiety Increased)  
> **Sensor Modalities:** **Sliced Smartwatch Sensors Only** — Wrist Accelerometer (ACC), Wrist Electrodermal Activity (EDA), Wrist Skin Temperature (TEMP)  
> **Excluded Modalities:** All RespiBAN Chest Sensors (Chest ACC, ECG, Chest EDA, EMG, Respiration, Chest Temp) and Optical Photoplethysmography (BVP)  
> **Labeling Strategy:** Participant-relative STAI self-report (rectified STAI-6 questionnaire row-wise parsing)  
> **Sensor Clock Alignment:** Alkurdi `pd.to_datetime` outer-join clock alignment across differing frequencies  
> **Validation Paradigms:** Validation Scheme A (Naive Random Split 80/20) & Validation Scheme B (15-Fold Leave-One-Subject-Out CV)  

---

## 1. Experimental Motivation & Sliced Feasibility Rationale

As detailed in *Proposal Skripsi Bab 3*, practical consumer deployment of affective computing systems necessitates unobtrusive, wrist-worn wearable devices (such as smartwatches running Wear OS). Medical-grade research setups that rely on laboratory chest harnesses (e.g., chest ECG electrodes, respiration belts, chest EMG) present severe user encumbrance and cannot realistically transition to daily living environments.

Furthermore, empirical testing with the **Samsung Health Sensor SDK (Wear OS)** on real smartwatch hardware (documented in [[2026-10-29 Research Note]]) revealed that even though sensors like **ECG** and **BVP/PPG** physically exist on modern smartwatches (e.g., Samsung Galaxy Watch), the manufacturer SDK runtime **locks developers out with `PERMISSION_ERROR`** unless paired with specialized Samsung-exclusive apps (Samsung Health Monitor) and specific regional regulatory approvals. 

Therefore, true wearable feasibility must be established on the **sensor modalities that are genuinely accessible to third-party developers**:
1. **Feasible & Accessible Sensors Retained:**
   - **Wrist 3-Axis Accelerometer (ACC - 32 Hz):** Captures physical motion patterns, behavioral agitation, tremor bursts, and rest states.
   - **Wrist Electrodermal Activity (EDA - 4 Hz):** Captures sympathetic nervous system arousal via tonic skin conductance levels (SCL) and phasic skin conductance responses (SCR).
   - **Wrist Skin Temperature (TEMP - 4 Hz):** Captures autonomic peripheral vasomotor tone (stress-induced vasoconstriction causing peripheral cooling).
2. **Excluded Sensors (SDK-Restricted or Encumbering):**
   - **Optical BVP & Wrist ECG:** Excluded due to Samsung Health Sensor SDK runtime permission blocks (`PERMISSION_ERROR`) and third-party developer access restrictions ([[2026-10-29 Research Note]]).
   - **RespiBAN Chest Suite (700 Hz):** Chest ACC, ECG (Heart Rate Variability), Chest EDA, EMG (trapezius muscle tension), Respiration dynamics, and Chest Temperature (excluded due to wearable harness encumbrance).
3. **Feature Dimensionality Reduction:**
   - The feature space is reduced from **155 full-modality features** down to **62 wrist-only features** (60.0% reduction in feature footprint).

---

## 2. Dataset & Cohort Statistics

| Parameter | Value |
|---|---|
| Dataset | WESAD (Wearable Stress and Affect Detection) |
| Total Participants | 15 (S2–S11, S13–S17; S1 & S12 absent in public release) |
| Total Sliced Datasets | 15 pickle files (`Data/6th try/2class/Sliced_sensor_modals/S{id}_stai_2class_60s_shift6s_sliced.pkl`) |
| Window Length | 60 seconds (bounded within condition protocol boundaries) |
| Window Shift (Stride) | 6 seconds (90% temporal overlap) |
| Total Cohort Windows | **4,095 windows** |
| Total Features per Window | **62 predictive features** + 12 metadata columns (74 columns total) |
| Class 0 Windows (No Anxiety Increase, $\Delta \le 0$) | **1,785 windows (43.59%)** |
| Class 1 Windows (Anxiety Increased, $\Delta > 0$) | **2,310 windows (56.41%)** |
| Zero-Inflation Status | **0% (15/15 participants exhibit verified positive TSST anxiety windows)** |

### Per-Subject Window Breakdown
| Subject | Total Windows | Class 0 ($\Delta \le 0$) | Class 1 ($\Delta > 0$) | TSST Assigned Class |
| :---: | :---: | :---: | :---: | :---: |
| **S2** | 257 | 166 (64.6%) | 91 (35.4%) | Class 1 ($\Delta = +10$) |
| **S3** | 260 | 48 (18.5%) | 212 (81.5%) | Class 1 ($\Delta = +6$) |
| **S4** | 272 | 55 (20.2%) | 217 (79.8%) | Class 1 ($\Delta = +4$) |
| **S5** | 270 | 113 (41.9%) | 157 (58.1%) | Class 1 ($\Delta = +7$) |
| **S6** | 269 | 111 (41.3%) | 158 (58.7%) | Class 1 ($\Delta = +8$) |
| **S7** | 275 | 56 (20.4%) | 219 (79.6%) | Class 1 ($\Delta = +6$) |
| **S8** | 263 | 103 (39.2%) | 160 (60.8%) | Class 1 ($\Delta = +6$) |
| **S9** | 262 | 113 (43.1%) | 149 (56.9%) | Class 1 ($\Delta = +8$) |
| **S10** | 289 | 110 (38.1%) | 179 (61.9%) | Class 1 ($\Delta = +7$) |
| **S11** | 285 | 178 (62.5%) | 107 (37.5%) | Class 1 ($\Delta = +3$) |
| **S13** | 280 | 116 (41.4%) | 164 (58.6%) | Class 1 ($\Delta = +9$) |
| **S14** | 282 | 176 (62.4%) | 106 (37.6%) | Class 1 ($\Delta = +8$) |
| **S15** | 282 | 174 (61.7%) | 108 (38.3%) | Class 1 ($\Delta = +5$) |
| **S16** | 274 | 164 (59.9%) | 110 (40.1%) | Class 1 ($\Delta = +6$) |
| **S17** | 275 | 102 (37.1%) | 173 (62.9%) | Class 1 ($\Delta = +9$) |
| **TOTAL** | **4,095** | **1,785 (43.59%)** | **2,310 (56.41%)** | **15/15 (100%)** |

---

## 3. Feature Taxonomy (62 Predictive Features)

| Sensor Modality | Feature Count | Specific Feature Columns | Extraction Technique & Rationale |
|---|:---:|---|---|
| **Wrist Accelerometer (Basic Window Stats)** | 16 | `mean`, `std`, `min`, `max` for `ACC_x`, `ACC_y`, `ACC_z`, `net_acc` | Captures overall kinetic exertion and arm posture across the 60s window. |
| **Wrist Accelerometer (Sub-Window Aggregations)** | 18 | `ACC_sub_{col}_{stat}_mean`, `ACC_sub_{col}_{stat}_std` for `col` $\in$ {`ACC_x`, `ACC_y`, `ACC_z`} and `stat` $\in$ {`std`, `abs_integral`, `peak_freq`} | 5-second sub-windows sliding by 0.25 seconds (221 sub-windows per main window). Captures micro-movement variance, tremor bursts, and kinetic energy dynamics. |
| **Wrist Accelerometer (Kinetic Dynamics)** | 8 | `ACC_subwindow_count`, `net_acc_abs_integral`, `ACC_x_abs_integral`, `ACC_y_abs_integral`, `ACC_z_abs_integral`, `ACC_x_peak_freq`, `ACC_y_peak_freq`, `ACC_z_peak_freq` | Measures cumulative integral energy and dominant movement frequency per axis. |
| **Wrist Electrodermal Activity (EDA)** | 14 | `mean`, `std`, `min`, `max` for raw `EDA`, phasic `EDA_phasic` (SCR), and tonic `EDA_tonic` (SCL); `EDA_slope`, `EDA_drange` | NeuroKit2 phasic/tonic decomposition after Butterworth lowpass ($f_c = 1.0\text{ Hz}$). Measures sympathetic arousal, sweat gland conductance, and trend trajectory. |
| **Wrist Skin Temperature (TEMP)** | 6 | `mean`, `std`, `min`, `max` for `TEMP`; `TEMP_slope`, `TEMP_drange` | Native pass-through with linear slope and range extraction. Measures peripheral vasoconstriction and cooling rate. |
| **TOTAL** | **62** | — | **Smartwatch-Wearable Only Feature Vector** |

---

## 4. Evaluation Benchmark Results

### 4.1 Validation Scheme A: Naive Random Split (80/20)
*Evaluates the Alkurdi baseline split (`train_test_split(test_size=0.2, random_state=0)`) demonstrating the severe inflation caused by temporal autocorrelation and inter-subject leakage.*

| Model | Test Accuracy | Note on Performance |
|---|:---:|---|
| **Random Forest (`max_depth=25`)** | **100.00%** | Memorizes overlapping temporal frames across identical subjects. |
| **XGBoost** | **100.00%** | Memorizes overlapping temporal frames across identical subjects. |
| **KNN ($k=5$)** | **97.56%** | Finds near-identical neighbor windows from the same subject in training. |
| **SVM (RBF)** | **95.36%** | Highly inflated test accuracy due to subject contamination. |
| **SVM Linear ($C=1.0$)** | **82.05%** | Separates leaked linear manifolds. |
| **LDA** | **79.61%** | Projective linear separation on leaked training frames. |

---

### 4.2 Validation Scheme B: 15-Fold Leave-One-Subject-Out (LOSO) Cross-Validation
*Evaluates true subject-independent generalization. In each fold, 1 subject is strictly held out for testing while the remaining 14 subjects are used for training. Feature standardization is computed strictly within-fold to prevent leakage.*

| Model | Pooled Accuracy | Pooled Balanced Acc | Pooled Macro F1 | Class 0 F1 (No Anxiety) | Class 1 F1 (Anxiety Inc) | Macro-Subject Accuracy | Macro-Subject Balanced Acc | Macro-Subject Macro F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (`max_depth=4`, Alkurdi)** | **71.14%** | **71.22%** | **0.7093** | **0.6846** | **0.7339** | **71.36%** | **76.35%** | **0.7010** |
| **Random Forest (`max_depth=25`)** | **69.87%** | **69.62%** | **0.6951** | **0.6621** | **0.7281** | **70.11%** | **74.19%** | **0.6861** |
| **LDA** | **67.62%** | **67.94%** | **0.6749** | **0.6549** | **0.6950** | **67.64%** | **72.97%** | **0.6700** |
| **XGBoost** | **66.37%** | **66.18%** | **0.6604** | **0.6265** | **0.6942** | **66.52%** | **70.54%** | **0.6476** |
| **SVM (RBF)** | **63.91%** | **63.35%** | **0.6334** | **0.5878** | **0.6790** | **63.91%** | **64.74%** | **0.6088** |
| **SVM Linear ($C=1.0$)** | **63.17%** | **63.00%** | **0.6285** | **0.5935** | **0.6634** | **63.20%** | **68.56%** | **0.6211** |
| **SVM Linear ($C=0.9$)** | **63.15%** | **62.98%** | **0.6282** | **0.5934** | **0.6631** | **63.17%** | **68.57%** | **0.6209** |
| **KNN ($k=5$)** | **58.32%** | **57.26%** | **0.5728** | **0.5062** | **0.6393** | **58.29%** | **57.75%** | **0.5529** |

---

## 5. Comparative Shortfall Analysis: Sliced Modalities vs. Full-Modality Suite

This section evaluates the exact performance penalty incurred when discarding all chest-worn sensors and optical BVP (reducing the feature set from 155 to 62 features) across both **Validation Scheme B (15-Fold Leave-One-Subject-Out CV)** and **Validation Scheme A (Naive Random Split 80/20)**.

$$\text{Shortfall} = \text{Metric}_{\text{Full}} - \text{Metric}_{\text{Sliced}} \quad\Big|\quad \text{Retention Rate} = \frac{\text{Metric}_{\text{Sliced}}}{\text{Metric}_{\text{Full}}} \times 100\%$$

---

### 5.1 Validation Scheme B: 15-Fold Leave-One-Subject-Out (LOSO) Cross-Validation Shortfall
*Measures true subject-independent generalization shortfall without inter-subject data leakage:*

| Classifier Model | Full Modality LOSO Acc (155 feats) | Sliced Sensor LOSO Acc (62 feats) | LOSO Acc Shortfall ($\Delta$) | LOSO Retention Rate | Full LOSO Bal Acc | Sliced LOSO Bal Acc | Full LOSO Macro F1 | Sliced LOSO Macro F1 | Full C0 / C1 F1 | Sliced C0 / C1 F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (`max_depth=4`, Alkurdi)** | **74.43%** | **71.14%** | **+3.29%** | **95.57%** | 74.90% | **71.22%** | 0.7434 | **0.7093** | 0.7281 / 0.7587 | **0.6846 / 0.7339** |
| **Random Forest (`max_depth=25`)** | 73.72% | **69.87%** | **+3.85%** | **94.77%** | 74.11% | **69.62%** | 0.7361 | **0.6951** | 0.7191 / 0.7532 | **0.6621 / 0.7281** |
| **SVM Linear ($C=1.0$)** | 64.96% | **63.17%** | **+1.79%** | **97.25%** | 64.64% | **63.00%** | 0.6455 | **0.6285** | 0.6074 / 0.6836 | **0.5935 / 0.6634** |
| **SVM Linear ($C=0.9$)** | 64.98% | **63.15%** | **+1.83%** | **97.18%** | 64.70% | **62.98%** | 0.6459 | **0.6282** | 0.6088 / 0.6830 | **0.5934 / 0.6631** |
| **LDA** | 62.66% | **67.62%** | **-4.96% (Gain)** | **107.91%** | 62.70% | **67.94%** | 0.6244 | **0.6749** | 0.5952 / 0.6535 | **0.6549 / 0.6950** |
| **XGBoost** | 74.14% | **66.37%** | **+7.77%** | **89.52%** | 74.24% | **66.18%** | 0.7394 | **0.6604** | 0.7168 / 0.7621 | **0.6265 / 0.6942** |
| **SVM (RBF)** | 72.04% | **63.91%** | **+8.13%** | **88.71%** | 71.59% | **63.35%** | 0.7158 | **0.6334** | 0.6797 / 0.7519 | **0.5878 / 0.6790** |
| **KNN ($k=5$)** | 67.64% | **58.32%** | **+9.32%** | **86.21%** | 68.53% | **57.26%** | 0.6763 | **0.5728** | 0.6703 / 0.6823 | **0.5062 / 0.6393** |

---

### 5.2 Validation Scheme A: Naive Random Split (80/20) Shortfall & Leakage Comparison
*Evaluates the naive split (`train_test_split(test_size=0.2, random_state=0)`) to contrast with LOSO subject independence:*

| Classifier Model | Full Modality Scheme A Acc | Sliced Sensor Scheme A Acc | Scheme A Acc Shortfall ($\Delta$) | Scheme A vs. LOSO Sliced Discrepancy (Leakage Gap) |
|---|:---:|:---:|:---:|:---:|
| **Random Forest (`max_depth=25`)** | 100.00% | **100.00%** | **0.00%** | **+30.13%** (100.00% vs 69.87% LOSO) |
| **XGBoost** | 99.76% | **100.00%** | **-0.24%** | **+33.63%** (100.00% vs 66.37% LOSO) |
| **KNN ($k=5$)** | 99.39% | **97.56%** | **+1.83%** | **+39.24%** (97.56% vs 58.32% LOSO) |
| **SVM (RBF)** | 100.00% | **95.36%** | **+4.64%** | **+31.45%** (95.36% vs 63.91% LOSO) |
| **SVM Linear ($C=1.0$)** | 96.58% | **82.05%** | **+14.53%** | **+18.88%** (82.05% vs 63.17% LOSO) |
| **LDA** | 93.77% | **79.61%** | **+14.16%** | **+11.99%** (79.61% vs 67.62% LOSO) |

> [!WARNING]
> **Data Leakage Persists in Sliced Features under Scheme A:** Even with 60% of features removed (wrist sensors only), naive random splitting still yields pseudo-perfect test accuracies (82% to 100%). This confirms that high random-split performance is an artifact of inter-subject window leakage and temporal autocorrelation rather than true affective signal modeling. Only **Validation Scheme B (LOSO CV)** provides scientifically valid evaluation for real-world smartwatch deployment.

---

## 6. Key Scientific Findings & Discussion

### 1. High Wearable Retention Rate (>95% for Random Forest)
The top-performing classifier, **Random Forest (`max_depth=4`)**, achieves **71.14% subject-independent accuracy** with sliced wrist sensors compared to **74.43%** with the full 155-modality chest-and-wrist suite. This represents a minimal shortfall of only **3.29%**, retaining **95.57%** of the full multimodal predictive power. This proves that an unobtrusive consumer smartwatch setup (measuring only wrist movement, galvanic skin response, and skin temperature) is scientifically viable for continuous subjective state anxiety detection.

### 2. Dimensionality Reduction Benefits Linear Discriminant Analysis (LDA)
Remarkably, **LDA performed +4.96% better** on the sliced sensor subset (67.62% vs. 62.66%, a 107.91% retention gain). In the 155-feature space, LDA suffered from severe multicollinearity and near-singular pooled covariance matrices caused by cross-channel redundancy (e.g., chest vs. wrist EDA, multiple ECG HRV time/frequency domains, respiration cycle features). Reducing the feature space to 62 non-redundant wrist features regularized the discriminant projection and eliminated over-parameterization noise.

### 3. Tree Depth & Regularization in Wearable Slices
Similar to the full-modality experiment, shallow decision trees (`max_depth=4`, Alkurdi specification) outperformed deep trees (`max_depth=25`) by **+1.27%** (71.14% vs. 69.87%). Restricting tree depth acts as an effective regularizer against inter-subject physiological variance, preventing the forest from overfitting to idiosyncratic subject baselines in the training folds.

### 4. Sensitivity to Loss of Cardiac (ECG) Features in Distance-Based & Kernel Models
Non-tree models that rely on Euclidean metric distances or kernel hyperplanes (KNN, SVM RBF) experienced the steepest performance drops (KNN shortfall: +9.32%, SVM RBF shortfall: +8.13%). These models heavily leveraged the compact, high-SNR geometric separation provided by chest ECG Heart Rate Variability (HRV) metrics (SDNN, RMSSD, pNN50). In contrast, ensemble decision trees adapted smoothly to the remaining wrist EDA phasic peaks and skin temperature trajectories.

---

## 7. Artifacts & Deliverables Summary

1. **Jupyter Notebook:**
   - [`Scripts/6th try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb) (executed with all interactive tables, confusion matrices, and shortfall comparisons).
2. **Processed Dataset Directory:**
   - [`Data/6th try/2class/Sliced_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Sliced_sensor_modals/) (contains 15 subject pickle files `S{id}_stai_2class_60s_shift6s_sliced.pkl`).
3. **Model Artifacts Directory:**
   - [`Models/6th try/2class/Sliced_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Sliced_sensor_modals/) (contains all trained `.joblib` model binaries, [`evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Sliced_sensor_modals/evaluation_summary.json), and [`per_subject_loso_results.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Sliced_sensor_modals/per_subject_loso_results.json)).
