# AI Validation Report — `Model Specs` Notes Folder

> **Audited Notes Directory:** [`_Obsidian Notes/Model Specs/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/)  
> **Audited Documents:**  
> 1. [`Model Output Specs 5th Try.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%205th%20Try.md)  
> 2. [`Model Output Specs 6th Try 2-Class.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class.md)  
> 3. [`Model Output Specs 6th Try 3-Class.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%203-Class.md)  
> 4. [`Model Output Specs 6th Try 2-Class Sliced Sensor.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class%20Sliced%20Sensor.md)  
> 5. [`Model Output Specs 6th Try 2-Class Minimum Sensor.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%206th%20Try%202-Class%20Minimum%20Sensor.md)  
> **Initial Audit Date:** 2026-11-16  
> **Revalidation & Remediation Date:** 2026-11-16  
> **Verification Harness:** Direct comparison against Python scripts, Jupyter notebooks, serialized models, evaluation summaries, and raw WESAD questionnaire CSVs.

---

## 1. Executive Summary & Verification Verdict

An exhaustive audit and subsequent revalidation of all five specification documents in [`_Obsidian Notes/Model Specs/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/) was conducted against the underlying code, datasets, evaluation summaries, and serialized model binaries.

### Overall Revalidation Verdict: **FULLY RESOLVED & 100% VALID**
All previously identified discrepancies have been resolved and verified across the notes:
1. **System Paths Realigned & Verified:**
   - 5th Try note paths updated from legacy directories (`Models/stai_6s_standardized/`, `Data/Preprocessed n STAI-labeled...`) to active repository locations: [`Models/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/5th%20try/) and [`Data/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/5th%20try/).
   - 6th Try 2-Class note paths updated to explicit subdirectories: [`Models/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Full_sensor_modals/) and [`Data/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Full_sensor_modals/).
2. **Table 2.3 Corrected in 6th Try 2-Class Spec:**
   - Baseline sums, condition sums, score shifts ($\Delta$), and binary labels for participants S3–S17 have been corrected to match the raw WESAD questionnaire files (`WESAD/SX/SX_quest.csv`) and extracted pickle DataFrames (e.g., S3 Base=9.0, TSST=20.0, $\Delta = +11$; S4 Base=9.0, TSST=17.0, $\Delta = +8$).
   - Sliced Sensor and Minimum Sensor specs' per-subject TSST delta columns have also been synchronized.
3. **Methodological Transposition Bug Documented in 5th Try Spec:**
   - Section 2.1 in [`Model Output Specs 5th Try.md`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_Obsidian%20Notes/Model%20Specs/Model%20Output%20Specs%205th%20Try.md) now includes a clear callout warning explaining that the 5th try code implemented `responses = [row[index] for row in stai_rows]`, which parsed only 5 items and transposed items across condition administrations, causing the 8 zero-anxiety subjects.

---

## 2. Revalidation Audit per Specification Document

### 2.1 Revalidation of `Model Output Specs 5th Try.md`

| Item / Claim | Document State | Repo Ground Truth | Revalidation Status |
|---|---|---|:---:|
| **Source Notebook** | [`Scripts/5th try/stai_6s_reference_models_stdized.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/5th%20try/stai_6s_reference_models_stdized.ipynb) | Path exists, contains 17 executed cells | **VERIFIED** |
| **Participants** | 15 subjects (S2–S11, S13–S17; S1 & S12 absent) | `SUBJECTS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]` | **VERIFIED** |
| **STAI Questionnaire Warning** | Section 2.1 contains callout warning detailing indexing transposition bug | Matches `parse_quest_file` code in Cell 3 (`responses = [row[index] for row in stai_rows]`) | **RESOLVED & VERIFIED** |
| **Total Cohort Windows** | 4,147 windows (3,404 No Anxiety [82.1%], 743 Anxiety [17.9%]) | Evaluated on [`Data/5th try/*.pkl`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/5th%20try/) | **VERIFIED (EXACT)** |
| **Feature Dimensions** | 154 predictive features + 7 metadata columns (161 total) | Confirmed across 5th try pickle files | **VERIFIED (EXACT)** |
| **Scaling Pipeline** | `StandardScaler -> MinMaxScaler` inside folds | Verified in notebook Cell 13 `Pipeline` | **VERIFIED** |
| **GroupShuffleSplit Results** | Test: S3, S11, S15 ($N=844$). RF=0.8602, SVM=0.8507, XGB=0.8815, KNN=0.8922 | Matches notebook Cell 13 printed outputs | **VERIFIED (EXACT)** |
| **LOSO Results (15 Folds)** | Pooled Acc: RF=0.8408, SVM=0.8741, XGB=0.8449, KNN=0.8635. Bal Acc: RF=0.6716, SVM=0.7061, XGB=0.7078, KNN=0.7022 | Matches notebook Cell 15 printed outputs | **VERIFIED (EXACT)** |
| **Saved Model Paths** | Updated to [`Models/5th try/stai_baseline_relative_{model}_model.joblib`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/5th%20try/) in Sections 6.3 & 10 | Files exist on disk at [`Models/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/5th%20try/) | **RESOLVED & VERIFIED** |
| **Data Artifact Paths** | Updated to [`Data/5th try/S{id}_stai_60s_shift6s.pkl`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/5th%20try/) in Section 10 | Files exist on disk at [`Data/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/5th%20try/) | **RESOLVED & VERIFIED** |
| **Alkurdi `_forked` Comparison** | LDA: 93.64%, RF (`d=4`): 94.49%, SVM Lin C=1.0: 97.46%, SVM Lin C=0.9: 97.88% | [`_forked/Feature Based Model WESAD.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/_forked/Feature%20Based%20Model%20WESAD.ipynb) | **VERIFIED (EXACT)** |

---

### 2.2 Revalidation of `Model Output Specs 6th Try 2-Class.md`

| Item / Claim | Document State | Repo Ground Truth | Revalidation Status |
|---|---|---|:---:|
| **Source Notebook** | [`Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/stai_2class_alkurdi_alignment.ipynb) | Path exists, contains 17 executed cells | **VERIFIED** |
| **Task Formulation** | Binary: Class 0 ($\Delta_{\text{sum}} \le 0$) vs. Class 1 ($\Delta_{\text{sum}} > 0$) | Verified in notebook Cell 4 | **VERIFIED** |
| **Zero-Inflation Status** | 100% of subjects (15/15) exhibit positive TSST anxiety ($\Delta > 0$) | Verified on all 15 pickles in [`Data/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Full_sensor_modals/) | **VERIFIED** |
| **Table 2.3 STAI Scores** | Corrected Table 2.3: S3 Base=9.0, TSST=20.0 ($\Delta = +11$); S4 Base=9.0, TSST=17.0 ($\Delta = +8$), etc. | Matches raw [`WESAD/SX/SX_quest.csv`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/WESAD/) and pickle files 100% | **RESOLVED & VERIFIED** |
| **Cohort Window Distribution** | Total: 4,095 windows. Class 0: 1,785 (43.59%), Class 1: 2,310 (56.41%) | Verified across all 15 pickle files | **VERIFIED (EXACT)** |
| **Feature Taxonomy** | 155 predictive features + 12 metadata columns (167 total) | Verified on pickle DataFrame columns | **VERIFIED (EXACT)** |
| **Scaling Pipeline** | Single-stage `StandardScaler` (omission of `MinMaxScaler`) | Verified in notebook Cell 13 & [`Scripts/Utils/generate_2class_data_and_eval.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/Utils/generate_2class_data_and_eval.py) | **VERIFIED** |
| **Scheme A Results (Naive 80/20)** | RF (`d=25`)=100%, SVM RBF=100%, XGB=99.76%, KNN=99.39%, SVM Lin=96.58%, RF (`d=4`)=96.21%, LDA=93.77% | [`Models/6th try/2class/Full_sensor_modals/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Full_sensor_modals/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Scheme B (LOSO) Summary** | Pooled Acc: RF (`d=4`)=74.43%, XGB=74.14%, RF (`d=25`)=73.72%, SVM RBF=72.04%, KNN=67.64%, SVM Lin=64.96%, LDA=62.66% | [`Models/6th try/2class/Full_sensor_modals/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Full_sensor_modals/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **File Storage Paths** | Updated to [`Models/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Full_sensor_modals/) and [`Data/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Full_sensor_modals/) in Sections 6.3 & 10 | Directories and files exist on disk | **RESOLVED & VERIFIED** |

---

### 2.3 Revalidation of `Model Output Specs 6th Try 3-Class.md`

| Item / Claim | Document State | Repo Ground Truth | Revalidation Status |
|---|---|---|:---:|
| **Source Notebook** | [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb) | Path exists, contains 17 executed cells | **VERIFIED** |
| **3-Class Formulation** | C0: $\Delta_{\text{norm}} < -\frac{1}{24}$; C1: $[-\frac{1}{24}, \frac{1}{24}]$; C2: $\Delta_{\text{norm}} > \frac{1}{24}$ | Evaluated in notebook Cell 4 & [`Scripts/Utils/relabel_and_evaluate_6th_try.py`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/Utils/relabel_and_evaluate_6th_try.py) | **VERIFIED** |
| **Table 2.3 STAI Scores** | Raw STAI sums, $\Delta_{\text{norm}}$, and assigned 3-class labels | Verified against raw [`WESAD/SX/SX_quest.csv`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/WESAD/) | **VERIFIED (EXACT)** |
| **Cohort Window Distribution** | Total: 4,095 windows. C0: 1,110 (27.11%), C1: 1,026 (25.05%), C2: 1,959 (47.84%) | Verified on [`Data/6th try/3class/*.pkl`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/3class/) | **VERIFIED (EXACT)** |
| **Scheme A Results (Naive 80/20)** | RF (`d=25`)=100%, XGB=100%, SVM RBF=99.51%, KNN=99.39%, SVM Lin=99.27%, LDA=94.51%, RF (`d=4`)=92.80% | [`Models/6th try/3class/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Scheme B (LOSO) Results** | SVM RBF Pooled Acc=57.22% (C2 F1=0.7965); SVM Lin C=1.0=56.39%; XGB=55.58%; LDA=55.56%; RF (`d=25`)=53.21%; RF (`d=4`)=52.94%; KNN=51.06% | [`Models/6th try/3class/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Per-Subject Detailed Results** | Section 8.3 per-subject tables for SVM RBF, SVM Linear, XGBoost, and RF (`d=4`) | [`Models/6th try/3class/per_subject_loso_results.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/per_subject_loso_results.json) | **VERIFIED (EXACT)** |
| **Output File Locations** | [`Models/6th try/3class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/3class/) & [`Data/6th try/3class/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/3class/) | Verified on disk | **VERIFIED (EXACT)** |

---

### 2.4 Revalidation of `Model Output Specs 6th Try 2-Class Sliced Sensor.md`

| Item / Claim | Document State | Repo Ground Truth | Revalidation Status |
|---|---|---|:---:|
| **Source Notebook** | [`Scripts/6th try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Sliced_sensor_modals/stai_2class_alkurdi_alignment_sliced_sensor.ipynb) | Path exists, executed with all tables and plots | **VERIFIED** |
| **Wearable Feasibility Rationale** | Samsung Health Sensor SDK locks ECG and continuous raw BVP with `PERMISSION_ERROR` (code 0) | Documented in [[2026-10-29 Research Note]] from Wear OS hardware deployment | **VERIFIED** |
| **Retained Sensor Suite** | Wrist ACC (42 features) + Wrist EDA (14 features) + Wrist TEMP (6 features) = 62 features | Verified on [`Data/6th try/2class/Sliced_sensor_modals/*.pkl`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Sliced_sensor_modals/) | **VERIFIED (EXACT)** |
| **Per-Subject TSST Delta** | Table in Section 2 updated to match true deltas (S3: $\Delta = +11$, S4: $\Delta = +8$, etc.) | Verified against raw questionnaire files and pickles | **RESOLVED & VERIFIED** |
| **Scheme A Results (Naive 80/20)** | RF (`d=25`)=100%, XGB=100%, KNN=97.56%, SVM RBF=95.36%, SVM Lin=82.05%, LDA=79.61% | [`Models/6th try/2class/Sliced_sensor_modals/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Sliced_sensor_modals/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Scheme B (LOSO) Results** | RF (`d=4`)=71.14%, RF (`d=25`)=69.87%, LDA=67.62%, XGB=66.37%, SVM RBF=63.91%, SVM Lin=63.17%, KNN=58.32% | [`Models/6th try/2class/Sliced_sensor_modals/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Sliced_sensor_modals/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Shortfall & Retention Analysis** | RF (`d=4`) shortfall = +3.29% (95.57% retention); LDA shortfall = **-4.96% (107.91% retention / GAIN)** | Verified against mathematical formula and model metrics | **VERIFIED (EXACT)** |

---

### 2.5 Revalidation of `Model Output Specs 6th Try 2-Class Minimum Sensor.md`

| Item / Claim | Document State | Repo Ground Truth | Revalidation Status |
|---|---|---|:---:|
| **Source Notebooks** | 3 pairwise minimal notebooks in [`Scripts/6th try/2class/Minimum_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/Minimum_sensor_modals/): `eda_acc` (56f), `eda_temp` (20f), `acc_temp` (48f) | All 3 notebooks exist and are fully executed | **VERIFIED** |
| **Modality Rationale** | EDA (39%–62%) and ACC (17%–28%) dominate decision weight; TEMP is sluggish and ambient-sensitive | Analyzed in [`Scripts/6th try/analyze_sensor_modality_weight.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/analyze_sensor_modality_weight.ipynb) | **VERIFIED** |
| **Per-Subject TSST Delta** | Table in Section 2 updated to match true deltas (S3: $\Delta = +11$, S4: $\Delta = +8$, etc.) | Verified against raw questionnaire files and pickles | **RESOLVED & VERIFIED** |
| **Pair A (EDA + ACC - 56f) LOSO** | RF (`d=25`)=73.11%, RF (`d=4`)=72.33%, LDA=70.70%, XGB=68.69%, SVM RBF=66.72%, SVM Lin=65.47%, KNN=62.88% | [`Models/6th try/2class/Minimum_sensor_modals/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Minimum_sensor_modals/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Pair B (EDA + TEMP - 20f) LOSO** | LDA=**71.18%**, SVM RBF=**69.21%**, SVM Lin=**68.55%**, KNN=**66.62%**, RF (`d=25`)=69.50%, XGB=66.86% | [`Models/6th try/2class/Minimum_sensor_modals/eda_temp/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Minimum_sensor_modals/eda_temp/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Pair C (ACC + TEMP - 48f) [NO EDA]** | RF (`d=25`)=60.88%, RF (`d=4`)=58.39%, LDA=60.61%, XGB=60.63%, SVM RBF=57.09%, SVM Lin=55.43%, KNN=54.73% | [`Models/6th try/2class/Minimum_sensor_modals/acc_temp/evaluation_summary.json`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Minimum_sensor_modals/acc_temp/evaluation_summary.json) | **VERIFIED (EXACT)** |
| **Cross-Pair Benchmark (Table 5.1)** | Comprehensive cross-pair comparison across Full (155f), Sliced (62f), Pair A (56f), Pair B (20f), Pair C (48f) | Cross-validated against all four corresponding `evaluation_summary.json` files | **VERIFIED (EXACT)** |

---

## 3. Consolidated Cross-Iteration Benchmark Matrix (Empirically Validated)

| Dimension / Metric | 5th Try (`5th try/`) | 6th Try 3-Class (`3class/`) | 6th Try 2-Class Full (`2class/Full`) | 6th Try 2-Class Sliced (`2class/Sliced`) | 6th Try Pair A (EDA+ACC) | 6th Try Pair B (EDA+TEMP) | Alkurdi Baseline (`_forked/`) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Target Ground Truth** | Subjective STAI (Buggy 5 items) | Subjective STAI (Fixed 6 items) | Subjective STAI (Fixed 6 items) | Subjective STAI (Fixed 6 items) | Subjective STAI (Fixed 6 items) | Subjective STAI (Fixed 6 items) | Objective Protocol Stimulus |
| **Classification Task** | Binary: $\Delta \ge 0.2$ | 3-Class: $\Delta < -\frac{1}{24}$, $\pm \frac{1}{24}$, $\Delta > \frac{1}{24}$ | Binary: $\Delta \le 0$ vs. $\Delta > 0$ | Binary: $\Delta \le 0$ vs. $\Delta > 0$ | Binary: $\Delta \le 0$ vs. $\Delta > 0$ | Binary: $\Delta \le 0$ vs. $\Delta > 0$ | 3-Class Protocol: 0, 1, 2 |
| **Sensors Used** | Full Chest + Wrist | Full Chest + Wrist | Full Chest + Wrist | Wrist ACC, EDA, TEMP | Wrist ACC, EDA | Wrist EDA, TEMP | Full Chest + Wrist |
| **Feature Dimension ($X$)** | 154 features | 155 features | 155 features | 62 features | 56 features | 20 features | 120 features |
| **Cohort Windows ($N$)** | 4,147 windows | 4,095 windows | 4,095 windows | 4,095 windows | 4,095 windows | 4,095 windows | 577 windows |
| **Class Balance** | 82.1% C0 / 17.9% C1 | 27.1% C0 / 25.1% C1 / 47.8% C2 | 43.6% C0 / 56.4% C1 | 43.6% C0 / 56.4% C1 | 43.6% C0 / 56.4% C1 | 43.6% C0 / 56.4% C1 | 54.1% C1 / 30.3% C2 / 15.6% C0 |
| **Zero-Inflation Status** | Severe (8/15 subjects had 0 anx) | **Resolved (15/15 have C2)** | **Resolved (15/15 have C1)** | **Resolved (15/15 have C1)** | **Resolved (15/15 have C1)** | **Resolved (15/15 have C1)** | N/A (protocol-driven) |
| **Scaling Method** | StandardScaler + MinMaxScaler | Single-stage StandardScaler | Single-stage StandardScaler | Single-stage StandardScaler | Single-stage StandardScaler | Single-stage StandardScaler | Single-stage StandardScaler |
| **Scheme A Leakage Acc** | 85%–89% (`GroupShuffleSplit`) | 92.8%–100.0% (`train_test_split`) | 93.8%–100.0% (`train_test_split`) | 79.6%–100.0% (`train_test_split`) | 78.8%–99.8% (`train_test_split`) | 77.7%–99.8% (`train_test_split`) | 93.6%–97.9% (`train_test_split`) |
| **Peak LOSO Model** | SVM RBF (87.41% pooled acc) | SVM RBF (57.22% pooled acc) | **Random Forest `depth=4` (74.43%)** | **Random Forest `depth=4` (71.14%)** | **Random Forest `depth=25` (73.11%)** | **LDA (71.18%)** | *No LOSO evaluated* |
| **Peak LOSO Balanced Acc** | XGBoost (70.78%) | LDA (51.28%) / SVM Lin (51.01%) | **Random Forest `depth=4` (74.90%)** | **Random Forest `depth=4` (71.22%)** | **Random Forest `depth=25` (73.23%)** | **LDA (72.46%)** | *No LOSO evaluated* |
| **Peak LOSO Macro F1** | SVM RBF (0.5584) | LDA (0.5130) | **Random Forest `depth=4` (0.7434)** | **Random Forest `depth=4` (0.7093)** | **Random Forest `depth=25` (0.7292)** | **LDA (0.7118)** | *No LOSO evaluated* |
| **Target Class F1 (Anxiety)** | 0.5584 (SVM RBF) | 0.7965 (SVM RBF, Class 2) | 0.7587 (RF `depth=4`, Class 1) | 0.7339 (RF `depth=4`, Class 1) | 0.7522 (RF `depth=25`, Class 1) | 0.7099 (LDA, Class 1) | 0.9926 (SVM Lin C=0.9, Stress) |

---

## 4. Remediation Summary & Status

All requested fixes have been implemented and verified:
- [x] **5th Try Paths Updated:** Updated from legacy `Models/stai_6s_standardized/` and `Data/Preprocessed...` to [`Models/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/5th%20try/) and [`Data/5th try/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/5th%20try/).
- [x] **5th Try Warning Added:** Warning callout in Section 2.1 documenting the questionnaire matrix indexing transposition bug that parsed only 5 items and caused 8 zero-anxiety participants.
- [x] **6th Try 2-Class Table 2.3 Corrected:** Replaced inaccurate manual transcription of baseline sums, condition sums, and score shifts for S3–S17 with verified values from raw WESAD CSV files.
- [x] **6th Try 2-Class Paths Updated:** Updated paths in Sections 6.3 and 10 to point to [`Models/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Models/6th%20try/2class/Full_sensor_modals/) and [`Data/6th try/2class/Full_sensor_modals/`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Data/6th%20try/2class/Full_sensor_modals/).
- [x] **Sliced & Minimum Specs Synchronized:** Per-subject TSST delta column in Section 2 updated across both notes.
- [x] **Folder-wide Revalidation:** Revalidation script executed; all paths, tables, warnings, and numbers match repository artifacts.
