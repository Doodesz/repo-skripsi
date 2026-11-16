---
tags:
  - skripsi/models
  - machine-learning
  - evaluation
  - random-forest
  - xgboost
  - svm
  - knn
  - lda
aliases:
  - ML Models and Evaluation
  - Evaluasi Model
  - Machine Learning Models & Evaluation
---
# Model Machine Learning & Hasil Evaluasi (`Models/`)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Scripts Pipeline Evolution (1st to 6th Try)]], [[Model Output Specs 5th Try]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]], [[2026-11-12 Research Note]]

Catatan komprehensif mengenai arsitektur model, artefak bobot tersimpan, metrik evaluasi lintas generasi (1st try hingga 6th try), komparasi kebocoran data (*data leakage* vs *LOSO generalization*), dan temuan empiris untuk laporan skripsi.

---

## 1. Inventarisasi Artefak Model (`Models/`)

```
Models/
├── 5th try/                                    # Model acuan standardisasi 5th try
│   ├── stai_baseline_relative_random_forest_model.joblib
│   ├── stai_baseline_relative_xgboost_model.joblib
│   ├── stai_baseline_relative_svm_model.joblib
│   └── stai_baseline_relative_knn_model.joblib
│
├── 6th try/                                    # Model final aligned Alkurdi et al. (2025)
│   ├── 2class/                                 # Model klasifikasi biner terektifikasi (155 fitur)
│   │   ├── stai_2class_schemeA_lda_model.joblib
│   │   ├── stai_2class_schemeA_random_forest_model.joblib
│   │   ├── stai_2class_schemeA_svm_linear_model.joblib
│   │   ├── stai_2class_schemeA_svm_model.joblib
│   │   ├── stai_2class_schemeA_xgboost_model.joblib
│   │   ├── stai_2class_schemeA_knn_model.joblib
│   │   ├── evaluation_summary.json             # Metrik lengkap Scheme A & B
│   │   ├── per_subject_loso_results.json       # Rincian per subjek LOSO
│   │   ├── Sliced_sensor_modals/               # Sliced (ACC, EDA, TEMP - 62 fitur)
│   │   │   ├── stai_2class_schemeA_sliced_*.joblib (6 model)
│   │   │   ├── evaluation_summary.json & per_subject_loso_results.json
│   │   └── Minimum_sensor_modals/              # Pasangan Sensor Minimal
│   │       ├── stai_2class_schemeA_minimum_*.joblib (EDA+ACC, 56 fitur)
│   │       ├── eda_temp/                       # EDA+TEMP (20 fitur)
│   │       │   └── stai_2class_schemeA_minimum_eda_temp_*.joblib (6 model)
│   │       └── acc_temp/                       # ACC+TEMP (48 fitur)
│   │           └── stai_2class_schemeA_minimum_acc_temp_*.joblib (6 model)
│   │
│   └── 3class/                                 # Model klasifikasi 3-kelas kontinu (±1/24, 155 fitur)
│       ├── stai_3class_schemeA_lda_model.joblib
│       ├── stai_3class_schemeA_random_forest_model.joblib
│       ├── stai_3class_schemeA_svm_linear_model.joblib
│       ├── stai_3class_schemeA_svm_model.joblib
│       ├── stai_3class_schemeA_xgboost_model.joblib
│       ├── stai_3class_schemeA_knn_model.joblib
│       ├── evaluation_summary.json
│       └── per_subject_loso_results.json
│
├── binary_random_forest_model.joblib           # Model awal 1st try biner STAI
├── binary_xgboost_model.joblib
├── binary_svm_model.joblib
├── binary_knn_model.joblib
├── random_forest_model.joblib                  # Model awal 1st try 3-kelas protokol WESAD
├── xgboost_model.joblib
├── knn_model.joblib
├── svm_model.joblib
└── wesad_stress_model.tflite                   # Model TensorFlow Lite untuk deployment smartwatch
```

---

## 2. Perbandingan Metrik Evaluasi Lintas Generasi

### A. Generasi Pertama (`1st try`): Random Split Window-Level
- **Klasifikasi 3-Kelas Protokol WESAD (`v1.ipynb`):**
  - Random Forest: **100.00%** | XGBoost: **100.00%** | KNN: **85.85%** | SVM: **72.03%**
- **Klasifikasi Biner STAI ($\Delta \ge 0{,}2$, `v2_stai.ipynb`):**
  - XGBoost: **99.42%** | Random Forest: **99.30%** | KNN: **93.84%** | SVM: **88.50%** (F1 cemas = 0.00, *majority class collapse*)
- *Keterbatasan Ilmiah:* Pembagian acak pada tingkat jendela (*window-level random split*) menyebabkan jendela yang saling bertumpukan dari subjek yang sama hadir pada data latih dan uji, sehingga model menghafal identitas fisiologis subjek (*severe data leakage*).

---

### B. Generasi Final (`6th try`): Alkurdi Benchmark Alignment

#### 1. Klasifikasi Biner Terektifikasi (`6th try/2class/`)
Formulasi target: Class 0 ($\Delta \le 0$, 43.59%) vs Class 1 ($\Delta > 0$, 56.41%). Total 4.095 jendela, 155 fitur prediktif $X$.

| Arsitektur Model | Scheme A (Random Split 80/20) Acc | Scheme B (15-Fold LOSO) Pooled Acc | Scheme B Balanced Acc | Scheme B Macro F1 | Class 1 F1 (Kecemasan) | Macro-Subject Balanced Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (`max_depth=4`, Alkurdi)** | 96.21% | **74.43%** | **74.90%** | **0.7434** | **0.7587** | **78.45%** |
| **XGBoost (`n_est=100`)** | 99.76% | **74.14%** | **74.24%** | **0.7394** | **0.7621** | **77.29%** |
| **Random Forest (`max_depth=25`)** | **100.00%** | **73.72%** | **74.11%** | **0.7361** | **0.7532** | **77.52%** |
| **SVM (RBF Kernel)** | **100.00%** | **72.04%** | **71.59%** | **0.7158** | **0.7519** | **72.49%** |
| **KNN ($k=5$)** | 99.39% | **67.64%** | **68.53%** | **0.6763** | **0.6823** | **69.83%** |
| **SVM Linear ($C=0.9$)** | 96.58% | **64.98%** | **64.70%** | **0.6459** | **0.6830** | **66.42%** |
| **SVM Linear ($C=1.0$)** | 96.58% | **64.96%** | **64.64%** | **0.6455** | **0.6836** | **66.39%** |
| **LDA** | 93.77% | **62.66%** | **62.70%** | **0.6244** | **0.6535** | **63.45%** |

*Rincian lengkap dan matriks konfusi: [[Model Output Specs 6th Try 2-Class]].*

---

#### 2. Klasifikasi 3-Kelas Kontinu (`6th try/3class/`)
Formulasi target: Class 0 ($\Delta < -1/24$, 27.11%), Class 1 ($-1/24 \le \Delta \le 1/24$, 25.05%), Class 2 ($\Delta > 1/24$, 47.84%).

| Arsitektur Model | Scheme A (Random Split 80/20) Acc | Scheme B (15-Fold LOSO) Pooled Acc | Scheme B Balanced Acc | Scheme B Macro F1 | Class 2 F1 (Kecemasan) | Macro-Subject Balanced Acc |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **SVM (RBF Kernel)** | 99.51% | **57.22%** | **50.27%** | **0.4976** | **0.7965** | **57.72%** |
| **SVM Linear ($C=1.0$)** | 99.27% | **56.39%** | **51.01%** | **0.5024** | **0.7841** | **56.03%** |
| **SVM Linear ($C=0.9$)** | 99.27% | **56.34%** | **50.98%** | **0.5015** | **0.7827** | **56.08%** |
| **XGBoost (`n_est=100`)** | **100.00%** | **55.58%** | **49.31%** | **0.4942** | **0.7951** | **57.78%** |
| **LDA** | 94.51% | **55.56%** | **51.28%** | **0.5130** | **0.7704** | **56.92%** |
| **Random Forest (`max_depth=25`)** | **100.00%** | **53.21%** | **45.78%** | **0.4538** | **0.7935** | **56.44%** |
| **Random Forest (`max_depth=4`, Alkurdi)** | 92.80% | **52.94%** | **45.43%** | **0.4510** | **0.7922** | **55.61%** |
| **KNN ($k=5$)** | 99.39% | **51.06%** | **45.85%** | **0.4555** | **0.7462** | **52.32%** |

*Rincian lengkap dan matriks konfusi: [[Model Output Specs 6th Try 3-Class]].*

---

## 3. Analisis Kritis & Temuan Ilmiah untuk Skripsi

> [!important] 1. Ilusi Akurasi Sempurna vs. Generalisasi Nyata (*Data Leakage*)
> Menggunakan jendela geser 60 detik dengan pergeseran 6 detik menghasilkan 90% tumpang-tindih (*temporal overlap*). Pada skema naive random split (Scheme A), akurasi model mencapai **93%–100%**. Namun, pengujian dengan **Leave-One-Subject-Out (LOSO)** membuktikan bahwa performa realistis terhadap partisipan baru adalah **74.4% pada klasifikasi biner** dan **57.2% pada klasifikasi 3-kelas** (jauh di atas baseline tebakan acak 50% dan 33.3%). Ini membuktikan adanya sinyal fisiologis kecemasan yang dapat ditransfer antar-individu.

> [!tip] 2. Regularisasi Pohon Dangkal (*Shallow Trees*) Melawan Variansi Subjek
> Pada klasifikasi biner, Random Forest dangkal (`max_depth=4`) mengungguli model pohon dalam (`max_depth=25`) dalam validasi LOSO (74.43% vs 73.72%). Pembatasan kedalaman cabang berfungsi sebagai regularisasi efektif yang mencegah model menghafal pola fisiologis spesifik subjek tertentu, menghasilkan generalisasi yang lebih tangguh terhadap partisipan baru.

> [!note] 3. Ketahanan Pengenalan Puncak Kecemasan (Class 2 / Class 1)
> Baik pada skema 2-kelas maupun 3-kelas, kondisi kecemasan yang diinduksi oleh TSST berhasil dikenali secara konsisten pada level subjek unseen dengan nilai **F1-Score mencapai 0.76–0.80**. Tantangan utama terletak pada pembedaan zona batas netral (Class 1 pada 3-kelas) akibat fluktuasi respirasi dan aktivitas otonom istirahat alami.

---

## 4. Tautan Terkait
- Peta konsep utama: [[MOC - Skripsi Repository]]
- Evolusi skrip: [[Scripts Pipeline Evolution (1st to 6th Try)]]
- Riset 6th try: [[2026-11-12 Research Note]]
- Riset modality weights: [[2026-11-05 Research Note]]
- Spesifikasi teknis: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
