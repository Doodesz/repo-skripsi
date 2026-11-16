---
tags:
  - skripsi/data
  - dataset/artefacts
  - storage
aliases:
  - Data Artefacts
  - Output Data
  - Dokumentasi Artefak Data
---
# Dokumentasi Artefak & Penyimpanan Data (`Data/`)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Scripts Pipeline Evolution (1st to 6th Try)]], [[Model Output Specs 5th Try]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]

Catatan inventarisasi berkas data hasil pra-pemrosesan, segmentasi jendela (*sliding window*), ekstraksi fitur multimodal, dan analisis kontribusi modalitas sensor di seluruh iterasi penelitian.

---

## 1. Struktur Lengkap Folder `Data/`

```
Data/
├── 5th try/                                    # Dataset fitur window 60s shift 6s (5th try)
│   ├── S2_stai_60s_shift6s.pkl                 # 15 berkas pickle per subjek
│   ├── S3_stai_60s_shift6s.pkl
│   └── ... (S4 s/d S17)
│
├── 6th try/                                    # Dataset Alkurdi outer-join alignment (6th try)
│   ├── 2class/                                 # Dataset klasifikasi biner terektifikasi
│   │   ├── S2_stai_2class_60s_shift6s.pkl      # Total 4.095 jendela (155 fitur X + 12 metadata)
│   │   └── ... (S3 s/d S17)
│   │
│   └── 3class/                                 # Dataset klasifikasi 3-kelas kontinu (±1/24)
│       ├── S2_stai_3class_60s_shift6s.pkl      # Total 4.095 jendela (155 fitur X + 12 metadata)
│       └── ... (S3 s/d S17)
│
├── Analysis/                                   # Artefak analisis kontribusi modalitas sensor
│   ├── sensor_modality_decision_contribution/  # Ablasi median per-window (7 modalitas lengkap)
│   │   ├── stai_baseline_relative_knn_*.csv / .png
│   │   ├── stai_baseline_relative_random_forest_*.csv / .png
│   │   ├── stai_baseline_relative_svm_*.csv / .png
│   │   └── stai_baseline_relative_xgboost_*.csv / .png
│   │
│   ├── sensor_modality_decision_contribution_acc_excluded/ # Ablasi tanpa sensor akselerometer
│   │   ├── knn_overall.csv / .png
│   │   ├── random_forest_overall.csv / .png
│   │   ├── svm_overall.csv / .png
│   │   └── xgboost_overall.csv / .png
│   │
│   └── sensor_modality_importance/             # Native & permutation feature importances
│       ├── stai_baseline_relative_knn_model_*.csv
│       ├── stai_baseline_relative_random_forest_model_*.csv
│       ├── stai_baseline_relative_svm_model_*.csv
│       └── stai_baseline_relative_xgboost_model_*.csv
│
└── Preprocessed n STAI-labeled/ (Historis 1st try)
    └── S{id}_stai.pkl                          # Dataset biner awal (window 60s, step 12s)
```

---

## 2. Karakteristik Berkas Dataset Utama

### A. Dataset 6th Try 2-Class (`Data/6th try/2class/`)
- **Dihasilkan oleh:** [`Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/2class/stai_2class_alkurdi_alignment.ipynb) & `Scripts/Utils/generate_2class_data_and_eval.py`.
- **Format:** Berkas serialized Python pickle (`pandas.DataFrame`).
- **Dimensi:** 167 kolom (155 fitur prediktif $X$, 12 kolom metadata dan label).
- **Windowing:** Durasi 60 detik, pergeseran (*stride*) 6 detik (overlap 90%).
- **Target:** `anxiety_label` biner ($0$ = No Anxiety Increase [43.59%], $1$ = Anxiety Increased [56.41%]).
- **Total Sampel Kohort:** 4.095 jendela (15 subjek × 4 kondisi: Fun, Medi 1, TSST, Medi 2). Lihat [[Model Output Specs 6th Try 2-Class]].

### B. Dataset 6th Try 3-Class (`Data/6th try/3class/`)
- **Dihasilkan oleh:** [`Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`](file:///e:/Rizqy/Kuliah/Skripsi/repo-skripsi/Scripts/6th%20try/3class/stai_3class_alkurdi_alignment.ipynb).
- **Format:** Berkas serialized Python pickle (`pandas.DataFrame`).
- **Dimensi:** 167 kolom (155 fitur prediktif $X$, 12 kolom metadata dan label).
- **Windowing:** Durasi 60 detik, pergeseran (*stride*) 6 detik (overlap 90%).
- **Target:** `anxiety_label` 3-kelas:
  - `0`: No Anxiety Increase ($\Delta < -1/24$, 27.11%)
  - `1`: Potential Anxiety Increase ($-1/24 \le \Delta \le 1/24$, 25.05%)
  - `2`: Anxiety Increased ($\Delta > 1/24$, 47.84%)
- **Total Sampel Kohort:** 4.095 jendela. Lihat [[Model Output Specs 6th Try 3-Class]].

### C. Dataset 5th Try (`Data/5th try/`)
- **Dihasilkan oleh:** `Scripts/5th try/stai_6s_reference_models_stdized.ipynb`.
- **Target:** Biner $\Delta \ge 0{,}2$ (terdapat isu zero-inflation historis pada 8 subjek).
- **Total Sampel Kohort:** 4.147 jendela (mencakup periode pembacaan `bRead`/`sRead`). Lihat [[Model Output Specs 5th Try]].

---

## 3. Data Keluaran Repositori Referensi (`_forked/data/`)

Untuk komparasi dengan penelitian terdahulu:
- `_forked/data/subject_feats/S{id}_feats.csv`: Berkas CSV fitur individual (segmentasi 60 detik tanpa overlap / tumbling window), target one-hot encoded: `0` (Amusement), `1` (Baseline), `2` (Stress).
- `_forked/data/oct5_feats4.csv` & `may14_feats4.csv`: Kumpulan fitur gabungan seluruh subjek dari eksperimen disertasi upstream.

---

## 4. Tautan Terkait
- Peta konsep vault: [[MOC - Skripsi Repository]]
- Evolusi skrip: [[Scripts Pipeline Evolution (1st to 6th Try)]]
- Model ML: [[Machine Learning Models & Evaluation]]
- Spesifikasi model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
- Katalog fitur lengkap: [[Katalog Fitur Fisiologis]]
