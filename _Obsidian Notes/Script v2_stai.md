---
tags:
  - skripsi/script
  - machine-learning
  - pipeline/stai
aliases:
  - Script v2_stai
  - v2_stai.ipynb
---
# Script: `Scripts/All-in-one/v2_stai.ipynb`

Notebook all-in-one yang mengimplementasikan end-to-end pipeline deteksi **State Anxiety** berbasis label laporan mandiri kuesioner STAI (State-Trait Anxiety Inventory).

---

## 1. Alur & Logika Script

1. **Parser Kuesioner STAI:**
   - Membaca `WESAD/S{id}/S{id}_quest.csv`.
   - Mengelompokkan dan merata-ratakan skor butir respons STAI yang valid pada tiap kondisi per peserta.
2. **Perumusan Label Biner State Anxiety:**
   - Rerata STAI kondisi `Base` dijadikan titik referensi individu.
   - Kondisi diberi label **Anxiety (1)** jika:
     $$\text{Mean STAI}_{\text{kondisi}} - \text{Mean STAI}_{\text{Base}} \ge 0{,}2$$
     Selain itu, diberi label **No Anxiety (0)**.
3. **Filtering Sinyal & Sliding Window:**
   - Panjang jendela: 60 detik.
   - Pergeseran (*stride*): 12 detik.
   - Hanya jendela yang **100% berada di dalam batas waktu kondisi bertarget** dengan respons STAI valid yang dipertahankan. Jendela perbatasan dibuang.
4. **Ekstraksi Fitur:** Menghasilkan 118 fitur fisiologis multisensor + 6 kolom metadata pelacak (`subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `condition_stai_mean`, `baseline_stai_mean`).
5. **Penyimpanan:** Disimpan dalam format serialized pickle ke `Data/Preprocessed n STAI-labeled/S{id}_stai.pkl`.
6. **Pelatihan & Evaluasi Model Biner:**
   - Menggabungkan dataset 15 subjek (total 4.305 jendela: 3.809 kelas 0, 496 kelas 1).
   - Melatih Random Forest, XGBoost, KNN, dan SVM.

---

## 2. Hasil Output & Performa Model (Split Acak)

| Model | Test Accuracy | Precision (Anxiety) | Recall (Anxiety) | F1-Score (Anxiety) | Catatan Perilaku |
|:---|:---:|:---:|:---:|:---:|:---|
| **XGBoost** | **99.42%** | 0.97 | 0.98 | 0.97 | Performa separasi non-linear terbaik |
| **Random Forest** | **99.30%** | 0.97 | 0.97 | 0.97 | Sangat stabil terhadap outlier |
| **KNN** | **93.84%** | 0.78 | 0.65 | 0.71 | Performa moderat pada batas ketetanggaan |
| **SVM** | **88.50%** | 0.00 | 0.00 | 0.00 | Mengalami *majority class collapse* |

*Artefak bobot model disimpan di:* `Models/binary_random_forest_model.joblib`, `Models/binary_xgboost_model.joblib`, `Models/binary_svm_model.joblib`, `Models/binary_knn_model.joblib`.

---

## 3. Tautan Terkait
- Metodologi STAI: [[Pipeline STAI User (v2_stai)]]
- Script protokol: [[Script v1]]
- Katalog fitur: [[Katalog Fitur Fisiologis]]
- Komparasi pipeline: [[Perbandingan Pipeline STAI vs Forked]]
- Analisis kritis model: [[Machine Learning Models & Evaluation]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
