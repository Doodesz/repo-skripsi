---
tags:
  - skripsi/models
  - machine-learning
  - evaluation
  - random-forest
  - xgboost
  - svm
  - knn
aliases:
  - ML Models and Evaluation
  - Evaluasi Model
---
# Model Machine Learning & Hasil Evaluasi (`Models/`)

Catatan komprehensif mengenai arsitektur model, artefak bobot tersimpan, metrik performa, serta analisis perbandingan antara klasifikasi multi-kelas protokol WESAD dan klasifikasi biner State Anxiety.

---

## 1. Inventarisasi Model Tersimpan (`Models/`)

Repositori ini menyimpan dua kelompok artefak model terlatih menggunakan format `joblib` dan `tflite`:

```
Models/
├── binary_random_forest_model.joblib   # Model Random Forest untuk biner STAI Anxiety
├── binary_xgboost_model.joblib         # Model XGBoost untuk biner STAI Anxiety
├── binary_knn_model.joblib             # Model KNN untuk biner STAI Anxiety
├── binary_svm_model.joblib             # Model SVM untuk biner STAI Anxiety
│
├── random_forest_model.joblib          # Model Random Forest untuk 3-kelas Protokol WESAD
├── xgboost_model.joblib                # Model XGBoost untuk 3-kelas Protokol WESAD
├── knn_model.joblib                    # Model KNN untuk 3-kelas Protokol WESAD
├── svm_model.joblib                    # Model SVM untuk 3-kelas Protokol WESAD
└── wesad_stress_model.tflite           # Model TensorFlow Lite untuk deployment edge/mobile
```

---

## 2. Perbandingan Performa Evaluasi

### A. Klasifikasi Biner State Anxiety (`v2_stai.ipynb`)
- **Target:** Biner (0 = No Anxiety, 1 = Anxiety).
- **Dataset:** 4.305 jendela fisiologis multimodal dari 15 subjek WESAD.
- **Rasio Kelas:** 3.809 No Anxiety vs 496 Anxiety (Imbalance rasio ~8:1).
- **Hasil Pengujian (Split Acak):**

| Model | Test Accuracy | Precision (Anxiety) | Recall (Anxiety) | F1-Score (Anxiety) | Analisis Perilaku |
|:---|:---:|:---:|:---:|:---:|:---|
| **XGBoost** | **99.42%** | 0.97 | 0.98 | 0.97 | Kemampuan boosting pohon keputusan sangat baik dalam memisahkan sebaran fitur non-linear. |
| **Random Forest** | **99.30%** | 0.97 | 0.97 | 0.97 | Sangat stabil dan tahan terhadap outlier sinyal fisiologis. |
| **KNN** | **93.84%** | 0.78 | 0.65 | 0.71 | Performa cukup baik, namun mengalami degradasi pada batas kepadatan antarkelas. |
| **SVM** | **88.50%** | 0.00 | 0.00 | 0.00 | **Majority Class Collapse**: Model memprediksi seluruh sampel uji sebagai kelas mayoritas (0). Membutuhkan *class-weight balancing* dan standardisasi fitur per subjek. |

---

### B. Klasifikasi 3-Kelas Protokol WESAD (`v1.ipynb`)
- **Target:** Multi-kelas (1 = Baseline, 2 = Stress, 0/3 = Amusement).
- **Hasil Pengujian:**
  - **Random Forest:** Akurasi **100.00%**
  - **XGBoost:** Akurasi **100.00%**
  - **KNN:** Akurasi **85.85%**
  - **SVM:** Akurasi **72.03%**

---

## 3. Analisis Kritis & Temuan Ilmiah untuk Skripsi

> [!important] 1. Perbedaan Mendasar Makna Target
> - Model protokol WESAD (`v1.ipynb`) mendeteksi **kondisi stimulus eksternal laboratorium**. Sangat wajar akurasi mencapai 100% karena respons otonom saat TSST (stres bicara di depan juri) sangat tajam berbeda dibanding saat menonton film komedi.
> - Model STAI (`v2_stai.ipynb`) mendeteksi **penilaian subjektif individu**. Keikutsertaan dalam TSST tidak menjamin subjek merasa cemas, dan kondisi lain pun dapat memicu kecemasan bila skor STAI meningkat $\ge 0{,}2$.

> [!warning] 2. Potensi Kebocoran Data (*Data Leakage*) pada Random Split
> Segmentasi jendela geser 60 detik dengan pergeseran 12 detik menghasilkan jendela yang saling bertumpang tindih (*overlap* 80%). Apabila pembagian data latih-uji dilakukan secara acak pada tingkat jendela (*window-level split*), maka jendela yang berdampingan dari subjek yang sama dapat masuk ke data latih dan uji sekaligus.
> 
> **Rekomendasi untuk Laporan Skripsi:**
> Untuk membuktikan kelayakan model secara realistis pada subjek baru (*unseen subjects*), pengujian wajib dilengkapi skema validasi **Leave-One-Subject-Out (LOSO)** atau **GroupKFold berdasarkan ID Subjek**.

---

## 4. Tautan Terkait
- Peta konsep: [[MOC - Skripsi Repository]]
- Script STAI: [[Script v2_stai]]
- Script Protokol: [[Script v1]]
- Catatan komparasi metodologis: [[Development Findings]]
- Proposal skripsi: [[Proposal Skripsi Overview]]
