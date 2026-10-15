---
tags:
  - skripsi/script
  - machine-learning
  - pipeline/forked
aliases:
  - Script v1
  - v1.ipynb
---
# Script: `Scripts/All-in-one/v1.ipynb`

Notebook all-in-one yang mereproduksi ekstraksi fitur referensi upstream (`_forked/fb_code/feature_extraction.py`) dengan modifikasi berupa penggunaan jendela geser (*sliding window*) untuk klasifikasi 3-kelas protokol WESAD.

---

## 1. Alur & Logika Script

1. **Pemuatan Data Mentah:** Membaca berkas biner `WESAD/S{id}/S{id}.pkl` untuk 15 subjek.
2. **Pra-Pemrosesan Sinyal:**
   - Filter lowpass Butterworth untuk EDA (cutoff 1.0 Hz, orde 6).
   - Filter FIR lowpass untuk akselerometer dada dan pergelangan.
   - Dekomposisi phasic (SCR) dan tonic (SCL) via NeuroKit2 (`nk.eda_phasic`).
   - Sinkronisasi sinyal multisensor berfrekuensi beda melalui datetime index outer-join pada basis frekuensi label (700 Hz).
3. **Segmentasi Jendela Geser (*Sliding Window*):**
   - Jendela: 60 detik ($42.000$ sampel).
   - Pergeseran (*stride*): 20 detik (1/3 panjang jendela).
4. **Target Pelabelan 3-Kelas Protokol WESAD:**
   - Mempertahankan label kondisi stimulus lab WESAD:
     - `1`: Baseline (istirahat netral)
     - `2`: Stress (induksi TSST)
     - `3` / `0`: Amusement (hiburan video lucu)
5. **Ekstraksi Fitur:** Menghasilkan 118 fitur fisiologis multimodal per jendela (statistik dasar, integral akselerometer, HRV via HeartPy, laju napas, dan kemiringan regresi linier).
6. **Ekspor & Pelatihan Model:**
   - Menyimpan target dalam format one-hot encoding (`0`, `1`, `2`).
   - Melatih 4 algoritma ML: Random Forest, XGBoost, SVM, dan KNN.

---

## 2. Hasil Output & Performa Model (Split Acak)

- **Random Forest:** Akurasi **100.00%**
- **XGBoost:** Akurasi **100.00%**
- **KNN:** Akurasi **85.85%**
- **SVM:** Akurasi **72.03%**

*Artefak bobot model disimpan di:* `Models/random_forest_model.joblib`, `Models/xgboost_model.joblib`, `Models/svm_model.joblib`, `Models/knn_model.joblib`.

---

## 3. Tautan Terkait
- Script STAI: [[Script v2_stai]]
- Metodologi protokol: [[Pipeline Forked (WESAD Protocol)]]
- Katalog fitur: [[Katalog Fitur Fisiologis]]
- Evaluasi & analisis model: [[Machine Learning Models & Evaluation]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
