---
tags:
  - skripsi/proposal
  - skripsi/teori
  - tinjauan-pustaka
aliases:
  - Proposal Skripsi
  - Overview Proposal
---
# Proposal Skripsi Overview

**Judul Dokumen:** *Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi State Anxiety*  
**Berkas Asli:** `PDF/Proposal Skripsi Rizqy J Rev0 - Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi Anxiety.pdf`

---

## 1. Latar Belakang & Masalah

- **State Anxiety vs. General Stress:**
  Sebagian besar literatur yang menggunakan dataset WESAD mendeteksi kondisi **stres eksperimental** yang diinduksi melalui protokol laboratorium (khususnya *Trier Social Stress Test* / TSST). Namun, stres fisiologis tidak serta-merta identik dengan kecemasan sesaat (*State Anxiety*) yang dirasakan secara psikologis oleh subjek.
- **Tujuan Penelitian:**
  Mengevaluasi kelayakan (*feasibility*) pemanfaatan sinyal fisiologis multimodal dari dataset WESAD (sensor dada dan pergelangan tangan) untuk mendeteksi kecemasan subjektif yang diukur menggunakan kuesioner **State-Trait Anxiety Inventory (STAI)**, serta membandingkan perilakunya terhadap klasifikasi stres protokol konvensional.

---

## 2. Rumusan Pertanyaan Riset

1. Bagaimana memetakan skor kuesioner mandiri STAI menjadi label diskret (*ground truth*) yang mencerminkan peningkatan kecemasan relatif terhadap kondisi dasar (*baseline*) masing-masing subjek?
2. Bagaimana efektivitas fitur fisiologis multimodal (ECG, Respiration, EDA, EMG, Accelerometer, BVP, Temperature) dalam membedakan kondisi cemas vs non-cemas?
3. Seberapa baik model machine learning (Random Forest, SVM, XGBoost, KNN) mampu menggeneralisasi klasifikasi State Anxiety pada data fisiologis multimodal?

---

## 3. Metodologi (Bab 3 Proposal)

### A. Alur Kerja (Workflow)
1. **Akuisisi Data:** Memuat 15 subjek dari dataset WESAD (sensor RespiBAN dada dan Empatica E4 pergelangan).
2. **Pra-Pemrosesan Sinyal:**
   - Filter lowpass Butterworth (EDA).
   - Filter FIR (Accelerometer).
   - Dekomposisi Phasic/Tonic (EDA SCR/SCL via NeuroKit2).
   - Ekstraksi puncak R ECG dan fitur pernapasan (BioSPPy).
3. **Segmentasi Jendela (*Windowing*):**
   - Jendela waktu 60 detik dengan pergeseran waktu (*stride*).
   - Memastikan jendela terinklusi penuh dalam kondisi bertarget.
4. **Pelabelan Target:**
   - Menggunakan ambang peningkatan rerata skor STAI kondisi terhadap rerata baseline:
     $$\Delta \text{STAI} = \text{Mean}_{\text{kondisi}} - \text{Mean}_{\text{Base}} \ge 0{,}2$$
5. **Ekstraksi Fitur Multi-Domain:**
   - Fitur statistik waktu (mean, std, min, max, dynamic range, slope, absolute integral).
   - Fitur frekuensi/spektral (peak frequency BVP dan ACC).
   - Fitur HRV (Heart Rate Variability: HR, mean_RR, rmssd, pnn50, dll.).
   - Fitur laju dan fase pernapasan (Inhalation/Exhalation mean, std, ratio).
6. **Pemodelan & Evaluasi:**
   - Algoritma: Random Forest, Support Vector Machine (SVM), Extreme Gradient Boosting (XGBoost), K-Nearest Neighbors (KNN).
   - Metrik evaluasi: Akurasi, Precision, Recall, F1-Score, Confusion Matrix.

---

## 4. Tautan Terkait
- Rencana riset & rekap komprehensif: [[Rencana Riset & Pembangunan Sistem Deteksi Anxiety]]
- Peta konsep: [[MOC - Skripsi Repository]]
- Pipeline STAI: [[Pipeline STAI User (v2_stai)]]
- Pipeline Protokol: [[Pipeline Forked (WESAD Protocol)]]
- Catatan Temuan & Analisis: [[Development Findings 1]]
- Hasil Evaluasi Model: [[Machine Learning Models & Evaluation]]
