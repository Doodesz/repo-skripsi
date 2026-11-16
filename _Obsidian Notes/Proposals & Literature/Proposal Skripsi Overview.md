---
tags:
  - skripsi/proposal
  - skripsi/teori
  - tinjauan-pustaka
aliases:
  - Proposal Skripsi
  - Overview Proposal
  - Proposal Skripsi Overview
---
# Proposal Skripsi Overview

> **Judul Dokumen:** *Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi State Anxiety*  
> **Berkas Asli:** `PDF/Proposal Skripsi Rizqy J Rev0 - Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi Anxiety.pdf`  
> **Penyusun:** Rizqy Jauhary Atsaany (NIM: 235150300111038) — Teknik Komputer FILKOM UB  
> **Indeks Vault:** [[MOC - Skripsi Repository]]

---

## 1. Latar Belakang & Masalah

- **State Anxiety vs. General Stress:**
  Sebagian besar literatur yang menggunakan dataset WESAD mendeteksi kondisi **stres eksperimental** yang diinduksi melalui protokol laboratorium (khususnya *Trier Social Stress Test* / TSST). Namun, stres fisiologis stimulus tidak serta-merta identik dengan kecemasan sesaat (*State Anxiety*) yang dirasakan secara psikologis oleh subjek.
- **Tujuan Penelitian:**
  Mengevaluasi kelayakan (*feasibility*) pemanfaatan sinyal fisiologis multimodal dari dataset WESAD (sensor dada dan pergelangan tangan) untuk mendeteksi kecemasan subjektif yang diukur menggunakan kuesioner **State-Trait Anxiety Inventory (STAI)**, serta membandingkan perilakunya terhadap klasifikasi stres protokol konvensional.

---

## 2. Rumusan Pertanyaan Riset

1. Bagaimana memetakan skor kuesioner mandiri STAI menjadi label diskret (*ground truth*) yang mencerminkan peningkatan kecemasan relatif terhadap kondisi dasar (*baseline*) masing-masing subjek?
2. Bagaimana efektivitas fitur fisiologis multimodal (ECG, Respiration, EDA, EMG, Accelerometer, BVP, Temperature) dalam membedakan kondisi cemas vs non-cemas?
3. Seberapa baik model machine learning (Random Forest, SVM, XGBoost, KNN, LDA) mampu menggeneralisasi klasifikasi State Anxiety pada data fisiologis multimodal terhadap partisipan baru (*subject-independent generalization*)?

---

## 3. Metodologi (Bab 3 Proposal) & Realisasi di Repositori

### A. Alur Kerja (Workflow)
1. **Akuisisi Data:** Memuat 15 subjek dari dataset WESAD (sensor RespiBAN dada 700 Hz dan Empatica E4 pergelangan 4–64 Hz). Dokumentasi: [[Dataset WESAD]].
2. **Pra-Pemrosesan Sinyal:**
   - Filter lowpass Butterworth (EDA 1.0 Hz, orde 6).
   - Filter FIR (Accelerometer).
   - Dekomposisi Phasic/Tonic (EDA SCR/SCL via NeuroKit2).
   - Ekstraksi puncak R ECG dan fitur pernapasan (HeartPy & BioSPPy). Dokumentasi: [[Pra-Pemrosesan Sinyal & Filtering]].
3. **Penyelarasan Jam & Segmentasi Jendela (*Windowing*):**
   - Penyelarasan outer-join datetime nanosekon Alkurdi ([[Wrangling & Sinkronisasi Multimodal]]).
   - Jendela waktu 60 detik dengan pergeseran waktu (*stride*) 6 detik (overlap 90%).
   - Memastikan jendela terinklusi penuh dalam kondisi bertarget dengan respons valid.
4. **Pelabelan Target Subjektif:**
   - Berevolusi dari heuristik $\Delta \ge 0{,}2$ menjadi formulasi biner terektifikasi $\Delta > 0$ ([[Model Output Specs 6th Try 2-Class]]) dan 3-kelas kontinu $\pm \frac{1}{24}$ ([[Model Output Specs 6th Try 3-Class]]).
5. **Ekstraksi Fitur Multi-Domain:**
   - Menghasilkan 155 fitur prediktif: statistik dasar, integral kinetik, frekuensi puncak FFT, metrik HRV HeartPy, rasio inhalasi/ekshalasi napas, dan 36 fitur sub-jendela akselerometer ([[Katalog Fitur Fisiologis]]).
6. **Pemodelan & Evaluasi:**
   - Algoritma: Random Forest, Support Vector Machine (SVM RBF & Linear), Extreme Gradient Boosting (XGBoost), K-Nearest Neighbors (KNN), Linear Discriminant Analysis (LDA).
   - Validasi: Replikasi Naive Random Split 80/20 (Scheme A) vs 15-Fold Leave-One-Subject-Out CV (Scheme B). Dokumentasi: [[Machine Learning Models & Evaluation]].

---

## 4. Tautan Terkait
- Rencana riset komprehensif: [[Rencana Riset & Pembangunan Sistem Deteksi Anxiety]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
- Pipeline STAI: [[Pipeline STAI User (v2_stai)]]
- Pipeline Protokol: [[Pipeline Forked (WESAD Protocol)]]
- Catatan Temuan Metodologis: [[2026-10-06 Research Note]]
- Spesifikasi Model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
- Evaluasi Model: [[Machine Learning Models & Evaluation]]
