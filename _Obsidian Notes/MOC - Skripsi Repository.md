---
tags:
  - skripsi/index
  - skripsi/overview
  - map-of-content
aliases:
  - MOC Skripsi
  - Index Vault
  - Home
---
# Peta Konsep & Indeks Vault Skripsi (MOC)

Selamat datang di repositori riset skripsi: **Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi State Anxiety**.

Repositori ini mencakup implementasi end-to-end mulai dari data mentah multisensor fisiologis, pra-pemrosesan sinyal, pemotongan jendela geser (*sliding window*), ekstraksi fitur multi-domain, perumusan label kuesioner State-Trait Anxiety Inventory (STAI), hingga pelatihan dan evaluasi model Machine Learning.

---

## 🗺️ Struktur & Navigasi Modul

### 1. Fondasi & Metodologi Riset
- [[2026-10-29 Research Note]] — Temuan integrasi Samsung Health Sensor SDK untuk BVP/ECG, `PERMISSION_ERROR`, dan dugaan pembatasan ponsel non-Samsung.
- [[Rencana Riset & Pembangunan Sistem Deteksi Anxiety]] — Ringkasan komprehensif rencana riset, target pembangunan sistem, 5 skenario komparasi modalitas, dan alur eksperimen proposal.
- [[Proposal Skripsi Overview]] — Ringkasan usulan riset, rumusan masalah, latar belakang kecemasan (anxiety) vs stres, dan alur bab 3.
- [[Dataset WESAD]] — Dokumentasi dataset WESAD (15 subjek, sensor dada RespiBAN 700 Hz, sensor pergelangan Empatica E4, protokol lab, dan instrumen STAI).
- [[Development Findings 1]] — Catatan komparasi paradigma pelabelan, analisis kebocoran data (*data leakage*), justifikasi ambang batas heuristik skor STAI, dan pertimbangan klinis.

### 2. Pipeline Pemrosesan & Komparasi Metodologis
- [[Pipeline STAI User (v2_stai)]] — Konsep pipeline State Anxiety berbasis delta skor kuesioner mandiri STAI terhadap baseline individu ($\Delta \ge 0{,}2$), window 60s step 12s.
- [[Pipeline Forked (WESAD Protocol)]] — Konsep pipeline referensi berbasis kondisi stimulus protokol laboratorium (Baseline vs Stress vs Amusement), window 60s step 20s.
- [[Perbandingan Pipeline STAI vs Forked]] — Komparasi mendalam perbedaan ground truth, metode windowing, filtering batas waktu, dan format target antara kedua pipeline.

### 3. Implementasi Script (`Scripts/`)
- [[Script v2_stai]] — Implementasi teknis `Scripts/All-in-one/v2_stai.ipynb` untuk ekstraksi fitur biner STAI dan pemodelan klasifikasi State Anxiety.
- [[Script v1]] — Implementasi teknis `Scripts/All-in-one/v1.ipynb` untuk ekstraksi fitur 3-kelas protokol WESAD.

### 4. Rekayasa Fitur & Pemrosesan Sinyal
- [[Katalog Fitur Fisiologis]] — Katalog lengkap taksonomi 118 fitur fisiologis multisensor (statistik dasar, akselerometer, kardiovaskular HRV, EDA, EMG, respirasi, suhu).
- [[Pra-Pemrosesan Sinyal & Filtering]] — Mekanisme pembersihan derau (Butterworth LP untuk EDA, FIR LP untuk akselerometer, dekomposisi phasic/tonic SCR/SCL via NeuroKit2).
- [[Wrangling & Sinkronisasi Multimodal]] — Mekanisme sinkronisasi data dari sensor berfrekuensi heterogen ($4\text{ Hz}$ hingga $700\text{ Hz}$) berbasis stempel waktu absolut dan outer join.

### 5. Repositori Referensi Upstream (`_forked`)
- [[Forked Repository Overview]] — Ikhtisar repositori referensi upstream disertasi terdahulu (`anxietyFB`).
- [[Modul Library fb_code]] — Dokumentasi teknis modul Python pada `_forked/fb_code/` (`feature_extraction.py`, `utils.py`, `ecg.py`, `respiration.py`).

### 6. Artefak Data & Evaluasi Model (`Data/` & `Models/`)
- [[Data Artefacts Overview]] — Dokumentasi struktur penyimpanan data mentah, data bersih, dan dataset fitur teranotasi (`Data/Preprocessed n STAI-labeled/` dan `_forked/data/`).
- [[Machine Learning Models & Evaluation]] — Analisis performa inferensi model ML (Random Forest, XGBoost, SVM, KNN) pada klasifikasi 3-kelas protokol dan klasifikasi biner State Anxiety.

---

## 📊 Ringkasan Alur Data (Data Flow)

```
[Raw WESAD Data (WESAD/S{id})]
       │
       ├──> Sinyal Fisiologis (Dada: ECG, Resp, EDA, EMG, Temp, ACC | Pergelangan: BVP, EDA, Temp, ACC)
       └──> Respons Kuesioner (quest.csv: STAI per kondisi)
       │
       ▼
[Filtering & Signal Decomposition] (NeuroKit2, Scipy Butterworth, FIR)
       │
       ▼
[Windowing & Pelabelan]
       ├─ (A) Pipeline Forked: Window 60s (step 20s) -> Label Protokol (0=Amusement, 1=Baseline, 2=Stress)
       └─ (B) Pipeline User:   Window 60s (step 12s) -> Label STAI Delta >= 0.2 (0=No Anxiety, 1=Anxiety)
       │
       ▼
[Ekstraksi Fitur Multi-Sensor] (Statistik dasar, dinamis, spektral, HRV/ECG, Respiration) -> 118 Fitur
       │
       ▼
[Pemodelan & Evaluasi] (RF, XGBoost, SVM, KNN) -> Simpan artefak ke Models/
```
