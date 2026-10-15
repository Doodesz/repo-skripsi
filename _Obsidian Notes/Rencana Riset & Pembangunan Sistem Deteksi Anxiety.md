---
title: "Rencana Riset & Pembangunan Sistem: Deteksi State Anxiety Berbasis Wearable Tunggal"
date: 2026-10-15
tags:
  - skripsi/rencana-riset
  - skripsi/proposal
  - dataset/wesad
  - machine-learning
  - smartwatch
  - state-anxiety
aliases:
  - Rencana Riset Skripsi
  - Recap Proposal & Build Plan
  - Roadmap Riset WESAD Anxiety
---

# Rencana Riset & Rekapitulasi Pengembangan Sistem
> **Dasar Dokumen Acuan:** [[Proposal Skripsi Rizqy J Rev0 - Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi Anxiety.pdf]]  
> **Judul Usulan:** *Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi Anxiety: Upaya Simplifikasi Modalitas Menggunakan Wearable Tunggal*  
> **Penyusun:** Rizqy Jauhary Atsaany (NIM: 235150300111038) — Teknik Komputer, FILKOM Universitas Brawijaya  
> **Peta Konsep Vault:** [[MOC - Skripsi Repository]]

---

## 🎯 1. Inti Masalah, Motivasi, & Research Gap

### Latar Belakang & Urgensi
1. **Pemanfaatan Sensor Smartwatch untuk Kesehatan Mental:** Smartwatch komersial modern telah dibekali beragam sensor biofisiologis (PPG/BVP, EDA, Suhu Kulit, Akselerometer, ECG). Namun, pemanfaatannya (*sensor utilization*) pada ranah kesehatan mental—khususnya kecemasan sesaat (*State Anxiety*)—masih sangat minim dibandingkan pelacak kebugaran biasa.
2. **Keterbatasan Setup Multimodal Laboratorium:** Riset acuan seperti [[Dataset WESAD]] menggunakan setup alat ganda (*dual-wearables*): perangkat medis di dada (RespiBAN, 700 Hz) dan pergelangan tangan (Empatica E4). Setup ini tidak praktis, invasif, dan tidak realistis untuk pemantauan sehari-hari (*in-the-wild*).
3. **Pembedaan State Anxiety vs. Stres Eksperimental:** Mayoritas riset terdahulu hanya mengklasifikasikan kondisi stimulus protokol (Baseline vs TSST/Stress vs Amusement). Padahal, respons stres laboratorium belum tentu merepresentasikan State Anxiety subjektif yang dirasakan partisipan.
4. **Peluang dari Studi Alkurdi et al. (2025):** Menjadi referensi utama dalam menganalisis kelayakan adaptasi (*transferability*) model dari dataset WESAD ke skenario dunia nyata serta menganalisis kontribusi signifikansi fitur untuk simplifikasi modalitas.

---

## 🔬 2. Rumusan Masalah & Tujuan Penelitian

| No | Rumusan Masalah | Target Tujuan Riset |
|:---:|:---|:---|
| **1** | Bagaimana kelayakan dan performa model ML yang dilatih dari dataset WESAD untuk mendeteksi kondisi anxiety tanpa bergantung pada data pelatihan primer? | Membuktikan kelayakan (*feasibility*) dan akurasi model ML pre-trained WESAD untuk klasifikasi State Anxiety. |
| **2** | Bagaimana degradasi performa deteksi saat model direduksi dari seluruh konfigurasi sensor smartwatch beririsan WESAD ke kombinasi sensor minimum dengan signifikansi fitur tertinggi? | Membandingkan performa model multi-sensor smartwatch vs model minimalis berfitur paling dominan untuk mengetahui batas efisiensi komputasi wearables. |
| **3** | Apakah upaya simplifikasi akuisisi data dari wearable ganda (dada & pergelangan) menjadi **wearable tunggal komersial (smartwatch)** mampu mempertahankan performa deteksi yang memadai pada skenario dunia nyata? | Mengevaluasi secara empiris ketahanan dan efektivitas inferensi model di lingkungan riil menggunakan smartwatch komersial dengan induksi stresor mental terstandar. |

---

## 🏗️ 3. Desain Komparasi 5 Skenario Penelitian

Penelitian ini membagi evaluasi ke dalam **5 skenario bertahap** untuk mengisolasi pengaruh reduksi sensor dan transisi ke lingkungan nyata:

```
[Skenario 1: Full WESAD (Dada + Pergelangan)] ──> Upper Bound Benchmark (Laboratorium)
                     │
                     ▼ (Reduksi: Hilangkan sensor dada)
[Skenario 2: Irisan Smartwatch pada Data WESAD] ──> Evaluasi Penurunan Performa akibat Reduksi Modalitas
                     │
                     ▼ (Reduksi: Hanya sensor dominan/paling signifikan)
[Skenario 3: Sensor Minimalis Dominan pada Data WESAD] ──> Batas Efisiensi Model Wearable Ringan
                     │
       ┌─────────────┴─────────────┐
       ▼ (Uji Dunia Nyata)         ▼ (Uji Dunia Nyata)
[Skenario 4: Testing Smartwatch]  [Skenario 5: Testing Minimalis]
```

### Matriks Komparasi Skenario (Tabel 3.1 Proposal)

| Skenario | Sumber Data Uji | Konfigurasi Sensor yang Digunakan | Lingkungan | Target & Tujuan Analisis |
|:---:|:---|:---|:---:|:---|
| **Skenario 1** | [[Dataset WESAD]] (Sekunder) | **Seluruh Sensor WESAD**<br>• Dada: ECG, EDA, EMG, Resp, Temp, ACC<br>• Pergelangan: BVP, EDA, Temp, ACC | Laboratorium | Tolok ukur performa tertinggi (*Upper Bound Benchmark*). |
| **Skenario 2** | [[Dataset WESAD]] (Sekunder) | **Irisan Sensor Smartwatch & WESAD**<br>• Pergelangan: BVP/PPG, EDA, Temp, ACC | Laboratorium | Mengukur dampak murni degradasi performa akibat hilangnya modalitas dada tanpa noise lingkungan luar. |
| **Skenario 3** | [[Dataset WESAD]] (Sekunder) | **Sensor Minimalis Signifikan**<br>• Kombinasi sensor minimum dengan signifikansi fitur tertinggi (berdasarkan literatur Alkurdi et al. 2025) | Laboratorium | Menemukan titik kompromi terbaik antara keringanan komputasi model dan akurasi deteksi. |
| **Skenario 4** | Partisipan Primer (Smartwatch) | **Seluruh Irisan Sensor Smartwatch**<br>• Akuisisi data via Smartwatch Komersial | Lingkungan Nyata Terkontrol | Menguji resistensi model terhadap noise artefak gerak & variasi fisiologis riil vs Skenario 1 & 2. |
| **Skenario 5** | Partisipan Primer (Smartwatch) | **Sensor Minimalis Signifikan**<br>• Akuisisi via Smartwatch Komersial | Lingkungan Nyata Terkontrol | Memvalidasi apakah model ultra-minimalis tetap layak dan reliabel diterapkan pada wearable konsumen. |

---

## ⚙️ 4. Metodologi Teknis & Rencana Pipeline

### A. Alur Pemrosesan Sinyal & Filtering
Mengacu pada implementasi [[Pra-Pemrosesan Sinyal & Filtering]] dan modul [[Modul Library fb_code]]:
1. **EDA (Aktivitas Elektrodermal):** Filter Low-pass Butterworth ($1\text{ Hz}$ cut-off) untuk membuang noise frekuensi tinggi; dekomposisi komponen tonik (SCL) dan fasik (SCR).
2. **Akselerometer (ACC):** Filter FIR Low-pass ($5\text{ Hz}$ cut-off) untuk memisahkan gravitasi tubuh dari getaran dinamis.
3. **Kardiovaskular (ECG & BVP / PPG):** Band-pass filter ($0{,}5 - 40\text{ Hz}$) untuk isolasi denyut jantung, dilanjutkan ekstraksi puncak R (ECG) atau peak sistolik (BVP).
4. **Respirasi & Suhu:** Penyaringan derau baseline wander dan normalisasi rentang dinamis.

### B. Segmentasi Jendela Waktu (*Windowing*)
- **Jendela Utama:** Sliding window berdurasi **60 detik** dengan pergeseran (*stride*) bertahap (standar WESAD Schmidt et al. 2018 & Alkurdi et al. 2025).
- **Akselerometer Kontekstual:** Jendela pendek (5 detik) untuk menangkap perubahan postur/aktivitas gerak sesaat.

### C. Ekstraksi Fitur Multi-Domain
Mengacu pada taksonomi lengkap di [[Katalog Fitur Fisiologis]]:
- **Domain Waktu:** Mean, simpangan baku (STD), min, max, dynamic range, integral absolut, kemiringan (*slope*).
- **Kardiovaskular & HRV:** HR mean/STD, RMSSD, pNN50, TINN, rasio spektral LF/HF, Normalized LF/HF, Peak BVP.
- **Sistem Saraf Simpatis (EDA):** Jumlah SCR peaks, startle magnitude, area under curve (AUC) SCR, korelasi SCL-waktu.
- **Respirasi & EMG:** Durasi inspirasi/ekspirasi ($I/E$ ratio), respiratory rate, median frequency EMG, total power band.

### D. Ground Truth & Formulasi Label
- **Data Sekunder WESAD:** Respons kuesioner `quest.csv` State-Trait Anxiety Inventory (STAI Form Y-1). Kondisi non-baseline diberi label **Anxiety (1)** jika mengalami peningkatan signifikan relatif terhadap kondisi santai awal:
  $$\Delta \text{STAI} = \text{Mean}_{\text{kondisi}} - \text{Mean}_{\text{Base}} \ge 0{,}2$$
  *(Detail aturan pelabelan dibahas pada [[Pipeline STAI User (v2_stai)]] dan pertimbangan analitisnya di [[Development Findings]]).*
- **Data Primer Pengujian Riil:** Partisipan mengisi **STAI Form Y-1** pada dua titik waktu: tepat setelah Fase Tenang (Baseline 5–10 menit) dan tepat setelah induksi mental.

---

## 🧪 5. Skenario Pengujian Dunia Nyata (Data Primer)

1. **Kriteria Partisipan:** Mahasiswa usia 18–24 tahun, kondisi fisik sehat, tidak memiliki riwayat kardiovaskular akut, dan tidak buta warna (syarat tes Stroop).
2. **Protokol Induksi State Anxiety:**
   - **Fase 1: Kondisi Santai / Baseline (5–10 menit):** Duduk rileks di ruangan tenang tanpa distraksi audio/visual untuk merekam kalibrasi fisiologis awal. Pengisian STAI Baseline.
   - **Fase 2: Stresor Mental via Stroop Color-Word Test (SCWT) (5–10 menit):** Tugas kognitif terkomputerisasi dengan batas waktu cepat dan umpan balik kesalahan visual/audio untuk memicu stres mental akut (*mental overload*). Pengisian STAI Post-Induksi.
3. **Perangkat Wearable:** Smartwatch komersial pergelangan tangan (misalnya Samsung Galaxy Watch Series) yang mendukung akuisisi biosinyal mentah/terekstraksi minimal 2 sensor beririsan WESAD.

---

## 🤖 6. Pengembangan Model, Validasi, & Deployment

### Algoritma yang Diuji
- **Random Forest (RF):** Unggul dalam mengatasi sebaran non-linear dan stabil terhadap outlier fisiologis.
- **Extreme Gradient Boosting (XGBoost / XGB):** Menghasilkan akurasi tinggi melalui boosting pohon keputusan bertahap.
- **Support Vector Machine (SVM):** Menggunakan trik kernel RBF untuk pemisahan hyperplane berdimensi tinggi.
- **k-Nearest Neighbor (k-NN):** Tolok ukur non-parametrik berbasis kedekatan jarak fitur.

### Validasi Model & Mitigasi Leakage
- **Validasi Silang:** Skema 5-Fold Cross-Validation / GroupKFold berdasarkan ID Subjek untuk memastikan jendela dari individu yang sama tidak bocor ke data latih dan uji secara bersamaan (*subject-wise separation*).
- **Metrik Evaluasi:** Akurasi, Precision, Recall, Confusion Matrix, dan terutama **F1-Score** (esensial karena rasio kelas cemas vs tidak cemas mengalami *class imbalance* alami).

### Arsitektur Deployment Smartwatch
1. **Konversi Model:** Model terbaik diekspor ke format **TensorFlow Lite (`.tflite`)** agar ringan dan hemat memori (sejalan dengan artefak pada [[Machine Learning Models & Evaluation]]).
2. **Strategi Eksekusi:**
   - *Opsi Utama (Koleksi Mandiri):* Aplikasi smartwatch fokus merekam data sensor mentah secara stabil tanpa membebani baterai/CPU. Data diekspor ke komputer untuk pipeline deteksi.
   - *Opsi Edge/Streaming:* Pengiriman sinyal secara kontinu ke edge/komputer lokal via Bluetooth/WiFi untuk inferensi *near real-time*.

---

## 📋 7. Matriks Status Implementasi di Workspace

| Modul / Rencana | Status di Repositori | Berkas / Catatan Terkait |
|:---|:---:|:---|
| **Ekstraksi Fitur Protokol (3-Kelas)** | Selesai | [[Script v1]] (`Scripts/All-in-one/v1.ipynb`) |
| **Ekstraksi Fitur STAI Anxiety (Biner)** | Selesai | [[Pipeline STAI User (v2_stai)]], [[Script v2_stai]] (`Scripts/All-in-one/v2_stai.ipynb`) |
| **Pipeline Referensi Upstream WESAD** | Tersedia | [[Forked Repository Overview]], [[Modul Library fb_code]] |
| **Pelatihan Model ML Baseline (Skenario 1 & 2)** | Selesai | [[Machine Learning Models & Evaluation]] (`Models/*.joblib`, `wesad_stress_model.tflite`) |
| **Eksperimen Reduksi Sensor Signifikan (Skenario 3)** | Perlu Dijalankan | Analisis fitur dominan Alkurdi et al. (2025) pada subset fitur WESAD |
| **Protokol SCWT & Akuisisi Smartwatch (Skenario 4 & 5)** | Tahap Perancangan | Setup SCWT terkomputerisasi, kuesioner STAI Form Y-1, dan logging data smartwatch |

---

## 🔗 8. Tautan Catatan Terkait di Vault
- 🏠 **Indeks Utama Vault:** [[MOC - Skripsi Repository]]
- 📄 **Proposal Dokumen Sumber:** [[Proposal Skripsi Overview]] & [[Proposal Skripsi Rizqy J Rev0 - Analisis Kelayakan Model Machine Learning Menggunakan Dataset WESAD untuk Deteksi Anxiety.pdf]]
- 📊 **Basis Data & Rekayasa Fitur:** [[Dataset WESAD]], [[Katalog Fitur Fisiologis]], [[Pra-Pemrosesan Sinyal & Filtering]], [[Wrangling & Sinkronisasi Multimodal]]
- 💡 **Metodologi & Validasi:** [[Pipeline STAI User (v2_stai)]], [[Perbandingan Pipeline STAI vs Forked]], [[Development Findings]]
- 🤖 **Evaluasi Model & Artefak:** [[Machine Learning Models & Evaluation]], [[Data Artefacts Overview]]
