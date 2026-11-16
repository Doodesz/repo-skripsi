---
tags:
  - skripsi/forked
  - skripsi/referensi
  - skripsi/codebase
aliases:
  - Forked Overview
  - anxietyFB
  - Forked Repository Overview
---
# Repositori Referensi Upstream (`_forked`)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Modul Library fb_code]], [[Pipeline Forked (WESAD Protocol)]], [[Perbandingan Pipeline STAI vs Forked]], [[Scripts Pipeline Evolution (1st to 6th Try)]], [[2026-10-06 Research Note]]

Folder `_forked` merupakan repositori referensi dari proyek penelitian/disertasi sebelumnya yang berjudul **`anxietyFB`** (Abdul Alkurdi et al. — penelitian terkait prediksi kecemasan menggunakan sensor Empatica E4, Hexoskin, dan eksplorasi dataset WESAD/RADWear).

---

## 1. Peran dalam Penelitian Skripsi Ini

Folder ini berfungsi sebagai **basis pustaka sinyal dan tolok ukur komparasi (*baseline benchmark engine*)**:
1. Menjadi sumber algoritma pemrosesan sinyal fisiologis tervalidasi yang diadaptasi ke dalam folder `Scripts/` (dari `1st try` hingga penyelarasan final `6th try`).
2. Menyediakan modul ekstraksi fitur khusus untuk sensor dada dan pergelangan tangan pada `_forked/fb_code/`.
3. Menjadi pembanding metodologis antara pelabelan objektif stimulus protokol lab WESAD dengan pelabelan subjektif laporan mandiri kuesioner STAI pengguna.

---

## 2. Struktur Berkas & Modul Utama

```
_forked/
├── fb_code/                    # Modul pustaka utama pemrosesan sinyal & fitur
│   ├── feature_extraction.py   # Skrip utama ekstraksi fitur per subjek
│   ├── utils.py                # Fungsi pembantu filter sinyal, statistik, & FFT
│   ├── ecg.py                  # Pemrosesan sinyal ECG & ekstraksi fitur HRV
│   ├── respiration.py          # Pemrosesan sinyal napas & rasio I/E (inhalasi/ekshalasi)
│   ├── noise.py                # Analisis derau & pembersihan sinyal
│   ├── demographics.py         # Metadata subjek & korelasi demografis
│   ├── SQI.ipynb               # Signal Quality Index (indeks kualitas sinyal)
│   └── fb_environment.yml      # Spesifikasi conda environment bawaan
│
├── data/                       # Direktori keluaran data forked
│   ├── oct5_feats4.csv         # Dataset gabungan fitur WESAD (versi Oktober)
│   ├── may14_feats4.csv        # Dataset gabungan fitur WESAD (versi Mei)
│   └── subject_feats/          # CSV fitur individual (S2_feats.csv s/d S17_feats.csv)
│
├── Feature Based Model WESAD.ipynb # Eksplorasi model ML pada fitur WESAD
├── E4+Hexoskin_preprocess.ipynb    # Pra-pemrosesan sinyal wearable E4 & Hexoskin
├── Label_analysis.ipynb            # Analisis sebaran dan transisi label afektif
├── RADWear_correlation.ipynb       # Korelasi data fisiologis lapangan (RADWear study)
├── emd_noise/                      # Analisis dekomposisi derau berbasis Empirical Mode Decomposition
├── README.md & SET_UP.md           # Panduan konfigurasi lingkungan fb_code
└── syncfcns.py & data_wrangling.py # Sinkronisasi stempel waktu dan wrangling data
```

---

## 3. Perbedaan Kunci antara `_forked` dan Pipeline Skripsi (6th Try)

| Aspek | Repositori `_forked` (Alkurdi) | Pipeline Skripsi Pengguna (6th Try) |
|---|---|---|
| **Definisi Target** | Kondisi protokol WESAD: Baseline (1), Stress (2), Amusement (0) | Peningkatan skor STAI mandiri: 2-Class ($\Delta > 0$) & 3-Class ($\Delta > 1/24$) |
| **Metode Segmentasi** | Jendela 60 detik non-overlapping (tanpa stride) | Jendela 60 detik geser (*sliding window*) dengan pergeseran 6 detik (overlap 90%) |
| **Dimensi Fitur** | 120 fitur fisiologis dasar | 155 fitur (119 fitur dasar + 36 fitur kinetik sub-jendela akselerometer) |
| **Integritas Waktu** | Mengikuti timestamp sinkronisasi lab mentah | Memotong jendela ketat hanya pada rentang kondisi bertarget dengan respons STAI valid |
| **Metode Evaluasi** | Naive random split 80/20 (rentan data leakage) | Skema A (Replikasi Naive Split) dan Skema B (15-Fold Leave-One-Subject-Out CV) |

---

## 4. Tautan Terkait
- Peta konsep: [[MOC - Skripsi Repository]]
- Modul pustaka Python: [[Modul Library fb_code]]
- Pemfilteran sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Sinkronisasi data multimodal: [[Wrangling & Sinkronisasi Multimodal]]
- Dataset WESAD: [[Dataset WESAD]]
- Catatan komparasi metodologis: [[2026-10-06 Research Note]]
- Evolusi skrip: [[Scripts Pipeline Evolution (1st to 6th Try)]]
