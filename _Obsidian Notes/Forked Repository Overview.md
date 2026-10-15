---
tags:
  - skripsi/forked
  - skripsi/referensi
  - skripsi/codebase
aliases:
  - Forked Overview
  - anxietyFB
---
# Repositori Referensi Upstream (`_forked`)

Folder `_forked` merupakan repositori referensi dari proyek penelitian/disertasi sebelumnya yang berjudul **`anxietyFB`** (penelitian terkait prediksi kecemasan menggunakan sensor Empatica E4, Hexoskin, dan eksplorasi dataset WESAD/RADWear).

---

## 1. Peran dalam Penelitian Skripsi Ini

Folder ini berfungsi sebagai **basis pustaka sinyal dan tolok ukur komparasi (*baseline engine*)**:
1. Menjadi sumber algoritma pemrosesan sinyal fisiologis tervalidasi yang diadaptasi ke dalam folder `Scripts/All-in-one/` (`v1.ipynb` dan `v2_stai.ipynb`).
2. Menyediakan modul ekstraksi fitur khusus untuk sensor dada dan pergelangan tangan pada `_forked/fb_code/`.
3. Menjadi pembanding hasil ekstraksi fitur antara protokol WESAD asli dengan pipeline kuesioner STAI pengguna.

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

## 3. Perbedaan Kunci antara `_forked` dan Pipeline Skripsi

| Aspek | Repositori `_forked` | Pipeline Skripsi Pengguna (`v2_stai.ipynb`) |
|---|---|---|
| **Definisi Target** | Kondisi protokol WESAD: Baseline (1), Stress (2), Amusement (0) | Peningkatan skor kuesioner STAI $\ge 0{,}2$ terhadap baseline individu (Biner: 0 vs 1) |
| **Metode Segmentasi** | Jendela 60 detik non-overlapping (tanpa stride) | Jendela 60 detik geser (*sliding window*) dengan pergeseran 12 detik |
| **Integritas Waktu** | Mengikuti timestamp sinkronisasi lab mentah | Memotong jendela ketat hanya pada rentang kondisi bertarget dengan respons STAI valid |
| **Tujuan Akhir** | Pengenalan respons stres laboratorium | Evaluasi kelayakan prediksi State Anxiety psikologis |

---

## 4. Tautan Terkait
- Peta konsep: [[MOC - Skripsi Repository]]
- Modul pustaka Python: [[Modul Library fb_code]]
- Pemfilteran sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Sinkronisasi data multimodal: [[Wrangling & Sinkronisasi Multimodal]]
- Dataset WESAD: [[Dataset WESAD]]
- Catatan komparasi: [[Development Findings]]
