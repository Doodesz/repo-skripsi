---
tags:
  - skripsi/data
  - dataset/artefacts
  - storage
aliases:
  - Data Artefacts
  - Output Data
---
# Dokumentasi Artefak & Penyimpanan Data (`Data/`)

Catatan inventarisasi berkas data hasil pra-pemrosesan, segmentasi jendela, dan ekstraksi fitur di dalam repositori.

---

## 1. Struktur Folder `Data/`

```
Data/
└── Preprocessed n STAI-labeled/
    ├── S2_stai.pkl     # Fitur sliding-window subjek 2 + label kuesioner STAI
    ├── S3_stai.pkl     # Fitur sliding-window subjek 3 + label kuesioner STAI
    ├── S4_stai.pkl     # ...
    ├── S5_stai.pkl
    ├── S6_stai.pkl
    ├── S7_stai.pkl
    ├── S8_stai.pkl
    ├── S9_stai.pkl
    ├── S10_stai.pkl
    ├── S11_stai.pkl
    ├── S13_stai.pkl
    ├── S14_stai.pkl
    ├── S15_stai.pkl
    ├── S16_stai.pkl
    └── S17_stai.pkl    # Total 15 berkas pickle
```

---

## 2. Karakteristik Berkas `S{id}_stai.pkl`

- **Dihasilkan oleh:** `Scripts/All-in-one/v2_stai.ipynb`.
- **Format:** Berkas serialized Python pickle yang berisi objek `pandas.DataFrame`.
- **Parameter Windowing:**
  - Durasi jendela: **60 detik**.
  - Pergeseran (*stride*): **12 detik**.
- **Jumlah Kolom:** 119 hingga 124 kolom:
  - 118 kolom fitur fisiologis multimodal (statistik, spektral, HRV, laju napas).
  - Kolom metadata dan target:
    - `subject_id`: Nomor identitas subjek.
    - `condition`: Nama kondisi eksperimen (Base, TSST, Fun, Medi, dll.).
    - `baseline_stai_mean`: Rata-rata skor STAI kondisi Base subjek tersebut.
    - `condition_stai_mean`: Rata-rata skor STAI pada kondisi terkait.
    - `binary_anxiety`: Target biner klasifikasi ($1$ jika $\Delta \text{STAI} \ge 0{,}2$, $0$ jika tidak).

---

## 3. Data Keluaran Repositori Referensi (`_forked/data/`)

Untuk perbandingan arsitektur, repositori referensi upstream menyimpan data fitur pada:
- `_forked/data/subject_feats/S{id}_feats.csv`: File CSV berisi fitur per subjek dengan segmentasi jendela 60 detik non-overlapping (tanpa stride).
- Kolom target berupa one-hot encoded: `0` (Amusement), `1` (Baseline), `2` (Stress).
- `_forked/data/oct5_feats4.csv` & `may14_feats4.csv`: Kumpulan fitur gabungan seluruh subjek yang digunakan pada eksperimen disertasi sebelumnya.

---

## 4. Tautan Terkait
- Peta konsep: [[MOC - Skripsi Repository]]
- Pipeline STAI: [[Pipeline STAI User (v2_stai)]]
- Script STAI: [[Script v2_stai]]
- Model ML: [[Machine Learning Models & Evaluation]]
- Katalog fitur lengkap: [[Katalog Fitur Fisiologis]]
