---
tags:
  - skripsi/komparasi
  - skripsi/metodologi
  - feature-engineering
  - dataset/wesad
aliases:
  - Perbandingan Pipeline STAI vs Forked
  - STAI vs Forked Comparison
  - Komparasi Pipeline
---
# Perbandingan Pipeline STAI vs Forked

Catatan analisis perbedaan mendasar arsitektur ekstraksi fitur, pelabelan, segmentasi jendela, dan format data antara skrip pengguna (`Scripts/All-in-one/v2_stai.ipynb`) dengan kode referensi upstream (`_forked/fb_code/feature_extraction.py`).

---

## 1. Tabel Komparasi Fitur & Pipeline

| Aspek Komparasi | Repositori Referensi `_forked` | Pipeline Pengguna Skripsi (`v2_stai.ipynb`) |
|:---|:---|:---|
| **Definisi Ground Truth** | **Kondisi Protokol Eksperimen WESAD**<br>• Baseline (1)<br>• Stress / TSST (2)<br>• Amusement (0) | **Peningkatan Respons Mandiri STAI**<br>• 1 (Anxiety): $\text{Mean STAI}_{\text{cond}} - \text{Mean STAI}_{\text{Base}} \ge 0{,}2$<br>• 0 (No Anxiety): delta $< 0{,}2$ |
| **Sifat Anotasi** | Objektif eksperimental (stimulus terjadwal lab). | Subjektif psikologis relatif terhadap baseline individu. |
| **Durasi Jendela (*Window Length*)** | 60 detik ($42.000$ sampel pada $700\text{ Hz}$). | 60 detik ($42.000$ sampel pada $700\text{ Hz}$). |
| **Pergeseran Jendela (*Stride / Shift*)** | **60 detik (Non-overlapping / Tumbling Window)**.<br>Tidak ada overlap antar-jendela berturutan. | **12 detik (Sliding Window dengan overlap 80%)**.<br>Menghasilkan sampel lebih padat pada durasi kondisi pendek. |
| **Penyaringan Batas Waktu (*Filtering*)** | Mengelompokkan seluruh sinyal berdasarkan label protokol mentah via `.groupby('label')`, lalu memotong jendela pada tiap grup. | **Strict Temporal Enclosure**.<br>Jendela wajib 100% berada dalam rentang kondisi bertarget dengan respons STAI valid. Jendela transisi dibuang (*dropped*). |
| **Format Kolom Target** | **One-Hot Encoded (3 Kolom)**:<br>`['0', '1', '2']` | **Target Biner Tunggal**:<br>`binary_anxiety` ($0$ atau $1$) |
| **Metadata Pelacak Jendela** | *Hanya kolom `subject`* saat file digabung. | **6 Kolom Metadata Pelacak**:<br>`subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `condition_stai_mean`, `baseline_stai_mean` |
| **Format Berkas Output** | CSV tabular (`S{id}_feats.csv`). | Serialized Pickle (`S{id}_stai.pkl`), menjaga native dtypes. |
| **Jumlah Jendela per Subjek (Contoh S2)** | **37 jendela**. | **131–301 jendela** (tergantung durasi kondisi valid). |

---

## 2. Implikasi Metodologis

1. **Inkompatibilitas Skor Evaluasi Head-to-Head:**
   Metrik performa model antara kedua pipeline tidak dapat dibandingkan langsung secara angka nominal karena targetnya berbeda secara konseptual:
   - Pipeline forked mengevaluasi: *"Apakah sinyal ini berasal dari subjek yang sedang diintervensi stres TSST?"*
   - Pipeline STAI mengevaluasi: *"Apakah subjek secara sadar merasakan kecemasan subjektif di atas baseline-nya?"*

2. **Dampak Pergeseran Jendela (Sliding Window 12s):**
   Overlap 80% meningkatkan jumlah sampel observasi, namun menghasilkan autokorelasi tinggi antar-jendela berturutan. Evaluasi model klasifikasi wajib menggunakan skema pengelompokan berdasarkan subjek (*GroupKFold* / *Leave-One-Subject-Out*) guna mencegah kebocoran data (*data leakage*).

---

## 3. Tautan Terkait
- Katalog 118 fitur fisiologis: [[Katalog Fitur Fisiologis]]
- Catatan pipeline STAI: [[Pipeline STAI User (v2_stai)]]
- Catatan pipeline protokol: [[Pipeline Forked (WESAD Protocol)]]
- Catatan temuan metodologis: [[Development Findings]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
