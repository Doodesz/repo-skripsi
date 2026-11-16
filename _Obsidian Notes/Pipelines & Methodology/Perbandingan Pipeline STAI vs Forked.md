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

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[2026-10-06 Research Note]], [[Pipeline STAI User (v2_stai)]], [[Pipeline Forked (WESAD Protocol)]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]

Catatan analisis perbedaan mendasar arsitektur ekstraksi fitur, pelabelan, segmentasi jendela, dan format data antara pipeline kuesioner mandiri STAI dengan kode referensi upstream (`_forked/fb_code/feature_extraction.py`).

---

## 1. Tabel Komparasi Fitur & Pipeline

| Aspek Komparasi                           | Repositori Referensi `_forked` (Alkurdi)                                                                                       | Pipeline STAI Pengguna (6th Try)                                                                                                                                 |
| :---------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Definisi Ground Truth**                 | **Kondisi Protokol Eksperimen WESAD**<br>• Baseline (1)<br>• Stress / TSST (2)<br>• Amusement (0)                              | **Peningkatan Respons Mandiri STAI**<br>• 2-Class: $\Delta_{\text{sum}} > 0$ (Anxiety Increased)<br>• 3-Class: $\Delta_{\text{norm}} > 1/24$ (Anxiety Increased) |
| **Sifat Anotasi**                         | Objektif eksperimental (stimulus terjadwal laboratorium).                                                                      | Subjektif psikologis relatif terhadap baseline individu.                                                                                                         |
| **Durasi Jendela (*Window Length*)**      | 60 detik ($42.000$ sampel pada $700\text{ Hz}$).                                                                               | 60 detik ($42.000$ sampel pada $700\text{ Hz}$).                                                                                                                 |
| **Pergeseran Jendela (*Stride / Shift*)** | **60 detik (Non-overlapping / Tumbling Window)**.<br>Tidak ada overlap antar-jendela berturutan.                               | **6 detik (Sliding Window dengan overlap 90%)**.<br>Menghasilkan sampel representatif per kondisi afektif.                                                       |
| **Penyaringan Batas Waktu (*Filtering*)** | Mengelompokkan seluruh sinyal berdasarkan label protokol mentah via `.groupby('label')`, lalu memotong jendela pada tiap grup. | **Strict Temporal Enclosure**.<br>Jendela wajib 100% berada dalam rentang kondisi bertarget dengan respons STAI valid. Jendela perbatasan dibuang (*dropped*).   |
| **Dimensi Fitur Prediktif ($X$)**         | 120 fitur fisiologis.                                                                                                          | **155 fitur prediktif** (119 fitur dasar + 36 fitur kinetik sub-jendela akselerometer 5s/0.25s).                                                                 |
| **Format Kolom Target**                   | **One-Hot Encoded (3 Kolom)**:<br>`['0', '1', '2']`                                                                            | **Kolom Numerik Tunggal**:<br>`anxiety_label` (0/1 pada 2-class, 0/1/2 pada 3-class)                                                                             |
| **Metadata Pelacak Jendela**              | *Hanya kolom `subject`* saat file digabung.                                                                                    | **12 Kolom Metadata Pelacak**:<br>`subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `condition_stai_sum`, `baseline_stai_sum`, dll.       |
| **Format Berkas Output**                  | CSV tabular (`S{id}_feats.csv`).                                                                                               | Serialized Pickle (`S{id}_stai_*.pkl`), menjaga native dtypes.                                                                                                   |
| **Jumlah Jendela per Subjek**             | **~37–40 jendela** per subjek (total 577 jendela kohort).                                                                      | **257–289 jendela** per subjek (total 4.095 jendela kohort).                                                                                                     |

---

## 2. Implikasi Metodologis

1. **Inkompatibilitas Skor Evaluasi Head-to-Head:**
   Metrik performa model antara kedua pipeline tidak dapat dibandingkan langsung secara angka nominal karena targetnya berbeda secara konseptual:
   - Pipeline forked mengevaluasi: *"Apakah sinyal ini berasal dari subjek yang sedang diintervensi stres TSST?"*
   - Pipeline STAI mengevaluasi: *"Apakah subjek secara sadar merasakan kecemasan subjektif di atas baseline-nya?"*

2. **Dampak Pergeseran Jendela (Sliding Window 6s):**
   Overlap 90% meningkatkan jumlah sampel observasi, namun menghasilkan autokorelasi temporal tinggi antar-jendela berturutan. Evaluasi model klasifikasi wajib menggunakan skema pengelompokan berdasarkan subjek (**Leave-One-Subject-Out / LOSO**) guna mencegah kebocoran data (*data leakage*).

---

## 3. Tautan Terkait
- Katalog 155 fitur fisiologis: [[Katalog Fitur Fisiologis]]
- Catatan pipeline STAI: [[Pipeline STAI User (v2_stai)]]
- Catatan pipeline protokol: [[Pipeline Forked (WESAD Protocol)]]
- Catatan temuan metodologis: [[2026-10-06 Research Note]]
- Spesifikasi model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
- Evaluasi model: [[Machine Learning Models & Evaluation]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
