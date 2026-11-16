---
tags:
  - skripsi/dataset
  - dataset/wesad
  - sensor/physiological
aliases:
  - WESAD
  - Dataset WESAD
  - Wearable Stress and Affect Detection
---
# Dataset WESAD (Wearable Stress and Affect Detection)

> **Sumber:** UCI Machine Learning Repository — [WESAD Dataset #465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)  
> **Lokasi di Repo:** `WESAD/S{id}/` (S2–S17, subjek S1 dan S12 tidak tersedia dalam rilis publik WESAD)  
> **Subjek Valid:** 15 subjek sehat (`S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17`)  
> **Indeks Vault:** [[MOC - Skripsi Repository]]

---

## 1. Struktur Berkas per Subjek

```
WESAD/
└── S{id}/
    ├── S{id}.pkl           # Berkas utama — dict Python berisi sinyal sinkron + label + info subjek
    ├── S{id}_quest.csv     # Respons kuesioner self-report (STAI, PANAS, SAM, dll.) per kondisi
    ├── S{id}_respiban.txt  # Log alat RespiBAN dada (teks mentah)
    └── S{id}_readme.txt    # Metadata sesi eksperimen, anotasi anomali sensor per subjek
```

Berkas biner `.pkl` dimuat dengan pustaka Python:
```python
with open(f'WESAD/S{id}/S{id}.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
```

---

## 2. Sinyal Fisiologis yang Tersedia

### A. Sensor Dada — RespiBAN Professional (700 Hz)
| Sinyal | Kanal | Deskripsi |
|---|---|---|
| `ECG` | 1 Kanal | Electrocardiogram — ritme jantung, deteksi R-peaks, metrik HRV HeartPy |
| `EDA` | 1 Kanal | Electrodermal Activity dada (sering disebut `EDA_C`) |
| `EMG` | 1 Kanal | Electromyography otot trapezius — aktivitas kontraksi dan ketegangan otot |
| `Resp` | 1 Kanal | Respiration — sabuk pernapasan dada piezoelektrik, laju napas & rasio $I/E$ |
| `Temp` | 1 Kanal | Suhu kulit dada (`Temp_C`) |
| `ACC` | 3 Sumbu (x, y, z) | Accelerometer dada (`ACC_C`) |

### B. Sensor Pergelangan — Empatica E4
| Sinyal | Sampling Rate | Deskripsi |
|---|---|---|
| `BVP` | 64 Hz | Blood Volume Pulse (fotopletismografi optik wrist) |
| `EDA` | 4 Hz | Electrodermal Activity pergelangan tangan |
| `TEMP` | 4 Hz | Suhu kulit pergelangan tangan |
| `ACC` | 32 Hz | Accelerometer 3-sumbu pergelangan tangan (x, y, z) |

### C. Label Protokol Baku Laboratorium (700 Hz)
| Nilai | Kondisi Protokol | Deskripsi Stimulus |
|:---:|---|---|
| `0` | Transient / Unknown | Periode jeda dan transisi antar-kondisi |
| `1` | Baseline | Periode istirahat tenang 20 menit (membaca majalah santai) |
| `2` | Stress (TSST) | Trier Social Stress Test (tugas bicara di depan juri & aritmatika mental cepat) |
| `3` | Amusement | Menonton klip video komedi |
| `4` | Meditation | Panduan meditasi relaksasi |

---

## 3. Kuesioner STAI (State-Trait Anxiety Inventory)

Kuesioner STAI diisi peserta secara mandiri (*self-report*) setelah menyelesaikan tiap kondisi eksperimen. Data tersimpan di baris `# STAI` pada file `S{id}_quest.csv`.

### Aturan Skoring Baku Shortened STAI WESAD
Mengacu pada audit kuesioner pada [[2026-10-24 Research Note]]:
- WESAD menggunakan versi **shortened STAI (6 butir)** dengan skala Likert 1–4:
  1. *I feel at ease* (butir 0)
  2. *I feel nervous* (butir 1)
  3. *I am jittery* (butir 2)
  4. *I am relaxed* (butir 3)
  5. *I am worried* (butir 4)
  6. *I feel pleasant* (butir 5)
- **Reverse-Scoring:** Butir positif (1, 4, dan 6, atau indeks 0, 3, 5 pada Python) wajib dibalik dengan rumus:
  $$\text{scored} = 5 - \text{raw}$$
- Skor kondisi adalah jumlah (`sum`, rentang 6–24) atau rerata (`mean`, rentang 1.0–4.0) dari respons yang valid setelah reverse-scoring.

---

## 4. Kamus Frekuensi Sampling (*Sampling Rates*)

```python
fs_dict = {
    'ACC': 32,      # wrist ACC
    'BVP': 64,      # wrist BVP
    'EDA': 4,       # wrist EDA
    'TEMP': 4,      # wrist TEMP
    'label': 700,   # WESAD ground truth timeline
    'Resp': 700,    # chest respiration
    'ECG': 700,     # chest ECG
    'EMG': 700,     # chest EMG
    'Temp': 700,    # chest Temp
    'chest': 700    # chest ACC
}
```

---

## 5. Tautan Terkait
- Audit STAI dan reverse-scoring: [[2026-10-24 Research Note]]
- Pra-pemrosesan sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Sinkronisasi jam multimodal: [[Wrangling & Sinkronisasi Multimodal]]
- Katalog 155 fitur fisiologis: [[Katalog Fitur Fisiologis]]
- Spesifikasi model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
