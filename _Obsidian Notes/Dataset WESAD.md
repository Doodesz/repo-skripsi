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

**Sumber:** UCI Machine Learning Repository — [WESAD Dataset #465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)  
**Lokasi di repo:** `WESAD/S{id}/` (S2–S17, tidak ada S1 dan S12)  
**Subjek:** 15 subjek sehat (S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17)

---

## 1. Struktur File Per Subjek

```
WESAD/
└── S{id}/
    ├── S{id}.pkl       # File utama — dict Python berisi sinyal + label + info subjek
    ├── S{id}_quest.csv # Respons kuesioner (STAI, PANAS, SAM, dll.) per kondisi
    ├── S{id}_respiban.txt  # Log alat RespiBAN dada (teks mentah)
    └── S{id}_readme.txt    # Informasi metadata sesi eksperimen per subjek
```

Berkas `.pkl` dimuat dengan `pickle.load(file, encoding='latin1')`.

---

## 2. Sinyal yang Tersedia (per Sensor)

### Sensor Dada — RespiBAN (700 Hz)
| Sinyal | Deskripsi |
|--------|-----------|
| `ECG`  | Electrocardiogram — ritme jantung, HRV |
| `EDA`  | Electrodermal Activity dada — sering disebut EDA_C |
| `EMG`  | Electromyography — aktivitas otot |
| `Resp` | Respiration — laju & amplitudo napas |
| `Temp` | Temperature dada — Temp_C |
| `ACC`  | Accelerometer 3-sumbu dada (x, y, z) — ACC_C |

### Sensor Pergelangan — Empatica E4
| Sinyal | Fs | Deskripsi |
|--------|----|-----------|
| `BVP`  | 64 Hz | Blood Volume Pulse (basis HRV wrist) |
| `EDA`  | 4 Hz  | Electrodermal Activity pergelangan |
| `TEMP` | 4 Hz  | Temperature kulit pergelangan |
| `ACC`  | 32 Hz | Accelerometer 3-sumbu pergelangan (x, y, z) |

### Label Protokol (700 Hz)
| Nilai | Kondisi |
|-------|---------|
| `0`   | Tidak diketahui / Transisi |
| `1`   | Baseline (istirahat awal) |
| `2`   | Stress (TSST — Trier Social Stress Test) |
| `3`   | Amusement (Menonton video lucu) |
| `4`   | Meditation (Meditasi panduan) |

---

## 3. Kuesioner STAI (State-Trait Anxiety Inventory)

Kuesioner STAI diisi peserta secara mandiri (*self-report*) beberapa kali dalam sesi, setelah kondisi selesai. Data tersimpan di `S{id}_quest.csv`. Format kolom kuesioner mencakup kolom kondisi (`condition`) dan skor butir STAI individual.

**Penting:**
- Instrumen STAI-S (State component) memiliki butir *reverse-scored*; pastikan scoring rule diikuti dengan benar.
- Tidak semua peserta merespons semua butir di semua kondisi.
- Parser dalam [[Script v2_stai]] merata-ratakan respons butir yang tersedia per kondisi.

---

## 4. Sampling Rate (Kamus Frekuensi)

```python
fs_dict = {
    'ACC': 32,   # wrist
    'BVP': 64,
    'EDA': 4,
    'TEMP': 4,
    'label': 700,
    'Resp': 700,  # chest
    'ECG': 700,
    'chest': 700
}
```

---

## 5. Catatan Penting Penggunaan

> [!warning] Data Hilang
> Tidak semua subjek memiliki data atau kuesioner lengkap. Cek `S{id}_readme.txt` untuk informasi anomali sesi per subjek.

> [!info] Subjek yang digunakan
> `[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]` — S1 dan S12 tidak diikutsertakan.

---

## 6. Tautan Terkait
- Peta konsep: [[MOC - Skripsi Repository]]
- Pipeline berbasis protokol: [[Pipeline Forked (WESAD Protocol)]]
- Pipeline berbasis STAI: [[Pipeline STAI User (v2_stai)]]
- Catatan ringkasan fitur: [[Modul Library fb_code]]
- Katalog fitur multimodal: [[Katalog Fitur Fisiologis]]
