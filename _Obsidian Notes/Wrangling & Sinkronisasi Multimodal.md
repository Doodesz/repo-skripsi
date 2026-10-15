---
tags:
  - skripsi/data-wrangling
  - signal-processing
  - multimodal
aliases:
  - Wrangling & Sinkronisasi Multimodal
  - Multimodal Synchronization
---
# Wrangling & Sinkronisasi Multimodal

Catatan teknis mengenai mekanisme penggabungan data fisiologis dari perangkat dengan frekuensi sampling (*sampling rate*) berbeda pada dataset WESAD.

---

## 1. Tantangan Frekuensi Sampling Heterogen

Dataset WESAD merekam sinyal menggunakan dua perangkat simultan:
- **RespiBAN (Dada):** $700\text{ Hz}$ seragam untuk seluruh sensor (ECG, Resp, EDA, EMG, Temp, ACC).
- **Empatica E4 (Pergelangan):**
  - BVP: $64\text{ Hz}$
  - ACC: $32\text{ Hz}$
  - EDA: $4\text{ Hz}$
  - TEMP: $4\text{ Hz}$

---

## 2. Strategi Sinkronisasi Outer-Join Berbasis Datetime

Implementasi pada repositori (`_forked/fb_code/feature_extraction.py` dan `Scripts/All-in-one/`):
1. **Pemberian Indeks Waktu Relatif:**
   Setiap sampel sensor diberi indeks waktu relatif dalam detik:
   $$t_i = i \times \frac{1}{f_s}$$
2. **Konversi ke Datetime Absolut:**
   Indeks waktu dikonversi ke objek stempel waktu pandas:
   ```python
   df.index = pd.to_datetime(df.index, unit='s')
   ```
3. **Penggabungan Luar (*Outer Join*):**
   Seluruh DataFrame sensor digabungkan bertahap menggunakan operasi `outer join`. Basis resolusi waktu tertinggi ($700\text{ Hz}$) menjadi kerangka waktu acuan, sedangkan sensor berfrekuensi lebih rendah secara alami menghasilkan nilai `NaN` pada titik waktu di antaranya.
4. **Pembersihan Tanpa Interpolasi Buatan:**
   Pada tahap ekstraksi fitur per jendela, fungsi komputasi mengekstrak sinyal sensor spesifik menggunakan metode `.dropna()` pada kolom yang bersangkutan. Hal ini menjaga spektrum frekuensi asli sinyal tanpa distorsi interpolasi buatan.

---

## 3. Tautan Terkait
- Pemfilteran sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Modul Python library: [[Modul Library fb_code]]
- Katalog fitur yang dihasilkan: [[Katalog Fitur Fisiologis]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
