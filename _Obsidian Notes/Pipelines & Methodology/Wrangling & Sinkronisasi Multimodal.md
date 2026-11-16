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

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Pra-Pemrosesan Sinyal & Filtering]], [[Modul Library fb_code]], [[Katalog Fitur Fisiologis]], [[2026-11-12 Research Note]]

Catatan teknis mengenai mekanisme penggabungan data fisiologis dari perangkat dengan frekuensi sampling (*sampling rate*) heterogen pada dataset WESAD.

---

## 1. Tantangan Frekuensi Sampling Heterogen

Dataset WESAD merekam sinyal menggunakan dua perangkat simultan dengan clock terpisah:
- **RespiBAN (Dada):** $700\text{ Hz}$ seragam untuk seluruh sensor (ECG, Resp, EDA, EMG, Temp, ACC).
- **Empatica E4 (Pergelangan):**
  - BVP: $64\text{ Hz}$
  - ACC: $32\text{ Hz}$
  - EDA: $4\text{ Hz}$
  - TEMP: $4\text{ Hz}$

---

## 2. Strategi Sinkronisasi Outer-Join Berbasis Datetime (Alkurdi Style)

Diadopsi dari arsitektur Alkurdi et al. (`_forked/fb_code/feature_extraction.py`) dan dioptimalkan pada `6th try` (`Scripts/6th try/`):
1. **Pemberian Indeks Waktu Relatif Berbasis Nanosekon:**
   Setiap sampel sensor diberi indeks waktu berpresisi nanosekon:
   ```python
   time_ns = np.round(np.arange(len(signal)) * (1e9 / freq)).astype(np.int64)
   df.index = pd.to_datetime(time_ns, unit='ns')
   ```
2. **Penggabungan Luar (*Outer Join*):**
   Seluruh DataFrame sensor digabungkan secara vectorized menggunakan `pd.concat(..., axis=1)`. Basis resolusi waktu tertinggi ($700\text{ Hz}$) menjadi kerangka waktu acuan, sedangkan sensor berfrekuensi lebih rendah menghasilkan nilai `NaN` alami pada titik waktu di antaranya.
3. **Penyelarasan Label Afektif & Kuesioner:**
   Label eksperimen dan kuesioner diselaraskan ke dalam kerangka waktu menggunakan metode *forward fill* / *backward fill* (`bfill`/`ffill`) agar setiap stempel waktu memiliki penanda kondisi yang valid.
4. **Pembersihan Tanpa Distorsi Interpolasi:**
   Pada tahap ekstraksi fitur per jendela, fungsi komputasi mengekstrak sinyal sensor spesifik menggunakan metode `.dropna()` pada kolom yang bersangkutan. Hal ini menjaga spektrum frekuensi asli sinyal tanpa distorsi interpolasi buatan.

---

## 3. Tautan Terkait
- Pemfilteran sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Modul Python library: [[Modul Library fb_code]]
- Katalog fitur yang dihasilkan: [[Katalog Fitur Fisiologis]]
- Penyelarasan Alkurdi 6th try: [[2026-11-12 Research Note]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
