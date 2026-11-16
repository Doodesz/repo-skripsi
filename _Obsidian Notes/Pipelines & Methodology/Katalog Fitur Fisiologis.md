---
tags:
  - skripsi/features
  - skripsi/feature-engineering
  - dataset/wesad
  - multimodal
aliases:
  - Extracted Features Catalog
  - Katalog Fitur Fisiologis
  - Daftar Fitur
  - Fitur Fisiologis Multimodal
---
# Katalog Fitur Fisiologis Multimodal

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Pra-Pemrosesan Sinyal & Filtering]], [[Wrangling & Sinkronisasi Multimodal]], [[Scripts Pipeline Evolution (1st to 6th Try)]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]

Dokumentasi taksonomi dan katalog fitur fisiologis multimodal yang diekstraksi dari sinyal WESAD pada setiap jendela observasi 60 detik.

Pada implementasi awal (1st try), ekstraksi menghasilkan **118 fitur dasar**. Sejak iterasi kedua (2nd try) hingga iterasi final (6th try), fitur diperluas dengan **36 fitur kinematik sub-jendela akselerometer**, menghasilkan **154–155 dimensi fitur prediktif ($X$)** yang menggabungkan sensor dada (**RespiBAN**, 700 Hz) dan sensor pergelangan tangan (**Empatica E4**, 4–64 Hz).

---

## 1. Komposisi Taksonomi 155 Fitur Prediktif

```
Total Dimensi Fitur Prediktif (X): 155 Fitur
├── Statistik Dasar Sinyal (Mean, Std, Min, Max): 64 fitur (16 sinyal × 4 metrik)
├── Akselerometer Kinematik Global: 15 fitur (abs integral, peak freq)
├── Akselerometer Sub-Jendela Kinematik (5s / 0.25s): 36 fitur (std, integral, peak freq per sumbu)
├── Electrodermal Activity Dinamik (EDA): 4 fitur (slope, dynamic range)
├── Elektrokardiogram Dada (ECG / HRV HeartPy): 7 fitur (bpm, ibi, sdnn, sdsd, rmssd, pnn20, pnn50)
├── Blood Volume Pulse Pergelangan (BVP): 1 fitur (peak frequency)
├── Elektromiografi Dada (EMG): 2 fitur (dynamic range, abs integral)
├── Sabuk Respirasi Dada (Resp_C): 6 fitur (rate, inhal/exhal stats, I/E ratio)
└── Temperatur Kulit (TEMP Dinamik): 4 fitur (dynamic range, slope)
```

---

## 2. Rincian Fitur per Domain Sensor

### A. Fitur Statistik Dasar Jendela (64 Fitur)
Setiap jendela menghitung 4 metrik statistik agregasi dasar:
$$\text{mean}, \quad \text{std}, \quad \text{min}, \quad \text{max}$$
Diterapkan pada 16 kanal sinyal:
1. `net_acc_C` (Magnitudo akselerasi dada)
2. `net_acc` (Magnitudo akselerasi pergelangan)
3. `EDA` (Konduktansi kulit pergelangan tangan)
4. `EDA_SCR` (Komponen Phasic pergelangan via NeuroKit2)
5. `EDA_SCL` (Komponen Tonic pergelangan via NeuroKit2)
6. `BVP` (Blood Volume Pulse pergelangan)
7. `TEMP` (Temperatur kulit pergelangan)
8. `ACC_x` (Akselerometer sumbu X pergelangan)
9. `ACC_y` (Akselerometer sumbu Y pergelangan)
10. `ACC_z` (Akselerometer sumbu Z pergelangan)
11. `ECG` (Elektrokardiogram dada)
12. `EDA_C` (Konduktansi kulit dada)
13. `EDA_SCR_C` (Komponen Phasic dada via NeuroKit2)
14. `EDA_SCL_C` (Komponen Tonic dada via NeuroKit2)
15. `ACC_x_C` (Akselerometer sumbu X dada)
16. `ACC_y_C` (Akselerometer sumbu Y dada)
17. `ACC_z_C` (Akselerometer sumbu Z dada)
18. `EMG_C` (Elektromiografi dada)
19. `Resp_C` (Sabuk pernapasan dada)
20. `Temp_C` (Temperatur kulit dada)

*Format penamaan kolom:* `{Sinyal}_{Metrik}` (contoh: `net_acc_mean`, `EDA_SCR_std`, `ECG_max`, dll.).

---

### B. Akselerometer & Spektral Kinematik Global (15 Fitur)
Mengukur kuantitas pergerakan fisik dan frekuensi getaran tubuh:
- **Integral Nilai Absolut ($\int |a(t)| dt$):**
  - `net_acc_abs_integral` (wrist) & `net_acc_C_abs_integral` (chest)
  - `ACC_x_abs_integral`, `ACC_y_abs_integral`, `ACC_z_abs_integral` (wrist)
  - `ACC_x_C_abs_integral`, `ACC_y_C_abs_integral`, `ACC_z_C_abs_integral` (chest)
- **Frekuensi Puncak Spektral FFT/PSD (*Peak Frequency*):**
  - `ACC_x_peak_freq`, `ACC_y_peak_freq`, `ACC_z_peak_freq` (wrist)
  - `ACC_x_C_peak_freq`, `ACC_y_C_peak_freq`, `ACC_z_C_peak_freq` (chest)

---

### C. Akselerometer Sub-Jendela Kinematik (36 Fitur)
Diperkenalkan pada `2nd try` (`v2_stai_acc_subwindow.ipynb`) dan dipertahankan hingga `6th try`:
- Mengurai jendela utama 60 detik menjadi sub-jendela pendek berdurasi **5.0 detik** dengan pergeseran **0.25 detik**.
- Menghitung 3 metrik per sub-jendela pada 6 sumbu akselerometer (3 sumbu wrist: `ACC_x`, `ACC_y`, `ACC_z` dan 3 sumbu chest: `ACC_x_C`, `ACC_y_C`, `ACC_z_C`):
  1. Standar deviasi getaran (`std`)
  2. Luasan integral absolut (`abs_integral`)
  3. Frekuensi puncak getaran (`peak_freq`)
- Mengagregasikan hasil seluruh sub-jendela menjadi nilai **mean** dan **std**:
  - `ACC_sub_{axis}_std_mean` & `ACC_sub_{axis}_std_std`
  - `ACC_sub_{axis}_abs_integral_mean` & `ACC_sub_{axis}_abs_integral_std`
  - `ACC_sub_{axis}_peak_freq_mean` & `ACC_sub_{axis}_peak_freq_std`
  *(Total: 6 sumbu × 3 metrik × 2 agregasi = **36 fitur**)*.

---

### D. Kardiovaskular (ECG Dada & BVP Pergelangan) (8 Fitur)
- **BVP Pergelangan (1 Fitur):**
  - `BVP_peak_freq` — Frekuensi dominan denyut nadi perifer berbasis estimasi FFT.
- **ECG Dada via HeartPy (7 Fitur):**
  - `ECG_bpm` — Heart Rate rata-rata (Beats Per Minute).
  - `ECG_ibi` — Inter-Beat Interval rata-rata (ms).
  - `ECG_sdnn` — Standar deviasi interval normal-ke-normal (SDNN).
  - `ECG_sdsd` — Standar deviasi dari selisih interval RR yang berurutan.
  - `ECG_rmssd` — Root Mean Square of Successive Differences (indikator modulasi parasimpatis).
  - `ECG_pnn20` — Persentase selisih interval RR berturutan $> 20\text{ ms}$.
  - `ECG_pnn50` — Persentase selisih interval RR berturutan $> 50\text{ ms}$.

---

### E. Elektrodermal (EDA / GSR) (4 Fitur)
- **Kemiringan Regresi Linier (*Slope*):**
  - `EDA_slope` (wrist) & `EDA_C_slope` (chest)
- **Rentang Dinamis (*Dynamic Range* = $\max - \min$):**
  - `EDA_drange` (wrist) & `EDA_C_drange` (chest)

---

### F. Elektromiografi (EMG Dada) (2 Fitur)
- `EMG_drange` — Rentang dinamis amplitudo tegangan otot trapezius.
- `EMG_abs_integral` — Akumulasi energi kontraksi otot ($\int |\text{EMG}(t)| dt$).

---

### G. Pernapasan Sabuk Dada (Resp_C) (6 Fitur)
- `Resp_C_rate` — Laju frekuensi pernapasan (napas per menit).
- `Resp_C_Inhal_mean` & `Resp_C_Inhal_std` — Rerata dan simpangan baku durasi inspirasi.
- `Resp_C_Exhal_mean` & `Resp_C_Exhal_std` — Rerata dan simpangan baku durasi ekspirasi.
- `Resp_C_I/E` — Rasio durasi inhalasi terhadap ekshalasi ($T_{\text{inhal}} / T_{\text{exhal}}$).

---

### H. Suhu Kulit (TEMP) (4 Fitur)
- `TEMP_drange` & `TEMP_C_drange` — Rentang fluktuasi temperatur kulit.
- `TEMP_slope` & `TEMP_C_slope` — Gradien laju perubahan temperatur kulit terhadap waktu.

---

## 3. Tautan Terkait
- Sinkronisasi sinyal multimodal: [[Wrangling & Sinkronisasi Multimodal]]
- Pembersihan sinyal & filtering: [[Pra-Pemrosesan Sinyal & Filtering]]
- Evolusi skrip: [[Scripts Pipeline Evolution (1st to 6th Try)]]
- Spesifikasi model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
- Evaluasi & performa model: [[Machine Learning Models & Evaluation]]
