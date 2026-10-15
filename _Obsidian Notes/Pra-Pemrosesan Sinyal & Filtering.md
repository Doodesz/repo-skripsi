---
tags:
  - skripsi/preprocessing
  - signal-processing
  - sensor/filtering
aliases:
  - Pra-Pemrosesan Sinyal
  - Signal Filtering
---
# Pra-Pemrosesan Sinyal & Filtering

Catatan teknis mengenai mekanisme pembersihan derau (*filtering*) dan dekomposisi komponen fisiologis pada sensor dada (RespiBAN) dan pergelangan (Empatica E4).

---

## 1. Pemfilteran per Sensor

### A. Electrodermal Activity (EDA)
- **Karakteristik Fisik:** Sinyal konduktansi kulit bergerak lambat ($< 1\text{ Hz}$). Frekuensi tinggi merupakan artefak gerakan atau derau instrumen.
- **Filter Digital:** Butterworth Lowpass Filter (orde 6, frekuensi cut-off $f_c = 1.0\text{ Hz}$):
  - Diterapkan pada EDA pergelangan ($f_s = 4\text{ Hz}$) dan dada ($f_s = 700\text{ Hz}$).
- **Dekomposisi Phasic-Tonic via NeuroKit2 (`nk.eda_phasic`):**
  - **SCL (Skin Conductance Level / Tonic):** Komponen lambat yang mencerminkan tingkat gairah otonom dasar (*arousal*).
  - **SCR (Skin Conductance Response / Phasic):** Puncak-puncak cepat transien yang mencerminkan respons langsung terhadap stimulus sesaat.

### B. Accelerometer (ACC)
- **Filter Digital:** Finite Impulse Response (FIR) lowpass filter untuk memperhalus fluktuasi getaran mikro.
- **Resultan Akselerasi (*Net Acceleration*):**
  $$\text{net\_acc} = \sqrt{\text{ACC}_x^2 + \text{ACC}_y^2 + \text{ACC}_z^2}$$

### C. Kardiovaskular (ECG & BVP)
- **ECG Dada (700 Hz):** Dideteksi puncak R (*R-peaks*) menggunakan algoritma HeartPy dan BioSPPy untuk menghasilkan rangkaian Inter-Beat Interval (IBI).
- **BVP Pergelangan (64 Hz):** Dianalisis spektrum frekuensi dominannya menggunakan estimasi Fast Fourier Transform (FFT).

### D. Sabuk Pernapasan Dada (Resp_C)
- Sinyal sabuk piezoelektrik dada (700 Hz) dianalisis untuk mendeteksi siklus respirasi, memisahkan fase inspirasi (*trough-to-peak*) dan fase ekspirasi (*peak-to-trough*).

---

## 2. Tautan Terkait
- Sinkronisasi multisensor: [[Wrangling & Sinkronisasi Multimodal]]
- Modul Python library: [[Modul Library fb_code]]
- Katalog fitur yang dihasilkan: [[Katalog Fitur Fisiologis]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
