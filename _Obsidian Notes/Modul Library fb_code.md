---
tags:
  - skripsi/forked
  - skripsi/library
  - python-modules
aliases:
  - Modul Library fb_code
  - fb_code Library
---
# Modul Library `_forked/fb_code`

Dokumentasi berkas skrip pustaka Python pemrosesan sinyal fisiologis yang tersimpan di dalam direktori `_forked/fb_code/` dan diimpor oleh pipeline penelitian.

---

## 1. Rincian Modul Python

### A. `feature_extraction.py`
Skrip komputasi utama yang mengorkestrasi pemuatan data subjek melalui kelas `SubjectData`, pemanggilan filter sinyal, segmentasi jendela 60 detik, komputasi fitur multimodal, dan ekspor ke berkas CSV.

### B. `utils.py`
Modul utilitas matematika, filter, dan fungsi statistik sinyal:
- `butter_lowpass(cutoff, fs, order)` & `butter_lowpass_filter(data, cutoff, fs, order)`: Desain filter IIR Butterworth orde 5/6.
- `filterSignalFIR(values)`: Filter FIR untuk memperhalus sinyal akselerometer.
- `get_net_accel(data, part)`: Menghitung resultan akselerasi $\sqrt{x^2+y^2+z^2}$.
- `get_window_stats(data, label)`: Menghitung statistik mean, std, min, max per jendela.
- `get_peak_freq(series)`: Menghitung frekuensi puncak domain Fourier (FFT/PSD).
- `get_slope(series)`: Kemiringan regresi linier terhadap waktu.
- `get_dynamic_range(series)`: Rentang amplitudo sinyal ($\max - \min$).
- `get_absolute_integral(series)`: Luasan integral nilai absolut sinyal $\int |x(t)| dt$.

### C. `ecg.py`
Modul pemrosesan elektrokardiogram dada (700 Hz):
- Menggunakan pustaka `heartpy` untuk mendeteksi puncak R (R-peaks).
- Menghitung 7 metrik Heart Rate Variability (HRV): `bpm`, `ibi`, `sdnn`, `sdsd`, `rmssd`, `pnn20`, `pnn50`.

### D. `respiration.py`
Modul ekstraksi siklus pernapasan dari sinyal sabuk piezoelektrik dada (700 Hz):
- `get_resp_features(resp_series)`: Menghitung frekuensi pernapasan rata-rata (`rate`), durasi fase inhalasi (`Inhal_mean`, `Inhal_std`), durasi fase ekshalasi (`Exhal_mean`, `Exhal_std`), dan rasio durasi inhalasi/ekshalasi (`I/E`).

---

## 2. Tautan Terkait
- Gambaran umum repositori referensi: [[Forked Repository Overview]]
- Filtering sinyal: [[Pra-Pemrosesan Sinyal & Filtering]]
- Sinkronisasi data: [[Wrangling & Sinkronisasi Multimodal]]
- Katalog fitur yang dihasilkan: [[Katalog Fitur Fisiologis]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
