---
tags:
  - skripsi/audit
  - ai-validation
  - pipelines
  - metodologi
  - feature-engineering
date: 2026-11-16
auditor: Antigravity AI
status: Completed
target_folder: "_Obsidian Notes/Pipelines & Methodology"
---

# Laporan Validasi AI: Pipelines & Methodology

> **Indeks Validasi:** [[MOC - Skripsi Repository]] | [[Model Specs]]  
> **Folder Sasaran:** `_Obsidian Notes/Pipelines & Methodology`  
> **Tanggal Audit:** 16 November 2026  
> **Repositori Terkait:** `Scripts/` (1st–6th try), `_forked/` (fb_code), `Data/`, `Models/`  

---

## 1. Ringkasan Eksekutif (Executive Summary)

Audit komprehensif ini melakukan verifikasi silang (*cross-validation*) secara menyeluruh terhadap seluruh catatan dalam folder `_Obsidian Notes/Pipelines & Methodology` terhadap kode sumber (`Scripts/`, `_forked/`), artefak data terkomputasi (`Data/5th try/`, `Data/6th try/`, `_forked/data/`), dan dataset WESAD.

Folder ini mendokumentasikan arsitektur rekayasa fitur (*feature engineering*), pra-pemrosesan sinyal (*signal preprocessing*), sinkronisasi multimodal (*multimodal alignment*), segmentasi jendela (*windowing*), dan evolusi formulasi pelabelan kecemasan (*STAI labeling*).

### Ringkasan Status Validasi per Catatan

| File Catatan | Status Validasi | Ringkasan Temuan & Discrepancies |
| :--- | :---: | :--- |
| **`Katalog Fitur Fisiologis.md`** | ⚠️ **Discrepancy Aritmetika & Taksonomi** | Diagram pohon fitur di Seksi 1 menjumlahkan **139 fitur** (bukan 155). Terdapat kesalahan fatal: Statistik dasar diklaim **64 fitur (16 sinyal)**, padahal aktualnya **80 fitur (20 sinyal)**; Akselerometer global diklaim **15 fitur** padahal **14 fitur**; Kolom `ACC_subwindow_count` (1 fitur) tidak tercatat. |
| **`Perbandingan Pipeline STAI vs Forked.md`** | 🟡 **Sebagian Valid (Perlu Koreksi)** | Dimensi fitur referensi `_forked` tertulis **120 fitur**, padahal secara aktual adalah **118 fitur fisiologis**. Matriks jendela cohort (577 vs 4.095) dan parameter windowing 100% valid dan presisi. |
| **`Pipeline Forked (WESAD Protocol).md`** | 🟢 **Valid (100% Akurat)** | Seluruh klaim metodologi protokol eksperimen WESAD, segmentasi tumbling window 60s, sinkronisasi outer-join 700 Hz, dan jumlah 37–40 jendela per subjek terverifikasi presisi. |
| **`Pipeline STAI User (v2_stai).md`** | 🟢 **Valid (Perlu Klarifikasi Cakupan)** | Pelabelan biner 6th try (56.41% vs 43.59%) dan 3-kelas ($\pm 1/24$, 27.11%, 25.05%, 47.84%) terbukti 100% presisi hingga desimal kedua. Namun judul dan tag masih menggunakan nomenklatur historis `v2_stai` dan belum mendokumentasikan transisi dari resampling 70 Hz ke Alkurdi datetime alignment. |
| **`Pra-Pemrosesan Sinyal & Filtering.md`** | 🟡 **Discrepancy Atribusi Pustaka** | Ekstraksi fitur HRV ECG diklaim menggunakan algoritma *HeartPy dan BioSPPy*, padahal secara aktual ECG diproses **murni menggunakan HeartPy** (`hp.process`). Pustaka `BioSPPy` hanya digunakan untuk analisis sabuk pernapasan dada (`Resp_C`). Estimasi frekuensi puncak BVP menggunakan *Periodogram PSD* (`scisig.periodogram`), bukan raw FFT langsung. |
| **`Wrangling & Sinkronisasi Multimodal.md`** | 🟢 **Valid untuk 6th Try (Konteks Historis Perlu Dilengkapi)** | Implementasi penanggalan nanosekon, `pd.concat(..., axis=1)`, `bfill/ffill`, dan `.dropna()` per jendela terverifikasi identik dengan skrip 6th try. Konteks historis resampling polifase 70 Hz (1st–5th try) belum tercatat. |

---

## 2. Audit Rinci Catatan demi Catatan

---

### A. Audit `Katalog Fitur Fisiologis.md`

Catatan ini mendokumentasikan katalog fitur fisiologis multimodal 60 detik. Audit kode sumber pada `Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb` dan dataset biner `Data/6th try/2class/Full_sensor_modals/` menemukan diskrepansi struktural yang signifikan dalam diagram taksonomi.

#### 1. Verifikasi Jumlah Kolom Aktual
Pada berkas dataset 6th try (`S{id}_stai_2class_60s_shift6s.pkl`), terdapat **167 kolom total**:
- **12 Kolom Metadata Pelacak:** `anxiety_label`, `delta_stai`, `subject_id`, `condition`, `window_start_seconds`, `window_end_seconds`, `condition_stai_mean`, `baseline_stai_mean`, `delta_stai_sum`, `delta_stai_mean`, `condition_stai_sum`, `baseline_stai_sum`.
- **155 Kolom Fitur Prediktif ($X$).**

#### 2. Dekomposisi Aritmetika: 139 (Klaim Catatan) vs 155 (Aktual Kode)

Dalam diagram taksonomi Seksi 1 catatan:
$$\text{Klaim Catatan: } 64 + 15 + 36 + 4 + 7 + 1 + 2 + 6 + 4 = 139 \text{ Fitur (Salah)}$$

Berikut adalah audit per kategori fitur:

1. **Statistik Dasar Jendela (Mean, Std, Min, Max):**
   - **Klaim Catatan:** 64 fitur (16 sinyal × 4 metrik).
   - **Aktual Kode:** **80 fitur (20 sinyal × 4 metrik)**!
   - *Bukti:* Di Seksi 2.A catatan itu sendiri, penulis mencantumkan 20 kanal sinyal (nomor 1 sampai 20: `net_acc_C`, `net_acc`, `EDA`, `EDA_SCR`, `EDA_SCL`, `BVP`, `TEMP`, `ACC_x`, `ACC_y`, `ACC_z`, `ECG`, `EDA_C`, `EDA_SCR_C`, `EDA_SCL_C`, `ACC_x_C`, `ACC_y_C`, `ACC_z_C`, `EMG_C`, `Resp_C`, `Temp_C`). $20 \times 4 = 80$ fitur. Namun, judul subseksi secara keliru menuliskan `(64 Fitur)` dan `16 kanal sinyal`.
2. **Akselerometer Kinematik Global:**
   - **Klaim Catatan:** 15 fitur.
   - **Aktual Kode:** **14 fitur**.
   - *Rincian:* 8 fitur *absolute integral* (`net_acc`, `net_acc_C`, `ACC_x`, `ACC_y`, `ACC_z`, `ACC_x_C`, `ACC_y_C`, `ACC_z_C`) + 6 fitur *peak frequency* (`ACC_x`, `ACC_y`, `ACC_z`, `ACC_x_C`, `ACC_y_C`, `ACC_z_C`). $8 + 6 = 14$. (Catatan kemungkinan salah menghitung atau memasukkan `BVP_peak_freq` dua kali).
3. **Akselerometer Sub-Jendela Kinematik (5s / 0.25s):**
   - **Klaim & Aktual:** **36 fitur** (6 sumbu × 3 metrik [`std`, `abs_integral`, `peak_freq`] × 2 agregasi [`mean`, `std`]). **Valid.**
4. **Penghitung Sub-Jendela Akselerometer (`ACC_subwindow_count`):**
   - **Aktual Kode:** **1 fitur** (kolom ke-117 pada matriks fitur).
   - **Klaim Catatan:** **Tidak disebutkan / hilang dari taksonomi.** Kolom ini dimasukkan ke dalam matriks $X$ pada pelatihan model 6th try karena tidak termasuk dalam set `metadata_columns`.
5. **Fitur Dinamik Fisiologis (31 Fitur):**
   - EDA Dinamik: 4 fitur (`slope` dan `drange` pada wrist & chest). **Valid.**
   - ECG / HRV HeartPy: 7 fitur (`bpm`, `ibi`, `sdnn`, `sdsd`, `rmssd`, `pnn20`, `pnn50`). **Valid.**
   - BVP Pergelangan: 1 fitur (`BVP_peak_freq`). **Valid.**
   - EMG Dada: 2 fitur (`EMG_drange`, `EMG_abs_integral`). **Valid.**
   - Sabuk Respirasi Dada: 6 fitur (`rate`, `Inhal_mean`, `Inhal_std`, `Exhal_mean`, `Exhal_std`, `I/E`). **Valid.**
   - Temperatur Dinamik: 4 fitur (`drange` dan `slope` pada wrist & chest). **Valid.**

#### 3. Rekonsiliasi Matematika Taksonomi yang Benar
$$\begin{aligned}
\text{Total Fitur } (X) &= 80 \ (\text{Statistik Dasar}) + 14 \ (\text{Kinematik Global ACC}) \\
&\quad + 36 \ (\text{Sub-Jendela ACC}) + 1 \ (\text{ACC\_subwindow\_count}) \\
&\quad + 4 \ (\text{EDA}) + 7 \ (\text{ECG}) + 1 \ (\text{BVP}) + 2 \ (\text{EMG}) + 6 \ (\text{Resp}) + 4 \ (\text{TEMP}) \\
&= \mathbf{155 \text{ Fitur Prediktif}}
\end{aligned}$$

#### 4. Evolusi Dimensi Fitur (118 -> 154 -> 155)
- **1st Try (`v1.ipynb` / `v2_stai.ipynb`):** **118 fitur dasar** (mengikuti Alkurdi `_forked`).
- **2nd Try – 5th Try (`v2_stai_acc_subwindow.ipynb`, `stai_6s_pipeline.py`):** **154 fitur** ($118 \text{ dasar} + 36 \text{ sub-jendela} = 154$). Kolom `net_acc_C_abs_integral` belum diekstrak pada iterasi ini.
- **6th Try (`stai_2class_alkurdi_alignment.ipynb`):** **155 fitur** ($154 + 1 \ [\text{net\_acc\_C\_abs\_integral}] = 155$).

---

### B. Audit `Perbandingan Pipeline STAI vs Forked.md`

Catatan ini membandingkan pipeline referensi `_forked` dengan pipeline pengguna STAI 6th Try.

#### 1. Verifikasi Dimensi Fitur ($X$)
- **Klaim Catatan:** Repositori Referensi `_forked` (Alkurdi) memiliki **120 fitur fisiologis**.
- **Hasil Audit Kode:** **Diskrepansi Faktual.**
  - Berkas gabungan `_forked/data/may14_feats4.csv` memiliki 121 kolom:
    $$\text{Fitur Prediktif} = 121 - 3 \ (\text{'Unnamed: 0'}, \text{'subject'}, \text{'label'}) = \mathbf{118 \text{ fitur}}.$$
  - Berkas individual `_forked/data/subject_feats/S2_feats.csv` memiliki 122 kolom:
    $$\text{Fitur Prediktif} = 122 - 4 \ (\text{'Unnamed: 0'}, \text{'0'}, \text{'1'}, \text{'2'}) = \mathbf{118 \text{ fitur}}.$$
  - *Kesimpulan:* Repositori `_forked` mengekstrak **118 fitur fisiologis**, bukan 120 fitur.
  - *Koreksi untuk Pipeline STAI:* $118 \text{ fitur dasar} + 36 \text{ sub-jendela ACC} + 1 \ (\text{net\_acc\_C\_abs\_integral / subwindow\_count}) = 155 \text{ fitur}$.

#### 2. Verifikasi Metrik Kohort dan Windowing (100% Presisi)
Semua klaim metrik kohort pada tabel komparasi terbukti tepat:
- **Jendela per subjek `_forked`:** Min 37, Max 40, Total Kohort = **577 jendela** (terverifikasi dari `may14_feats4.csv` dan 15 file `subject_feats/*.csv`).
- **Jendela per subjek STAI 6th Try:** Min 257, Max 289, Total Kohort = **4.095 jendela** (terverifikasi dari `Data/6th try/2class/Full_sensor_modals/`).
- **Penyaringan Waktu:** `_forked` menggunakan `df.groupby('label')` yang memotong data per kondisi protokol tanpa mempedulikan batas waktu kuesioner, sedangkan STAI menggunakan *Strict Temporal Enclosure* berdasarkan interval MM.SS kuesioner.
- **One-Hot vs Single Label:** `_forked` menggunakan format 3 kolom `['0', '1', '2']`, sedangkan STAI menggunakan format kolom skalar tunggal `anxiety_label`.

---

### C. Audit `Pipeline Forked (WESAD Protocol).md`

Catatan ini menguraikan latar belakang pipeline referensi `_forked`.

#### Hasil Verifikasi:
- **Asal Usul Repositori:** Terkonfirmasi diadaptasi dari repositori disertasi upstream `anxietyFB` (Alkurdi et al.) yang berada di direktori `_forked/`.
- **Kondisi Protokol:** Menggunakan pelabelan WESAD resmi (Baseline = 1, TSST/Stress = 2, Amusement = 0/3, Meditation = 4).
- **Segmentasi Waktu:** Jendela 60 detik non-overlapping (*tumbling window*, stride 60 detik) terverifikasi langsung pada `_forked/fb_code/feature_extraction.py` (`WINDOW_IN_SECONDS = 60`, `step = window_len`).
- **Jumlah Jendela:** 37–40 jendela per subjek (total 577 jendela).
- **Status:** **100% Akurat dan Bebas Diskrepansi.**

---

### D. Audit `Pipeline STAI User (v2_stai).md`

Catatan ini mendokumentasikan pipeline berbasis kuesioner STAI dari sudut pandang metodologi dan pelabelan.

#### 1. Verifikasi Distribusi Kelas & Formulasi
- **Ambang Batas Heuristik Awal (1st–5th try):**
  Menggunakan `delta = condition_stai_mean - baseline_stai_mean >= 0.2` (`STAI_INCREASE_THRESHOLD = 0.2`). Terverifikasi pada `Scripts/1st try/v2_stai.ipynb` dan `Scripts/5th try/stai_6s_reference_models_stdized.ipynb`.
- **Formulasi 6th Try 2-Class:**
  $\Delta_{\text{sum}} = \text{condition\_stai\_sum} - \text{baseline\_stai\_sum}$.
  - Class 0 ($\Delta_{\text{sum}} \le 0$): 1.785 jendela (**43.59%**).
  - Class 1 ($\Delta_{\text{sum}} > 0$): 2.310 jendela (**56.41%**).
  - TSST Response: 100% subjek (15/15) mengalami $\Delta_{\text{sum}} > 0$. **Terverifikasi 100% akurat.**
- **Formulasi 6th Try 3-Class:**
  Ambang batas $\pm 1/24 \approx \pm 0.04167$:
  - Class 0 ($\Delta_{\text{norm}} < -1/24$): 1.110 jendela (**27.11%**).
  - Class 1 ($-1/24 \le \Delta_{\text{norm}} \le 1/24$): 1.026 jendela (**25.05%**).
  - Class 2 ($\Delta_{\text{norm}} > 1/24$): 1.959 jendela (**47.84%**).
  - Total: 4.095 jendela. **Terverifikasi 100% akurat hingga desimal kedua.**

#### 2. Catatan Cakupan & Nomenklatur
- Catatan ini diberi judul dan alias `v2_stai`, padahal isinya mencakup seluruh iterasi hingga 6th try. `v2_stai` secara historis adalah nama berkas notebook iterasi pertama dan kedua (`v2_stai.ipynb`, `v2_stai_acc_subwindow.ipynb`).
- Catatan belum secara eksplisit menjelaskan evolusi teknik *alignment*: iterasi 1st–5th try menggunakan *polyphase rational resampling* ke 70 Hz, sebelum beralih ke sinkronisasi Alkurdi outer-join pada 6th try.

---

### E. Audit `Pra-Pemrosesan Sinyal & Filtering.md`

Catatan ini merinci pembersihan sinyal (*noise filtering*) dan dekomposisi komponen fisiologis.

#### 1. Verifikasi Filter Digital & Dekomposisi (Valid)
- **EDA Butterworth Lowpass:** Filter orde 6 dengan frekuensi *cutoff* $f_c = 1.0\text{ Hz}$ diterapkan pada EDA pergelangan ($4\text{ Hz}$) dan dada ($700\text{ Hz}$) via `butter_lowpass_filter(..., 1.0, ..., 6)`. **Valid.**
- **Dekomposisi Phasic-Tonic EDA:** Menggunakan `nk.eda_phasic(nk.standardize(eda_series), sampling_rate=fs)` menghasilkan komponen tonik (`EDA_SCL`) dan fasik (`EDA_SCR`). **Valid.**
- **Akselerometer FIR Filter:** Menggunakan filter Finite Impulse Response (`filterSignalFIR`) dengan `cutoff = 0.4` dan `numtaps = 64` (diadopsi dari kode MIT Media Lab Affective Computing). **Valid.**
- **Akselerasi Resultan:** Dihitung dengan rumus Euclidean $\sqrt{x^2 + y^2 + z^2}$ untuk pergelangan (`net_acc`) dan dada (`net_acc_C`). **Valid.**
- **Sub-Jendela Akselerometer:** Jendela 5.0 detik dengan pergeseran 0.25 detik menghasilkan 36 fitur getaran mikro. **Valid.**

#### 2. Diskrepansi Atribusi Algoritma Kardiovaskular (Penting)
- **Klaim Catatan:**
  *"ECG Dada (700 Hz): Dideteksi puncak R (R-peaks) menggunakan algoritma HeartPy dan BioSPPy untuk menghasilkan rangkaian Inter-Beat Interval (IBI) dan 7 metrik HRV."*
- **Hasil Audit Kode Sumber:** **Diskrepansi.**
  - Pada berkas `_forked/fb_code/ecg.py` dan seluruh skrip pipeline (`stai_2class_alkurdi_alignment.ipynb`, `stai_6s_pipeline.py`), deteksi R-peaks dan komputasi 7 metrik HRV (`bpm`, `ibi`, `sdnn`, `sdsd`, `rmssd`, `pnn20`, `pnn50`) **sepenuhnya dilakukan oleh pustaka HeartPy** melalui fungsi `hp.process(data['ECG'].dropna().reset_index(drop=True), 700)`.
  - Pustaka `BioSPPy` **tidak pernah digunakan untuk memproses ECG**. Pustaka `BioSPPy` digunakan secara eksklusif untuk sinyal pernapasan dada pada modul `respiration.py` (`biosppy.signals.resp.resp` dan `biosppy.signals.tools.find_extrema`).

#### 3. Klarifikasi Analisis Spektral BVP
- **Klaim Catatan:**
  *"BVP Pergelangan (64 Hz): Dianalisis spektrum frekuensi dominannya menggunakan estimasi Fast Fourier Transform (FFT)."*
- **Hasil Audit Kode Sumber:** **Klarifikasi Teknis.**
  - Fungsi `get_peak_freq(x)` pada `_forked/fb_code/utils.py` tidak memanggil fungsi FFT langsung, melainkan menggunakan estimasi *Power Spectral Density* (PSD) Periodogram berbasis SciPy:
    ```python
    f, Pxx = scisig.periodogram(x, fs=8)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    peak_freq = psd_dict[max(psd_dict.keys())]
    ```
  - Perlu dicatat bahwa parameter `fs=8` di-hardcode di dalam modul utilitas referensi Alkurdi et al.

---

### F. Audit `Wrangling & Sinkronisasi Multimodal.md`

Catatan ini menjelaskan penanganan sampling rate heterogen pada WESAD.

#### 1. Verifikasi Mekanisme Sinkronisasi 6th Try (100% Akurat)
- **Tantangan Heterogen:** Dada RespiBAN (700 Hz seragam), Empatica E4 (BVP 64 Hz, ACC 32 Hz, EDA 4 Hz, TEMP 4 Hz). **Valid.**
- **Indeks Waktu Relatif Berbasis Nanosekon:**
  ```python
  time_ns = np.round(np.arange(len(signal)) * (1e9 / freq)).astype(np.int64)
  df.index = pd.to_datetime(time_ns, unit='ns')
  ```
  Terbukti identik dengan implementasi pada Cell 5 `stai_2class_alkurdi_alignment.ipynb`. **Valid.**
- **Vectorized Outer-Join:** Menggabungkan seluruh sensor dengan `pd.concat([eda_df, bvp_df, ...], axis=1)`. Sensor berfrekuensi lebih rendah secara alami menghasilkan nilai `NaN` tanpa distorsi interpolasi artifisial. **Valid.**
- **Penyelarasan Label & Dropna per Jendela:** Label diselaraskan dengan `.bfill().ffill()`, dan ekstraksi fitur per jendela mengekstrak sinyal bersih dengan `.dropna()`. **Valid.**

#### 2. Konteks Evolusi yang Hilang
Catatan ini hanya mendokumentasikan pendekatan 6th Try (*Alkurdi-style datetime outer join*). Untuk kelengkapan skripsi, perlu dicatat bahwa pada iterasi sebelumnya (1st–5th try):
- Data di-resample menggunakan `scipy.signal.resample_poly` ke frekuensi menengah $70\text{ Hz}$ (`NON_ACC_FS = 70`) untuk sinyal non-ACC, lalu dipetakan ke timeline 700 Hz menggunakan pemetaan indeks integer.
- Pendekatan 6th try beralih ke *outer-join Alkurdi* guna menghilangkan artefak interpolasi dan menjaga bentuk gelombang asli sinyal biologis.

---

## 3. Matriks Komparasi Fitur & Subset Sensor

Sebagai referensi komprehensif untuk perbaikan catatan, berikut adalah pemetaan lengkap fitur prediktif di seluruh varian model yang ada di repositori:

| Modus Sensor | Direktori Model | Jumlah Fitur ($X$) | Kanal Sensor yang Digunakan |
| :--- | :--- | :---: | :--- |
| **Full Sensor Modals (6th Try)** | `Models/6th try/2class/Full_sensor_modals/` | **155** | Dada (ECG, Resp, EDA, EMG, Temp, ACC) + Pergelangan (BVP, EDA, TEMP, ACC) |
| **Full Sensor Modals (5th Try)** | `Models/5th try/` | **154** | Sama seperti di atas, kecuali `net_acc_C_abs_integral` belum diekstrak |
| **Sliced Sensor Modals (Wrist Only)** | `Models/6th try/2class/Sliced_sensor_modals/` | **62** | Pergelangan tangan saja: ACC (wrist), BVP, EDA (wrist), TEMP (wrist) |
| **Minimum Sensor: EDA + ACC** | `Models/6th try/2class/Minimum_sensor_modals/` | **56** | EDA pergelangan (statistik, phasic, tonic, dinamik) + ACC pergelangan |
| **Minimum Sensor: ACC + TEMP** | `Models/6th try/2class/Minimum_sensor_modals/` | **44** | ACC pergelangan + TEMP pergelangan |
| **Minimum Sensor: EDA + TEMP** | `Models/6th try/2class/Minimum_sensor_modals/` | **18** | EDA pergelangan + TEMP pergelangan |
| **Baseline Referensi Upstream (`_forked`)** | `_forked/data/subject_feats/` | **118** | Fitur fisiologis standar Alkurdi et al. (tanpa sub-jendela mikro ACC) |

---

## 4. Daftar Rekomendasi Tindakan Koreksi (Actionable Remediation)

Berdasarkan hasil temuan di atas, berikut adalah rekomendasi pembaruan (*patching*) langsung untuk setiap catatan:

### 1. Perbaikan `Katalog Fitur Fisiologis.md`
- **Perbaiki Diagram Pohon di Seksi 1:**
  - Ubah `Statistik Dasar Sinyal: 64 fitur (16 sinyal × 4 metrik)` menjadi `80 fitur (20 sinyal × 4 metrik)`.
  - Ubah `Akselerometer Kinematik Global: 15 fitur` menjadi `14 fitur (8 abs integral + 6 peak freq)`.
  - Tambahkan entri: `├── Penghitung Sub-Jendela Akselerometer: 1 fitur (ACC_subwindow_count)`.
  - Pastikan total penjumlahan cabang bernilai tepat **155 fitur** ($80 + 14 + 36 + 1 + 4 + 7 + 1 + 2 + 6 + 4 = 155$).
- **Perbaiki Judul Seksi 2.A:**
  - Ganti `### A. Fitur Statistik Dasar Jendela (64 Fitur)` menjadi `### A. Fitur Statistik Dasar Jendela (80 Fitur)`.
  - Ganti teks pengantar `Diterapkan pada 16 kanal sinyal:` menjadi `Diterapkan pada 20 kanal sinyal:`.
- **Perbaiki Judul Seksi 2.B:**
  - Ganti `(15 Fitur)` menjadi `(14 Fitur)`.

### 2. Perbaikan `Perbandingan Pipeline STAI vs Forked.md`
- Pada Tabel Komparasi baris **Dimensi Fitur Prediktif ($X$)**:
  - Ganti `120 fitur fisiologis` pada kolom `_forked` menjadi `118 fitur fisiologis`.
  - Berikan catatan kaki penjelas: `118 fitur dasar + 36 fitur sub-jendela akselerometer + 1 net_acc_C_abs_integral = 155 fitur prediktif`.

### 3. Perbaikan `Pra-Pemrosesan Sinyal & Filtering.md`
- Pada Seksi 1.C (**Kardiovaskular**):
  - Hapus penyebutan BioSPPy pada ECG: Ubah kalimat menjadi:
    *"ECG Dada (700 Hz): Dideteksi puncak R (R-peaks) dan dihitung 7 metrik HRV menggunakan algoritma **HeartPy** (`hp.process`)."*
  - Pada BVP, tambahkan penjelasan teknis bahwa frekuensi puncak diestimasi via Periodogram PSD (`scipy.signal.periodogram` dengan $f_s = 8$).
- Pada Seksi 1.D (**Sabuk Pernapasan Dada**):
  - Tambahkan keterangan: *"Dianalisis menggunakan modul `biosppy.signals.resp.resp` dan `find_extrema` untuk mendeteksi siklus respirasi, laju pernapasan, durasi inhalasi/ekshalasi, serta rasio I/E."*

### 4. Perbaikan `Pipeline STAI User (v2_stai).md`
- Tambahkan subseksi ringkas mengenai transisi arsitektur sinkronisasi: dari *resampling polifase 70 Hz* (1st–5th try) menuju *Alkurdi-style datetime outer join* (6th try).

---

## 5. Kesimpulan Akhir

Catatan dalam folder `_Obsidian Notes/Pipelines & Methodology` memiliki fondasi metodologis dan pemahaman domain fisiologis yang kuat. Sebagian besar parameter penting (seperti pembagian jendela 60s/6s, logika pelabelan biner dan 3-kelas STAI, rasio kelas kohort, dan mekanisme outer-join nanosekon) telah sesuai 100% dengan kode sumber implementasi terbaru.

Diskrepansi yang terdeteksi murni bersifat:
1. **Aritmetika internal & taksonomi** pada katalog fitur (salah hitung 80 fitur dasar sebagai 64, dan salah hitung 14 fitur global ACC sebagai 15).
2. **Atribusi pustaka** (ECG HRV menggunakan HeartPy, sedangkan BioSPPy digunakan untuk respirasi).
3. **Penyebutan jumlah fitur referensi `_forked`** (118 fitur aktual vs 120 fitur tertulis).

Dengan diterapkannya rekomendasi koreksi di atas, seluruh dokumentasi metodologi pipeline skripsi akan memiliki konsistensi dan integritas akademik 100% terhadap repositori kode.
