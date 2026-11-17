---
tags:
  - skripsi/script
  - pipeline/6th-try
  - metodologi
  - alkurdi-alignment
  - validation
  - feature-engineering
date: 2026-11-17
aliases:
  - Script 6th Try
  - Script 6th Try Pipeline
  - 6th Try Pipeline
  - Pipeline 6th Try
  - stai_2class_alkurdi_alignment
  - stai_3class_alkurdi_alignment
  - Script 6th Try Pipeline (Alkurdi Alignment)
---

# Script: Pipeline Generasi Ke-6 (`6th try` — Alkurdi Alignment)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Scripts Pipeline Evolution (1st to 6th Try)]], [[Forked Repository Overview]], [[Perbandingan Pipeline STAI vs Forked]], [[Katalog Fitur Fisiologis]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]], [[Pipelines & Methodology]]  
> **Berkas Sumber Utama:**  
> - `Scripts/6th try/2class/stai_2class_alkurdi_alignment.ipynb`  
> - `Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`  
> - `Scripts/Utils/relabel_and_evaluate_6th_try.py`  
> - `_forked/fb_code/feature_extraction.py` (Repositori Asal Alkurdi)  

---

## 1. Ikhtisar & Latar Belakang (Overview)

Pipeline Generasi Ke-6 (**`6th try`**) merupakan arsitektur pra-pemrosesan sinyal dan rekayasa fitur termutakhir (*state-of-the-art*) dalam repositori ini. Pipeline ini dikembangkan untuk merealisasikan dua tujuan fundamental:

1. **Penyelarasan Metodologis dengan Literatur Upstream:** Menyelaraskan pra-pemrosesan sinyal dan format sinkronisasi multisensor dengan kode disertasi referensi Alkurdi et al. (`_forked`), menggantikan pendekatan *resampling polifase 70 Hz* terdahulu (1st–5th try) dengan mekanisme *outer-join datetime*.
2. **Rektifikasi Total Pelabelan Subjektif STAI:** Mengintegrasikan audit kuesioner State-Trait Anxiety Inventory 6-butir (STAI-6) dengan *reverse-scoring* terektifikasi, menyelesaikan isu *zero-inflation* dan transposisi indeks historis sehingga **100% subjek (15/15)** terbukti mengalami peningkatan kecemasan saat sesi induksi stres TSST.

Catatan ini menguraikan secara rinci cara kerja teknis pipeline 6th try, menganalisis klaim-klaim di dalam notebook (`.ipynb`), melakukan validasi silang terhadap repositori referensi Alkurdi (`_forked`), serta merumuskan standar ideal implementasi (*how it should have been*).

---

## 2. Cara Kerja Pipeline 6th Try (End-to-End Architecture)

Arsitektur pemrosesan data pada pipeline 6th try terbagi ke dalam 6 tahapan modular:

```
Raw WESAD Data (S{id}.pkl & S{id}_quest.csv)
  │
  ├── 1. Parser STAI-6 & Perumusan Ground Truth (Reverse Scoring Butir 1, 4, 6)
  │
  ├── 2. Pra-Pemrosesan Sinyal Fisiologis
  │     ├── EDA: Butterworth Lowpass (1.0 Hz, Order 6) + NeuroKit2 Phasic/Tonic
  │     ├── ACC: FIR Lowpass Filter (Cutoff 0.4, Numtaps 64) + Net Acceleration
  │     └── ECG, BVP, Resp, EMG, Temp
  │
  ├── 3. Sinkronisasi Jam Sensor Heterogen (Alkurdi Datetime Outer Join)
  │     └── Indeks Nanosekon -> pd.concat(axis=1) -> bfill()/ffill() pada Label
  │
  ├── 4. Segmentasi Jendela Geser & Strict Temporal Enclosure
  │     └── Jendela 60s, Stride 6s (Overlap 90%), Buang Jendela Perbatasan
  │
  ├── 5. Ekstraksi 155 Fitur Prediktif
  │     ├── 80 Statistik Dasar (20 kanal × 4 metrik)
  │     ├── 36 Sub-Jendela Akselerometer (5.0s / 0.25s)
  │     ├── 14 Kinematik Global ACC + 1 ACC_subwindow_count
  │     └── 24 Dinamik Fisiologis (ECG HRV, Resp, EDA, BVP, EMG, Temp)
  │
  └── 6. Evaluasi Pembelajaran Mesin Multi-Skema
        ├── Scheme A: Naive Random Split (80/20) -> Deteksi Kebocoran Data (Leakage)
        └── Scheme B: 15-Fold Leave-One-Subject-Out (LOSO) + In-Fold StandardScaler
```

---

### Tahap 1: Parser Kuesioner STAI-6 & Perumusan Ground Truth

1. **Pembacaan Berkas Kuesioner:** Membaca baris `# ORDER`, `# START`, `# END`, dan `# STAI` dari `WESAD/S{id}/S{id}_quest.csv`.
2. **Koreksi Transposisi Indeks:** Memperbaiki bug historis pemetaan kolom:
   ```python
   # Kode terektifikasi 6th try:
   responses = [float(v) for v in stai_rows[index]]
   ```
3. **Penerapan Reverse-Scoring STAI-6:**
   Butir 1 (*feel at ease*), 4 (*relaxed*), dan 6 (*pleasant*) dibalik nilainya menggunakan formula $5 - \text{raw}$ (skala Likert 1–4):
   $$\text{Skor Item Invers} = 5 - X_{\text{raw}}, \quad \text{untuk butir } \{1, 4, 6\}$$
4. **Perumusan Label Target:**
   - Kondisi `Base` dijadikan nilai acuan personal ($\text{Base}_{\text{sum}}$).
   - Selisih skor kondisi terhadap baseline:
     $$\Delta_{\text{sum}} = \text{condition\_stai\_sum} - \text{baseline\_stai\_sum}$$
   - **Formulasi Biner (2-Class):**
     $$\text{Label} = \begin{cases} 1 \ (\text{Anxiety Increased}), & \Delta_{\text{sum}} > 0 \\ 0 \ (\text{No Anxiety Increase}), & \Delta_{\text{sum}} \le 0 \end{cases}$$
   - **Formulasi Tripartit (3-Class):** Menggunakan ambang batas resolusi instrumen minimum ($\pm 1/24 \approx \pm 0{,}04167$):
     $$\text{Label} = \begin{cases} 2 \ (\text{Anxiety Increased}), & \Delta_{\text{norm}} > 1/24 \\ 1 \ (\text{Neutral / Potential}), & -1/24 \le \Delta_{\text{norm}} \le 1/24 \\ 0 \ (\text{No Anxiety Increase}), & \Delta_{\text{norm}} < -1/24 \end{cases}$$

---

### Tahap 2: Pra-Pemrosesan Sinyal Fisiologis

Sebelum penggabungan multisensor, setiap kanal sinyal mentah dibersihkan secara terisolasi pada laju sampling aslinya (*native sampling rate*):

1. **Electrodermal Activity (EDA):**
   - Disaring menggunakan Butterworth Lowpass Filter orde 6 dengan frekuensi cut-off $f_c = 1.0\text{ Hz}$ untuk pergelangan ($4\text{ Hz}$) dan dada ($700\text{ Hz}$).
   - Didekomposisi menggunakan NeuroKit2 (`nk.eda_phasic`) setelah standardisasi, memisahkan sinyal menjadi komponen tonik lambat (`EDA_SCL`) dan respons fasik transien (`EDA_SCR`).
2. **Akselerometer (ACC):**
   - Disaring menggunakan filter Finite Impulse Response (FIR) lowpass 64-tap dengan cut-off $0.4\text{ Hz}$ (`utils.filterSignalFIR`).
   - Dihitung vektor akselerasi resultan (*net acceleration*):
     $$\text{net\_acc} = \sqrt{\text{ACC}_x^2 + \text{ACC}_y^2 + \text{ACC}_z^2}$$
3. **Optimasi Memori:** Seluruh array sensor dikonversi ke tipe data `float32` untuk mencegah kehabisan RAM (*out-of-memory*) saat penggabungan matriks temporal resolusi tinggi.

---

### Tahap 3: Sinkronisasi Jam Sensor Heterogen (Alkurdi Datetime Outer Join)

Menggantikan metode *resample_poly 70 Hz* lama dengan arsitektur sinkronisasi Alkurdi:

1. **Indeks Waktu Berpresisi Nanosekon:**
   Setiap sampel sensor diberi stempel waktu relatif terhadap awal perekaman berbasis nanosekon:
   ```python
   for df_obj, freq in sensor_freq_pairs:
       ns_int = np.round(np.arange(len(df_obj)) * (1e9 / freq)).astype('int64')
       df_obj.index = pd.to_datetime(ns_int, unit='ns')
   ```
2. **Vectorized Outer-Join:**
   Seluruh DataFrame digabungkan menggunakan penggabungan luar bervektor:
   ```python
   df = pd.concat([
       eda_df, bvp_df, acc_df, temp_df,
       acc_c_df, ecg_df, eda_c_df, emg_c_df, resp_c_df, temp_c_df,
       label_df
   ], axis=1)
   ```
   Resolusi tertinggi ($700\text{ Hz}$) membentuk basis kisi waktu, sedangkan sensor berfrekuensi lebih rendah menghasilkan nilai `NaN` secara alami di antara titik-titik sampelnya.
3. **Penyelarasan Label:**
   Kolom label diekspansi ke seluruh baris kisi waktu menggunakan perpaduan *backward-fill* dan *forward-fill*:
   ```python
   df['protocol_label'] = df['protocol_label'].bfill().ffill()
   ```

---

### Tahap 4: Segmentasi Jendela Geser & Strict Temporal Enclosure

1. **Parameter Jendela:**
   - Durasi jendela: **60 detik** ($42.000$ titik sampel pada $700\text{ Hz}$).
   - Pergeseran (*stride*): **6 detik** (overlap temporal 90%).
2. **Strict Temporal Enclosure:**
   Jendela hanya diproses apabila **100% rentang waktunya** ($[t_{\text{start}}, t_{\text{end}}]$) berada utuh di dalam rentang waktu kondisi protokol yang memiliki skor STAI valid. Jendela yang melintasi batas pergantian kondisi atau berada di luar sesi kuesioner langsung dibuang (*dropped*).
3. **Isolasi Sinyal Bersih (`dropna`):**
   Saat mengekstrak fitur pada jendela 60 detik, setiap fungsi ekstraksi mengekstrak sinyal sensor menggunakan metode `.dropna()`. Ini memungkinkan komputasi FFT, HRV, dan statistik berjalan pada laju sampling aslinya tanpa distorsi interpolasi.

---

### Tahap 5: Ekstraksi 155 Fitur Prediktif Multimodal

Setiap jendela 60 detik menghasilkan matriks fitur berdimensi $1 \times 155$:

1. **Statistik Dasar Jendela (80 Fitur):**
   $$\text{mean}, \quad \text{std}, \quad \text{min}, \quad \text{max}$$
   Diterapkan pada **20 kanal sinyal**: `EDA`, `EDA_SCR`, `EDA_SCL`, `BVP`, `ACC_x`, `ACC_y`, `ACC_z`, `TEMP`, `net_acc`, `ECG`, `EDA_C`, `EDA_SCR_C`, `EDA_SCL_C`, `ACC_x_C`, `ACC_y_C`, `ACC_z_C`, `EMG_C`, `Resp_C`, `Temp_C`, `net_acc_C` ($20 \times 4 = 80$).
2. **Sub-Jendela Akselerometer Mikro (36 Fitur):**
   Jendela 60s dipecah menjadi sub-jendela pendek berdurasi **5.0 detik** dengan pergeseran **0.25 detik**. Pada 6 sumbu ACC (`ACC_x, y, z` wrist dan chest), dihitung `std`, `abs_integral`, dan `peak_freq`, lalu diagregasikan menjadi nilai `mean` dan `std` per jendela ($6 \times 3 \times 2 = 36$).
3. **Kinematik Global Akselerometer (14 Fitur):**
   - 8 fitur integral nilai absolut: `net_acc`, `net_acc_C`, dan 6 sumbu ACC.
   - 6 fitur frekuensi puncak FFT/PSD: 6 sumbu ACC.
4. **Penghitung Sub-Jendela (`ACC_subwindow_count`) (1 Fitur):**
   Jumlah sub-jendela valid yang terbentuk di dalam jendela utama.
5. **Dinamika Kardiovaskular & Otonom (24 Fitur):**
   - **ECG HRV (HeartPy) (7 fitur):** `bpm`, `ibi`, `sdnn`, `sdsd`, `rmssd`, `pnn20`, `pnn50`.
   - **BVP Pergelangan (1 fitur):** `BVP_peak_freq`.
   - **EDA Dinamik (4 fitur):** `EDA_slope`, `EDA_C_slope`, `EDA_drange`, `EDA_C_drange`.
   - **EMG Dada (2 fitur):** `EMG_drange`, `EMG_abs_integral`.
   - **Respirasi Dada (BioSPPy) (6 fitur):** `rate`, `Inhal_mean`, `Inhal_std`, `Exhal_mean`, `Exhal_std`, rasio $I/E$.
   - **Temperatur Dinamik (4 fitur):** `drange` dan `slope` pada pergelangan dan dada.

---

### Tahap 6: Evaluasi Pembelajaran Mesin Multi-Skema

Pipeline mengevaluasi performa model menggunakan dua skema pembagian data:

1. **Validation Scheme A (Naive Random Split 80/20):**
   - Meniru metodologi pelaporan Alkurdi et al. (`train_test_split(test_size=0.2, random_state=0)`).
   - Menghasilkan akurasi semu sangat tinggi (**93%–100%**) akibat kebocoran data temporal (*temporal proximity data leakage*) di antara jendela yang overlap 90%.
2. **Validation Scheme B (15-Fold Leave-One-Subject-Out / LOSO):**
   - Pada setiap fold, model dilatih pada 14 subjek dan diuji pada 1 subjek yang sama sekali belum pernah dilihat (*unseen subject*).
   - **In-Fold Scaling:** `StandardScaler` di-fit **hanya** pada 14 subjek data latih fold tersebut, lalu digunakan untuk mentransformasikan data uji subjek held-out, menjamin nol kebocoran data.

---

## 3. Validasi Komparatif: 6th Try vs. Repositori Referensi Alkurdi (`_forked`)

Meskipun notebook 6th try menyatakan bahwa pipeline ini *“diadaptasi dari arsitektur Alkurdi et al.”*, audit mendalam terhadap kode sumber `_forked/` mengungkap sejumlah kesamaan, optimasi modern, divergensi metodologis sengaja, dan keanehan (*quirks*) yang terwarisi:

| Komponen Metodologi                     | Upstream Alkurdi (`_forked/fb_code`)                                                                                      | Implementasi Aktual 6th Try                                                                                                                     | Analisis Validitas & Implikasi                                                                                                                                                              |
| :-------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Ground Truth (Target)**               | **Kondisi Protokol Objektif Lab:**<br>Baseline (1), TSST (2), Amusement (0). Format One-Hot `['0', '1', '2']`.            | **Respons Subjektif Kuesioner STAI:**<br>$\Delta_{\text{sum}} > 0$ (2-class) atau $\pm 1/24$ (3-class). Format numerik tunggal `anxiety_label`. | **Divergensi Sengaja.** Mengubah esensi deteksi dari *stimulus recognition* menjadi *subjective anxiety detection*. Metrik kedua pipeline tidak dapat dibandingkan nominal secara langsung. |
| **Penyelarasan Jam (*Outer Join*)**     | List comprehension detik float `[(1/fs)*i]` dikonversi via `pd_old.to_datetime(unit='s')`, lalu di-join berantai 10 kali. | Vektorisasi nanosekon integer 64-bit `np.round(np.arange * 1e9 / freq).astype('int64')`, lalu `pd.concat(..., axis=1)`.                         | **Optimasi Modern.** 6th try jauh lebih cepat, bebas *floating point rounding drift*, dan menggunakan sintaks Pandas modern.                                                                |
| **Pengisian Label (*Label Fill*)**      | `df['label'].fillna(method='bfill')`                                                                                      | `df['protocol_label'].bfill().ffill()`                                                                                                          | **Perbaikan Bug.** Alkurdi meninggalkan nilai `NaN` di akhir data jika array sensor lebih panjang dari label. 6th try menangani kedua ujung timeline secara bersih.                         |
| **Segmentasi Jendela**                  | **Tumbling Window (Non-overlapping):**<br>Panjang 60s, Stride 60s.<br>Total jendela kohort: **577 jendela**.              | **Sliding Window (Overlap 90%):**<br>Panjang 60s, Stride 6s.<br>Total jendela kohort: **4.095 jendela**.                                        | **Divergensi Metodologis.** Menghasilkan data 7× lebih padat untuk merefleksikan dinamika transisi stres, tetapi menuntut evaluasi berbasis subjek (LOSO).                                  |
| **Penyaringan Waktu Batas**             | `df.groupby('label')` lalu memotong jendela pada setiap grup secara terpisah.                                             | **Strict Temporal Enclosure:** Jendela wajib 100% berada dalam rentang waktu kondisi bertarget yang valid.                                      | **Keunggulan 6th Try.** `groupby` Alkurdi memotong segmen tanpa memperhatikan integritas waktu kronologis kuesioner. 6th try menjamin validitas kuesioner.                                  |
| **Dimensi Fitur ($X$)**                 | **118 Fitur Fisiologis** (tertulis salah sebagai 120 pada komentar repositori).                                           | **155 Fitur Prediktif** ($118 \text{ dasar} + 36 \text{ sub-ACC} + 1 \ \text{subwindow\_count}$).                                               | **Ekspansi Fitur.** Penambahan 36 fitur getaran mikro kinematik akselerometer terbukti meningkatkan stabilitas deteksi gerak.                                                               |
| **Estimasi Frekuensi Puncak BVP & ACC** | `f, Pxx = scisig.periodogram(x, fs=8)` dengan hardcoded `fs=8`.                                                           | **Rektifikasi Penuh (2026-11-17):**<br>Melokalisasi `fb_code` ke `Scripts/6th try/` dan mengoper parameter frekuensi aktual (`RAW_FS['BVP']=64`, `RAW_FS['ACC']=32`, `FS['chest']=700`). | **Terektifikasi.** Sebelumnya terdistorsi oleh faktor skala konstan ($0.125\times$ BVP, $0.25\times$ ACC). Kini mencerminkan satuan Hertz fisik murni dengan sedikit sensitivitas akurasi model ($\pm 0.5\%\text{--}1.8\%$). |
| **Skema Evaluasi Model**                | **Hanya Naive Random Split (80/20):**<br>`train_test_split(0.2, random_state=0)`. Tidak ada validasi LOSO sama sekali.    | **Dual Scheme:**<br>1. Scheme A (Naive Random Split 80/20)<br>2. Scheme B (15-Fold LOSO + In-Fold Scaling).                                     | **Keunggulan Ilmiah 6th Try.** 6th try membuktikan adanya *data leakage* fatal pada Scheme A Alkurdi dan menetapkan tolak ukur LOSO sejati.                                                 |
| **Spesifikasi Model ML**                | LDA, RF (`max_depth=4`), Linear SVM ($C=1.0, 0.9$).                                                                       | Menguji seluruh model Alkurdi, ditambah RF (`max_depth=25`), XGBoost, SVM RBF, KNN.                                                             | **Ekspansi Arsitektur.** Mengungkap bahwa `max_depth=4` Alkurdi mengalami *underfitting* parah pada fitur 155 dimensi.                                                                      |

---

## 4. Validasi Klaim Internal Notebook vs. Artefak Kode

Audit forensik terhadap isi teks Markdown dan sel eksekusi kode pada notebook `Scripts/6th try/` menemukan sejumlah temuan krusial:

### A. Validasi Klaim 2-Class (`stai_2class_alkurdi_alignment.ipynb`)
1. **Klaim TSST Peningkatan 100%:** **VALID.** Seluruh 15 subjek WESAD terbukti memiliki $\Delta_{\text{sum}} > 0$ saat TSST setelah indeks kuesioner diperbaiki.
2. **Klaim Keseimbangan Kelas:** **VALID.** Distribusi kelas biner pada 4.095 jendela kohort adalah 1.785 jendela Class 0 (**43.59%**) dan 2.310 jendela Class 1 (**56.41%**).
3. **Klaim Akurasi LOSO Scheme B:** **VALID.** Random Forest (`max_depth=4`) mencapai **74.43% pooled accuracy** dan **0.7587 F1-score kecemasan**, menjadikannya model benchmark terbaik.

---

### B. Temuan Diskrepansi Kritis: Klaim Markdown vs. Output Aktual pada 3-Class Notebook

Pada notebook `Scripts/6th try/3class/stai_3class_alkurdi_alignment.ipynb`, terdeteksi adanya **diskrepansi internal antara teks narasi Markdown di Cell 16 dengan hasil eksekusi kode aktual di Cell 13**:

```
Kontradiksi Internal pada Notebook 3-Class:

Teks Narasi Cell 16 (TIDAK AKURAT / OUTDATED):
├── Class 0 (No Anxiety Increase):        2.512 jendela (60.57%)
├── Class 1 (Potential Anxiety Increase):   832 jendela (20.06%)
└── Class 2 (Anxiety Increased):            803 jendela (19.36%)

Hasil Eksekusi Sel 13 & File Data Aktual (VALID & TEREKTIFIKASI):
├── Class 0 (No Anxiety Increase):        1.110 jendela (27.11%)
├── Class 1 (Potential Anxiety Increase): 1.026 jendela (25.05%)
└── Class 2 (Anxiety Increased):          1.959 jendela (47.84%)
```

> [!WARNING] Asal-Usul Diskrepansi Teks Cell 16
> Angka $2.512$ / $832$ / $803$ pada Cell 16 merupakan sisa dokumentasi teks lama ketika kuesioner masih mengalami *indexing transposition bug* (sebelum reverse-scoring dan pembacaan 6 butir diperbaiki). Saat Cell 13 dieksekusi ulang dengan parser kuesioner yang benar, distribusinya terektifikasi menjadi **1.110 / 1.026 / 1.959**. Teks Markdown Cell 16 lupa diperbarui oleh penulis notebook, meskipun catatan di vault (`Model Output Specs 6th Try 3-Class.md`) telah mencatat angka yang benar.

---

### C. Validasi Dimensi Fitur ($X = 155$ vs $154$)
- Notebook menyatakan ekstraksi menghasilkan 154–155 fitur.
- Pemeriksaan kolom membuktikan bahwa kolom `ACC_subwindow_count` tidak dimasukkan ke dalam `metadata_columns`, sehingga ikut terpilih ke dalam matriks fitur prediktif $X$.
- Selain itu, penambahan fitur `net_acc_C_abs_integral` (yang tidak ada pada 5th try) menggenapkan total dimensi fitur prediktif menjadi tepat **155 kolom**.

---

## 5. Rekomendasi Standardisasi ("How It Should Have Been")

Berdasarkan hasil komparasi terhadap `_forked` dan audit kelemahan internal notebook 6th try, berikut adalah panduan perbaikan ideal yang direkomendasikan untuk repositori dan penulisan skripsi:

### 1. Perbaikan Fungsi Spektral `get_peak_freq` (Terektifikasi per 17 November 2026)
Fungsi `get_peak_freq` pada `_forked/fb_code/utils.py` mulanya menggunakan parameter frekuensi konstan `fs=8`:
```python
# Implementasi Alkurdi yang cacat:
def get_peak_freq(x):
    f, Pxx = scisig.periodogram(x, fs=8)  # Salah: fs di-hardcode ke 8 Hz
    ...
```
*Standar Terektifikasi (Telah Diimplementasikan):*  
Helper module dilokalisasi ke `Scripts/6th try/fb_code/utils.py` dan disesuaikan agar menerima parameter $f_s$ aktual:
```python
# Implementasi terektifikasi:
def get_peak_freq(x, fs=8):
    f, Pxx = scisig.periodogram(x, fs=fs)
    psd_dict = {amp: freq for amp, freq in zip(Pxx, f)}
    if not psd_dict:
        return np.nan
    return psd_dict[max(psd_dict.keys())]
```
Hal ini krusial agar spektrum puncak denyut BVP ($64\text{ Hz}$) dan akselerasi dada ($700\text{ Hz}$) mencerminkan satuan Hertz fisik yang sesungguhnya.  
> [!NOTE] Dampak Sensitivitas Akurasi Model
> Perubahan skala frekuensi fisik ini berdampak pada 19 fitur (pada model full) dan 7 fitur (pada model wrist ACC), yang menyebabkan fluktuasi minor pada metrik akurasi evaluasi model ($\approx \pm 0.5\%\text{--}1.8\%$). Laporan lengkap didokumentasikan pada [[2026-11-17 Research Note]].

### 2. Sanitasi Kolom `ACC_subwindow_count` dari Matriks Fitur (Terektifikasi per 17 November 2026)
Kolom `ACC_subwindow_count` pada dasarnya adalah variabel pelacak operasional (bernilai ~221 sub-jendela per jendela 60s) dan bukan biomarker fisiologis tubuh.
- **Status:** **Terektifikasi.** Kolom `'ACC_subwindow_count'` telah resmi dimasukkan ke dalam daftar `metadata_columns` pada seluruh notebook (`stai_2class_alkurdi_alignment.ipynb`, `stai_3class_alkurdi_alignment.ipynb`, `minimum_sensor_*.ipynb`, `sliced_sensor.ipynb`, dan skrip analisis).
- **Dimensi Fitur Terektifikasi:**
  - Pipeline Penuh (2-Class & 3-Class): Tepat **154 fitur prediktif murni** ($155 - 1$).
  - Pipeline Sliced Sensor (Smartwatch Pergelangan): Tepat **61 fitur prediktif murni** ($62 - 1$).
  - Pipeline Minimum Sensor (ACC+TEMP & EDA+ACC): Tepat **55 fitur prediktif murni** ($56 - 1$).

### 3. Pembaruan Teks Narasi Cell 16 Notebook 3-Class (Terektifikasi per 17 November 2026)
Diskrepansi internal antara teks narasi Markdown Cell 16 dengan eksekusi kode aktual Cell 13 pada `stai_3class_alkurdi_alignment.ipynb` telah diselaraskan sepenuhnya:
- Distribusi kelas usang $2.512$ ($60.57\%$) / $832$ ($20.06\%$) / $803$ ($19.36\%$) telah diarsipkan di [[2026-11-17 Research Note]].
- Teks Cell 16 kini selaras dengan ground-truth terektifikasi:
  - **Class 0 (No Anxiety Increase):** **$1.110$ jendela ($27.11\%$)**
  - **Class 1 (Potential Anxiety Increase):** **$1.026$ jendela ($25.05\%$)**
  - **Class 2 (Anxiety Increased):** **$1.959$ jendela ($47.84\%$)**
  - Total kohort: **$4.095$ jendela** ($100\%$ subjek / $15$ dari $15$ mengalami lonjakan kecemasan Class 2 saat TSST).
- Metrik LOSO diperbarui: **SVM RBF** mencapai akurasi pooled tertinggi (**$57.22\%$**, Macro F1: $0.4976$, Class 2 F1: $0.7965$) dan **SVM Linear** mencapai balanced accuracy tertinggi (**$51.01\%$**).

### 4. Penebalan Argumen Metodologis pada Naskah Skripsi

Naskah Skripsi (khususnya Bab 3 Metodologi dan Bab 4 Hasil & Pembahasan) harus secara kritis dan mendalam membedah komparasi antara **Validation Scheme A (Naive Random Split 80/20)** dan **Validation Scheme B (15-Fold Leave-One-Subject-Out CV)** melalui tiga pilar argumen ilmiah:

#### A. Ketidakcocokan Mutlak Metode Alkurdi pada Pipeline 6th Try
Pipeline 6th try mengadopsi segmentasi *sliding window* berdurasi 60 detik dengan pergeseran 6 detik (overlap temporal sebesar $90\%$). Apabila skema pembagian acak naive milik Alkurdi (`train_test_split(test_size=0.2, random_state=0)`) diterapkan pada data ber-overlap ini:
1. **Kebocoran Autokorelasi Temporal (*Cross-Window Temporal Leakage*):** Dua jendela yang berdampingan ($[t, t+60\text{s}]$ dan $[t+6\text{s}, t+66\text{s}]$) berbagi $54\text{ detik}$ ($90\%$) sampel data mentah yang persis sama. Memasukkan satu jendela ke data latih dan jendela di sebelahnya ke data uji berarti menguji model pada data yang sudah dihafalnya sebesar $90\%$.
2. **Hasil Akurasi Palsu (*Artificial Metric Inflation*):** Akurasi Scheme A yang mencapai **$93\%\text{--}100\%$** bukanlah bukti kecerdasan model, melainkan ilusi kebocoran data temporal yang secara ilmiah tidak valid.

#### B. Kritik Kritis terhadap Metodologi Evaluasi pada Penelitian Alkurdi et al. Sendiri
Argumen skripsi dapat melangkah lebih jauh dengan mengkritisi metodologi evaluasi internal pada riset Alkurdi et al. (`_forked`) itu sendiri:
1. **Meskipun Non-Overlapping, Pembagian Tetap Berbasis Sampel Global (Bukan Berbasis Subjek):** Alkurdi et al. memang menggunakan *tumbling window* (panjang 60s, stride 60s tanpa overlap). Namun, mereka menerapkan `train_test_split(0.2)` **secara acak di seluruh kumpulan jendela gabungan dari 15 subjek**.
2. **Dominasi Variabilitas Antar-Individu (*Inter-Subject Variability*):** Dalam fisiologi manusia, perbedaan sinyal dasar antar-orang (seperti *baseline* tonik EDA, rentang detak jantung istirahat, atau konduktansi kulit individu) jauh lebih besar daripada perbedaan sinyal akibat stres pada satu orang (*intra-subject variability*).
3. **Model Diuntungkan oleh "Sidik Jari Fisiologis" (*Subject Fingerprinting*):** Karena jendela-jendela dari Subjek $X$ tersebar ke dalam data latih dan data uji sekaligus, jendela uji memiliki karakteristik fisiologis yang identik dengan jendela latih milik subjek yang sama. Hal ini **sangat mempermudah model untuk memprediksi**: model tidak benar-benar mempelajari dinamika transisi stres/kecemasan universal, melainkan hanya mengenali *"ini adalah pola fisiologis Subjek 10 yang sedang berada pada label kondisi tertentu"*.
4. **Ketidakmampuan Bertahan pada Pengguna Baru (*Zero Ambulatory Generalization*):** Akurasi $>90\%$ yang dilaporkan oleh Alkurdi et al. pada dasarnya adalah akurasi yang *optimistically biased*. Ketika sistem tersebut diuji pada pengguna baru di dunia nyata (*unseen subjects*), model akan gagal total karena pengguna baru membawa "sidik jari fisiologis" yang belum pernah ada di data latih.

#### C. Penegasan Scheme B (15-Fold LOSO) sebagai Standar Emas Pengujian Wearable
1. **Isolasi Total Subjek:** Setiap fold melatih model pada 14 subjek dan mengujinya pada 1 subjek yang sama sekali asing, dengan `StandardScaler` di-fit ketat hanya pada 14 subjek latih (*in-fold scaling*).
2. **Tolak Ukur Ilmiah yang Jujur dan Kompetitif:** Performa Random Forest **$74.43\%$** (pada klasifikasi biner 2-class) dan SVM RBF **$57.22\%$** (pada 3-class) merupakan capaian representasi fisiologis sejati yang objektif, kokoh, dan mencerminkan kesiapan implementasi nyata (*real-world deployment*) pada perangkat *smartwatch*.

---

## 6. Ringkasan Kesimpulan

Pipeline Generasi Ke-6 (`6th try`) adalah iterasi paling kokoh, matang, dan valid secara ilmiah dalam repositori ini:
- Berhasil mengadaptasi arsitektur pra-pemrosesan sinyal Alkurdi et al. dengan implementasi kode modern dan efisien.
- Berhasil merevolusi tugas klasifikasi dari ketergantungan protokol lab menjadi deteksi kecemasan subjektif berbasis kuesioner tervalidasi psikometrik (STAI-6).
- Mampu membuktikan secara empiris perbedaan drastis antara model yang mengalami kebocoran data (*Scheme A*) dengan model yang siap dideploy pada subjek baru secara independen (*Scheme B*).

---

## 7. Tautan Terkait
- Laporan validasi metodologi: [[Pipelines & Methodology]]
- Laporan validasi model: [[Model Specs]]
- Spesifikasi model biner: [[Model Output Specs 6th Try 2-Class]]
- Spesifikasi model 3-kelas: [[Model Output Specs 6th Try 3-Class]]
- Repositori referensi: [[Forked Repository Overview]]
- Modul pembantu: [[Modul Library fb_code]]
- Kronologi seluruh skrip: [[Scripts Pipeline Evolution (1st to 6th Try)]]
- Peta konsep utama: [[MOC - Skripsi Repository]]
