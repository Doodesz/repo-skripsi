---
tags:
  - skripsi/scripts
  - pipeline/evolution
  - git-history
  - machine-learning
aliases:
  - Scripts Pipeline Evolution
  - Scripts Overview
  - Pipeline Evolution (1st to 6th Try)
---
# Evolusi Pipeline & Direktori Skrip (`Scripts/`)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[Script v1]], [[Script v2_stai]], [[Model Output Specs 5th Try]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]

Dokumen ini merekonstruksi kronologi pengembangan teknis dan evolusi arsitektur skrip dari **1st try** hingga **6th try**, cabang eksperimen **Plan-b**, utilitas pemrosesan data, serta submodul deployment smartwatch di repositori skripsi ini.

---

## 🗺️ Peta Direktori `Scripts/`

```
Scripts/
├── 1st try/                        # Fondasi awal (All-in-one v1 & v2_stai)
│   ├── v1.ipynb                    # 3-kelas protokol WESAD (Base, TSST, Fun)
│   └── v2_stai.ipynb               # Pelabelan biner STAI awal (ambang 0.2)
│
├── 2nd try/                        # Audit skoring STAI & fitur sub-jendela
│   ├── Utils.ipynb                 # Audit reverse-scoring STAI seluruh 15 subjek
│   └── v2_stai_acc_subwindow.ipynb # Penambahan 36 fitur sub-jendela akselerometer (5s/0.25s)
│
├── 3rd try/                        # Eksperimen downsampling & modul pernapasan
│   ├── fb_code/respiration.py      # Modul ekstraksi pernapasan lokal
│   └── v3_downsampled.ipynb        # Eksperimen downsampling sinyal
│
├── 4th try/                        # Standardisasi sliding window 6s & analisis modalitas
│   ├── stai_6s_pipeline.py         # Pipeline otomasi window 60s step 6s
│   ├── stai_6s_reference_models.ipynb # Model acuan RF, XGBoost, SVM, KNN
│   ├── train_stai_6s_models.py     # Skrip pelatihan CLI modular
│   ├── analyze_sensor_modality_weight.ipynb # Analisis kontribusi modalitas sensor
│   └── Sliced_sensor_modals/       # Eksperimen pemotongan modalitas sensor
│
├── 5th try/                        # Standardisasi penskalaan & isolasi leakage
│   ├── stai_6s_reference_models_stdized.ipynb # Pipeline StandardScaler -> MinMaxScaler
│   ├── analyze_sensor_modality_weight.ipynb   # Ablasi kontribusi keputusan per modalitas
│   ├── Minimum_sensor_modals/      # Eksperimen subset sensor minimal
│   └── Sliced_sensor_modals/       # Eksperimen sensor tersegmentasi
│
├── 6th try/                        # Rekonsiliasi benchmark Alkurdi et al. (2025)
│   ├── 2class/                     # Klasifikasi biner terektifikasi (15/15 TSST positif)
│   │   └── stai_2class_alkurdi_alignment.ipynb # Alignment Alkurdi, Scheme A vs Scheme B (LOSO)
│   └── 3class/                     # Klasifikasi 3-kelas kontinu (ambang ±1/24)
│       └── stai_3class_alkurdi_alignment.ipynb # Partisi kontinu tanpa celah ambang
│
├── Plan-b/                         # Analisis korelasi kontinu & mitigasi penurunan STAI
│   ├── 01_analyze_stai_drops.py    # Investigasi anomali penurunan skor STAI saat TSST
│   ├── 02_feature_correlation_analysis.py # Korelasi fitur per subjek terhadap delta STAI
│   ├── 03_train_stai_drop_models.py # Model prediksi perubahan STAI kontinu
│   ├── stai_drop_analysis.ipynb
│   └── window_feature_stai_correlation_analysis.ipynb
│
├── Utils/                          # Alat bantu pemrosesan batch & inspeksi
│   ├── clear_kernel_memory.ipynb   # Manajemen memori RAM/VRAM
│   ├── inspect_acc_dataset.ipynb   # Validasi integritas sumbu akselerometer
│   ├── relabel_and_evaluate_6th_try.py # Relabeling batch & evaluasi LOSO otomatis
│   └── generate_2class_data_and_eval.py # Pembangkitan dataset biner 6th try
│
├── environment.yml                 # Spesifikasi lingkungan conda
└── requirements.txt                # Dependensi pustaka Python pip
```

---

## 📈 Kronologi Evolusi Metodologis

### 1. Generasi Pertama: `1st try` (Oktober 2026)
- **Fokus:** Mengadopsi kode upstream `_forked/fb_code` menjadi notebook mandiri.
- **`v1.ipynb`:** Mengklasifikasikan kondisi stimulus laboratorium WESAD (3-kelas: Baseline, Stress/TSST, Amusement) dengan jendela geser 60 detik (stride 20 detik), menghasilkan 118 fitur fisiologis. Akurasi random split mencapai **100.0%** ([[Script v1]]).
- **`v2_stai.ipynb`:** Merumuskan hipotesis State Anxiety subjektif menggunakan selisih rerata kuesioner STAI per kondisi terhadap baseline ($\Delta \ge 0{,}2$, stride 12 detik). Akurasi XGBoost mencapai **99.4%**, tetapi SVM mengalami *majority class collapse* akibat ketidakseimbangan kelas 8:1 ([[Script v2_stai]]).

### 2. Generasi Kedua: `2nd try` (Akhir Oktober 2026)
- **Temuan Kunci:** Audit terhadap `wesad_readme.pdf` mengungkapkan bahwa WESAD menggunakan shortened STAI (6 butir) dengan aturan *reverse-scoring* pada butir positif (butir 1, 4, dan 6 via $5 - \text{raw}$).
- **`Utils.ipynb`:** Melakukan audit menyeluruh pada 15 subjek WESAD (`S2`–`S17`). Terungkap bahwa heterogenitas respons STAI sangat tinggi; kenaikan skor TSST tidak terjadi secara merata pada semua subjek ([[2026-10-24 Research Note]]).
- **`v2_stai_acc_subwindow.ipynb`:** Menambahkan 36 fitur sub-jendela akselerometer kinematik (durasi 5.0 detik, pergeseran 0.25 detik) untuk menangkap getaran mikro dan dinamika postural, meningkatkan dimensi fitur menjadi 154 kolom.

### 3. Generasi Ketiga: `3rd try`
- **Eksperimen Resampling & Modul Lokal:** Menguji downsampling sinyal fisiologis untuk efisiensi komputasi wearable, serta memisahkan algoritma ekstraksi respirasi ke dalam modul lokal `fb_code/respiration.py`.

### 4. Generasi Keempat: `4th try` (Awal November 2026)
- **Standardisasi Jendela:** Menetapkan durasi jendela 60 detik dengan pergeseran waktu seragam **6 detik** (overlap temporal 90%), selaras dengan literatur Alkurdi et al. (2025).
- **Analisis Modalitas Sensor:** Notebook `analyze_sensor_modality_weight.ipynb` menguji kontribusi relatif 7 modalitas sensor. Ditemukan bahwa penghitungan bobot fitur mentah bias terhadap akselerometer karena jumlah kolom fitur ACC (82 fitur) mendominasi total fitur ([[2026-11-05 Research Note]]).
- **Metode Decision Ablation:** Mengganti analisis bobot mentah dengan ablasi per-jendela (mengganti nilai fitur suatu modalitas dengan median cohort dan mengukur deviasi keputusan model).

### 5. Generasi Kelima: `5th try` (November 2026)
- **Pencegahan Data Leakage pada Scaling:** Notebook `stai_6s_reference_models_stdized.ipynb` mengisolasi pipeline `StandardScaler -> MinMaxScaler` di dalam tiap lipatan (*fold*) validasi, mencegah kebocoran informasi antar subjek ([[Model Output Specs 5th Try]]).
- **Ablasi ACC-Excluded:** Melatih ulang model tanpa akselerometer untuk membuktikan kemampuan sensor otonom murni (EDA, ECG, BVP, Resp, Temp).
- **Identifikasi Bug Zero-Inflation:** Analisis kohort mengungkap bahwa 8 dari 15 subjek menghasilkan 0 jendela cemas akibat transposisi pembacaan baris kuesioner dan ketiadaan kelas positif pada sebagian partisipan.

### 6. Generasi Keenam: `6th try` (November 2026 — Final Benchmark)
Generasi final yang menyelaraskan seluruh metodologi pra-pemrosesan sinyal dan evaluasi dengan standar disertasi referensi Alkurdi et al. (2025):
- **Alignment Jam Sensor Alkurdi:** Menggunakan outer-join `pd.to_datetime` berbasis indeks nanosekon, mempertahankan laju sampling asli (4 Hz EDA/Temp, 32 Hz ACC, 64 Hz BVP, 700 Hz Chest) tanpa interpolasi semu.
- **Perbaikan Total Parser Kuesioner:** Memperbaiki pengindeksan butir STAI-6 sehingga **15 dari 15 subjek (100%)** terbukti mengalami peningkatan kecemasan saat TSST.
- **Cabang 3-Kelas (`Scripts/6th try/3class/`):**
  - Menggunakan ambang kontinu berbasis unit resolusi minimum kuesioner ($\pm \frac{1}{24} \approx \pm 0{,}04167$).
  - Menghilangkan celah data tak terlabel antara $0{,}0$ dan $0{,}2$.
  - Menghasilkan kohort seimbang: 27.1% C0, 25.1% C1, 47.8% C2 ([[Model Output Specs 6th Try 3-Class]]).
- **Cabang 2-Kelas (`Scripts/6th try/2class/`):**
  - Formulasi biner terektifikasi: $\Delta \le 0$ (Class 0: 43.59%) vs $\Delta > 0$ (Class 1: 56.41%).
  - Evaluasi LOSO menghasilkan **74.43% akurasi** dan **0.7587 F1 kecemasan** pada Random Forest (`max_depth=4`) ([[Model Output Specs 6th Try 2-Class]]).
- **Komparasi Skema Validasi:**
  - **Scheme A (Naive Random Split 80/20):** Akurasi mencapai 93%–100% akibat kebocoran data temporal (*data leakage*).
  - **Scheme B (15-Fold Leave-One-Subject-Out):** Mengukur kemampuan generalisasi murni terhadap subjek baru (*unseen subjects*), membuktikan kelayakan ilmiah model untuk skripsi.

### 7. Cabang Riset Alternatif: `Plan-b/`
- Meneliti hubungan kontinu (*correlation analysis*) antara perubahan fitur fisiologis per subjek dengan perubahan skor STAI.
- Menganalisis fenomena subjek yang mengalami penurunan skor STAI (*STAI drops*) saat TSST untuk mengkaji mekanisme koping adaptif atau variasi persepsi kuesioner.

---

## 🔗 Hubungan Modul Skrip dengan Catatan Vault

| Direktori Skrip | Catatan Dokumentasi Teknis | Artefak Model Terkait |
|---|---|---|
| `Scripts/1st try/v1.ipynb` | [[Script v1]] | `Models/random_forest_model.joblib`, `wesad_stress_model.tflite` |
| `Scripts/1st try/v2_stai.ipynb` | [[Script v2_stai]] | `Models/binary_*.joblib` |
| `Scripts/4th try/` & `5th try/` | [[2026-11-05 Research Note]], [[Model Output Specs 5th Try]] | `Models/5th try/stai_baseline_relative_*.joblib` |
| `Scripts/6th try/2class/` | [[Model Output Specs 6th Try 2-Class]], [[2026-11-12 Research Note]] | `Models/6th try/2class/stai_2class_schemeA_*.joblib` |
| `Scripts/6th try/3class/` | [[Model Output Specs 6th Try 3-Class]], [[2026-11-12 Research Note]] | `Models/6th try/3class/stai_3class_schemeA_*.joblib` |
| `Deployment/test1/` | [[2026-10-29 Research Note]] | Wear OS Samsung Health Sensor SDK Application |
