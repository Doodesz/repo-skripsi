---
tags:
  - skripsi/pipeline
  - dataset/wesad
  - metodologi
  - kuesioner/stai
aliases:
  - Pipeline User
  - Pipeline STAI
  - v2_stai
  - Pipeline STAI User (v2_stai)
---
# Pipeline STAI User (`v2_stai`)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[2026-10-06 Research Note]], [[2026-10-24 Research Note]], [[Perbandingan Pipeline STAI vs Forked]], [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]

## 1. Ikhtisar (Overview)
Pipeline STAI merupakan pendekatan pemrosesan data kustom yang dirancang untuk mendeteksi kecemasan subjektif (*State Anxiety*) berdasarkan respons laporan mandiri (*self-report*) kuesioner State-Trait Anxiety Inventory (STAI), bukan semata-mata mengandalkan kondisi stimulus protokol laboratorium WESAD.

- **Pertanyaan Utama:** *“Apakah skor STAI laporan mandiri peserta meningkat relatif terhadap kondisi dasar (baseline) pribadinya?”*
- **Sifat Anotasi:** Subjektif psikologis dan relatif terhadap baseline individu (*within-subject baseline comparison*).

---

## 2. Karakteristik & Evolusi Mekanisme Pelabelan

### A. Formulasi Awal (`1st try` – `5th try`)
- **Ambang Batas Heuristik:** Kondisi non-Base diberi label **positif (1)** apabila rerata STAI meningkat sekurang-kurangnya $0{,}2$ poin dari rerata Base:
  $$\text{Mean STAI}_{\text{kondisi}} - \text{Mean STAI}_{\text{Base}} \ge 0{,}2$$
- **Isu Historis:** Terdapat celah data tak terlabel antara $0{,}0$ dan $0{,}2$ serta isu zero-inflation pada 8 subjek akibat pergeseran indeks pembacaan kuesioner.

### B. Formulasi Final Terektifikasi (`6th try`)
Mengintegrasikan audit reverse-scoring shortened STAI 6-butir (inversi butir 1, 4, 6 via $5 - \text{raw}$) dari [[2026-10-24 Research Note]]:
1. **Klasifikasi Biner Terektifikasi ([[Model Output Specs 6th Try 2-Class]]):**
   - $\Delta_{\text{sum}} = \text{condition\_stai\_sum} - \text{baseline\_stai\_sum}$
   - **Class 0 (No Anxiety Increase):** $\Delta_{\text{sum}} \le 0$ (43.59%)
   - **Class 1 (Anxiety Increased):** $\Delta_{\text{sum}} > 0$ (56.41%)
   - *100% subjek (15/15)* terbukti mengalami peningkatan kecemasan saat sesi TSST.
2. **Klasifikasi 3-Kelas Kontinu ([[Model Output Specs 6th Try 3-Class]]):**
   - Menggunakan ambang $\pm \frac{1}{24} \approx \pm 0{,}04167$ (satuan 1 poin kuesioner):
     - **Class 0:** $\Delta_{\text{norm}} < -1/24$ (27.11%)
     - **Class 1 (Neutral Band):** $-1/24 \le \Delta_{\text{norm}} \le 1/24$ (25.05%)
     - **Class 2:** $\Delta_{\text{norm}} > 1/24$ (47.84%)

---

## 3. Segmentasi Jendela (*Windowing*) & Integritas Waktu

- **Parameter Windowing:** Jendela 60 detik dengan pergeseran (*stride*) 6 detik (overlap 90% pada 4th–6th try; awalnya 12 detik pada 1st try).
- **Strict Temporal Enclosure:** Jendela hanya dipertahankan apabila 100% rentang durasinya berada di dalam batas kondisi bertiming dengan respons STAI valid. Jendela transisi antarkondisi dibuang (*dropped*).
- **Sub-Jendela Akselerometer:** Mengurai jendela menjadi sub-jendela 5.0 detik (stride 0.25 detik) untuk mengekstrak 36 fitur kinematik mikro ([[Katalog Fitur Fisiologis]]).

---

## 4. Pertimbangan Metodologis & Evaluasi

> [!important] Pencegahan Kebocoran Data (*Data Leakage*)
> Karena beberapa jendela berurutan mewarisi satu label kondisi yang sama dan overlap mencapai 90%, pengujian model wajib menggunakan **Leave-One-Subject-Out (LOSO) Cross-Validation** dengan penskalaan fitur terisolasi per lipatan (*in-fold scaling*). Pengujian acak window-level (Scheme A) menghasilkan akurasi semu 93%–100% yang tidak mencerminkan generalisasi pada subjek baru.

---

## 5. Tautan Terkait
- Catatan temuan metodologis: [[2026-10-06 Research Note]]
- Audit kuesioner STAI: [[2026-10-24 Research Note]]
- Pipeline alternatif berbasis protokol: [[Pipeline Forked (WESAD Protocol)]]
- Komparasi komprehensif: [[Perbandingan Pipeline STAI vs Forked]]
- Spesifikasi model: [[Model Output Specs 6th Try 2-Class]], [[Model Output Specs 6th Try 3-Class]]
