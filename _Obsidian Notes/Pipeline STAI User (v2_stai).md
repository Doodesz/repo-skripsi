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
---
# Pipeline STAI User (`v2_stai.ipynb`)

## 1. Ikhtisar (Overview)
Pipeline `v2_stai.ipynb` merupakan pendekatan pemrosesan data kustom yang dirancang oleh pengguna. Pipeline ini menentukan label target berdasarkan respons laporan mandiri (*self-report*) kuesioner State-Trait Anxiety Inventory (STAI) peserta, bukan semata-mata mengandalkan kondisi protokol laboratorium.

- **Pertanyaan Utama:** *“Apakah rata-rata skor STAI laporan mandiri peserta untuk kondisi ini setidaknya $0{,}2$ poin lebih tinggi daripada baseline peserta tersebut?”*
- **Sifat Anotasi:** Subjektif dan relatif terhadap baseline individu (*within-subject baseline comparison*).

---

## 2. Karakteristik & Mekanisme Pelabelan

### A. Perhitungan & Aturan Ambang Batas (*Threshold*)
- **Rata-rata Skor per Kondisi:** Untuk setiap peserta, respons butir STAI yang valid dirata-ratakan pada tiap kondisi.
- **Kondisi Base sebagai Titik Acu:** Rata-rata kondisi `Base` peserta dijadikan nilai referensi awal.
- **Kriteria Positif:** Kondisi non-Base diberi label **positif (1)** apabila:
  $$\text{Mean STAI}_{\text{kondisi}} - \text{Mean STAI}_{\text{Base}} \ge 0{,}2$$
- **Cakupan Label Positif:** Aturan ini berlaku untuk setiap kondisi non-Base yang memenuhi kualifikasi, tidak terbatas pada sesi TSST saja.
- **Ambang Operasional:** Nilai $0{,}2$ merupakan ambang kerja operasional (*heuristic*), bukan ambang diagnostik klinis resmi.

### B. Segmentasi Jendela (*Windowing*) & Waktu
- **Parameter Windowing:** Menggunakan jendela 60 detik dengan pergeseran (*stride*) 12 detik.
- **Penyaringan Jendela (*Filtering*):** Hanya jendela yang **seluruh rentang durasinya** berada di dalam batas waktu kondisi bertiming dengan respons STAI valid yang dipertahankan.
  - Kondisi berdurasi singkat dapat gugur dan tidak menghasilkan jendela.
- **Sumber Waktu:** Waktu mulai (*start*) dan akhir (*end*) diambil dari berkas kuesioner, yang mungkin memiliki pergeseran batas dibandingkan penanda waktu protokol WESAD baku.

---

## 3. Pertimbangan Metodologis & Evaluasi

> [!important] Unit Analisis & Pencegahan Kebocoran Data (*Data Leakage*)
> - **Non-independensi Data:** Label STAI berada pada tingkat kondisi, sedangkan fitur fisiologis berada pada tingkat jendela 60 detik. Beberapa jendela berturut-turut mewarisi label kondisi yang sama.
> - **Strategi Evaluasi:** Validasi model (*cross-validation*) wajib dikelompokkan berdasarkan subjek (*GroupKFold* / *Leave-One-Subject-Out*) agar jendela dari subjek yang sama tidak tersebar di data latih dan uji.

> [!warning] Asumsi Pemrosesan Kuesioner STAI
> - Parser menghitung rata-rata dari butir jawaban yang tersedia.
> - Perlu dipastikan apakah terdapat butir pernyataan yang memerlukan pembalikan skor (*reverse scoring*) sesuai panduan instrumen STAI standar.
> - Keberadaan data hilang (*missing responses*) dapat memengaruhi pembobotan rata-rata antar-kondisi.

---

## 4. Tautan Terkait
- Catatan komparasi utama: [[Development Findings]]
- Alternatif pipeline berbasis protokol: [[Pipeline Forked (WESAD Protocol)]]
- Referensi paper windowing: *Extending Anxiety Detection from Multimodal Wearables in Controlled Conditions to Real-World Environments* (belum ada catatan vault)
