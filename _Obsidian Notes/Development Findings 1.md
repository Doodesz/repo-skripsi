---
tags:
  - skripsi/metodologi
  - dataset/wesad
  - evaluasi
---
# Temuan & Komparasi Pipeline (Findings)

## 1. Temuan Terkait Literatur
- **Paper referensi:** *Extending Anxiety Detection from Multimodal Wearables in Controlled Conditions to Real-World Environments*
  - Tidak menggunakan windowing overlap (hanya window berdurasi 60 detik tanpa overlap).

---

## 2. Perbandingan Paradigma Pelabelan

Perbedaan mendasar antara kedua pipeline terletak pada **makna semantik dari label target**:

| Aspek               | [[Pipeline Forked (WESAD Protocol)\|Pipeline _forked]]            | [[Pipeline STAI User (v2_stai)\|Pipeline User (v2_stai.ipynb)]]                             |
| :------------------ | :---------------------------------------------------------------- | :------------------------------------------------------------------------------------------ |
| **Pertanyaan Inti** | Jendela ini berasal dari kondisi protokol eksperimen yang mana?   | Apakah rata-rata STAI laporan mandiri peserta meningkat $\ge 0{,}2$ poin dari baseline-nya? |
| **Dasar Pelabelan** | Anotasi kondisi protokol WESAD (baseline, TSST/stres, amusement). | Respons kuesioner STAI peserta per kondisi terhadap rata-rata kondisi Base.                 |
| **Tingkat Anotasi** | Eksperimental / stimulus terinduksi protokol.                     | Subjektif / *self-report* berbasis ambang operasional individu.                             |

---

## 3. Detail Perbedaan Teknis & Metodologis

### Cakupan Label Positif
- **Kondisi Beragam:** Label positif berlaku untuk setiap kondisi non-Base yang memenuhi kriteria peningkatan skor, bukan hanya kondisi TSST.
- **Definisi:** Label positif menandakan peningkatan $\ge 0{,}2$ terhadap baseline individu, bukan semata-mata kecemasan murni akibat TSST.

### Penentuan Ambang Batas (*Threshold*)
- Nilai `0.2` merupakan ambang operasional kerja (*operational heuristic*), bukan ambang diagnostik klinis resmi.
- Perlu dijustifikasi secara eksplisit dalam bab metodologi serta diuji sensitivitasnya (apakah dinamika model berubah jika ambang digeser).

### Strategi Windowing & Segmentasi Waktu
- **Filtering Jendela:** Jendela (durasi 60 detik, step 12 detik) hanya dipertahankan apabila seluruh rentang waktunya berada di dalam batas kondisi bertiming yang memiliki respons STAI valid.
- **Dampak Data:** Kondisi dengan durasi singkat berisiko tidak menghasilkan jendela sama sekali, sehingga data yang teranalisis merupakan subset dari keseluruhan protokol.
- **Ketidakselarasan Waktu:** Titik awal dan akhir kondisi pada file kuesioner tidak selalu berimpit persis dengan anotasi protokol `_forked` (terutama pada area perbatasan antar-kondisi).

### Unit Analisis & Independensi Data
- Fitur fisiologis dihitung pada level jendela (60 detik), sedangkan label STAI melekat pada level kondisi per peserta.
- **Non-independensi:** Beberapa jendela berurutan mewarisi satu label kondisi yang sama. Evaluasi model (seperti *cross-validation*) wajib dikelompokkan berdasarkan subjek (*Subject-wise / GroupKFold*) agar tidak terjadi kebocoran data (*data leakage*).

### Integritas Pemrosesan Respons STAI
- Parser menghitung rata-rata dari butir kuesioner yang tersedia.
- Pastikan apakah aturan skoring baku STAI membutuhkan pembalikan skor (*reverse-scored items*) sebelum dihitung.
- Antisipasi keberadaan *missing values* pada butir kuesioner agar bobot rata-rata antar-kondisi tetap berimbang.

### Validitas Komparasi & Batasan Klinis
- **Inkompatibilitas Skor:** Karena perbedaan segmentasi jendela, pemrosesan sinyal, dan definisi label target, metrik performa kedua pipeline tidak dapat dibandingkan secara *head-to-head*.
- **Interpretasi Klinis:** Baik stres protokol maupun STAI mandiri bukanlah diagnosis kecemasan klinis; respons fisiologis dan pelaporan subjektif dapat berbeda antar-individu.

### Implikasi untuk Hipotesis TSST dan Implementasi Wearable
- TSST dirancang sebagai pemicu stres, tetapi tidak boleh diasumsikan meningkatkan skor STAI setiap peserta. Peserta dapat berbeda dalam respons subjektif; karena itu, tidak adanya peningkatan pada sebagian peserta merupakan hasil yang perlu dianalisis, bukan bukti otomatis bahwa parser atau metode label salah.
- Pisahkan dua pertanyaan penelitian: **(1)** apakah STAI berubah dari Base ke TSST pada tingkat kelompok/peserta, dan **(2)** apakah fitur wearable dapat memprediksi label STAI operasional. Label protokol TSST dan label perubahan STAI mewakili target berbeda.
- Pendekatan STAI-baseline-relative dapat dilanjutkan jika targetnya dinyatakan sebagai **peningkatan STAI yang dilaporkan sendiri relatif terhadap baseline**, bukan diagnosis kecemasan atau respons TSST yang pasti.
- Sebelum melatih ulang model, periksa perubahan Baseâ€“TSST per peserta dan distribusi perubahan; laporkan ukuran efek/ketidakpastian dan proporsi peserta yang naik, tetap, atau turun. Lakukan analisis sensitivitas terhadap ambang `0.2` dan verifikasi aturan skoring STAI yang berlaku.
- Untuk deteksi di smartwatch dunia nyata, kumpulkan sinyal dan *self-report* yang tersinkronisasi pada waktu yang sesuai. STAI per kondisi memberi label tingkat kondisi, bukan *ground truth* kontinu per jendela; pertimbangkan target fisiologis/stres dan kecemasan subjektif secara eksplisit dan terpisah.
- Evaluasi model dengan pemisahan peserta (tanpa peserta yang sama di data latih dan uji), lalu validasi dalam konteks realistis. Ukur juga *false alarms*, kejadian terlewat, dan variasi antar-peserta, bukan hanya akurasi.
- **Kesimpulan praktis:** temuan ini tidak mengharuskan proyek dihentikan; ia memperjelas definisi target dan desain validasi yang dibutuhkan sebelum implementasi perangkat.

---

## 4. Draf Redaksi untuk Skripsi / Laporan

> [!quote] Ringkasan Metodologis untuk Bab Metodologi / Diskusi
> Pipeline WESAD `_forked` menetapkan label berdasarkan anotasi protokol eksperimen, sedangkan analisis pada penelitian ini merumuskan label berdasarkan respons kuesioner STAI pada tingkat peserta dan kondisi. Untuk setiap peserta, respons STAI yang valid dirata-ratakan dalam setiap kondisi. Suatu kondisi non-Base diberi label sebagai kondisi dengan peningkatan STAI apabila rata-ratanya melebihi rata-rata Base peserta tersebut sekurang-kurangnya 0,2 poin. Ambang ini merupakan aturan operasional berdasarkan laporan diri (*self-report*), bukan ambang diagnostik klinis.
> 
> Oleh karena itu, label STAI tidak setara secara langsung dengan label stres yang ditentukan oleh protokol, dan keikutsertaan dalam TSST tidak menjamin adanya peningkatan skor STAI. Jendela fisiologis hanya disimpan jika seluruh rentangnya berada di dalam batas kondisi bertiming yang memiliki label valid; jumlah jendela yang dihasilkan bukanlah jumlah observasi peserta yang independen.

