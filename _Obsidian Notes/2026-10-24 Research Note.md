---
tags:
  - skripsi/metodologi
  - dataset/wesad
  - evaluasi
  - stai
---
# Research Note — 2026-10-24

## 1. Aturan STAI WESAD dari `wesad_readme.pdf`

`wesad_readme.pdf` mengonfirmasi bahwa WESAD tidak menyimpan STAI-S lengkap, melainkan **shortened STAI** dengan enam butir berikut, dalam urutan CSV:

1. *I feel at ease*
2. *I feel nervous*
3. *I am jittery*
4. *I am relaxed*
5. *I am worried*
6. *I feel pleasant*

Skala jawaban adalah `1 = Not at all`, `2 = Somewhat`, `3 = Moderately so`, dan `4 = Very much so`.

Butir positif yang harus dibalik sebelum agregasi adalah:

- item 1 — *I feel at ease*;
- item 4 — *I am relaxed*;
- item 6 — *I feel pleasant*.

Reverse-scoring menggunakan transformasi:

```python
scored = 5 - raw
```

Dengan demikian `1 ↔ 4` dan `2 ↔ 3`. Item 2, 3, dan 5 sudah berarah positif terhadap state anxiety dan tidak dibalik. Skor kondisi harus dihitung sebagai rata-rata item valid **setelah** reverse-scoring, bukan rata-rata mentah.

PDF tersebut juga mengonfirmasi bahwa kuesioner diisi setelah lima kondisi utama—baseline, amusement, stress, meditation 1, dan meditation 2. Kondisi `bRead`, `fRead`, dan `sRead` harus diabaikan sebagai kondisi dataset. Waktu pada `START` dan `END` menggunakan format `minutes.seconds`, dihitung dari awal perekaman RespiBAN.

## 2. Koreksi pada Notebook STAI

Notebook [v2_stai_acc_subwindow.ipynb](../Scripts/All-in-one/v2_stai_acc_subwindow.ipynb) telah diperbaiki agar:

- mendefinisikan enam item shortened STAI WESAD;
- melakukan reverse-scoring pada item 1, 4, dan 6;
- menyimpan `stai_responses_raw` dan `stai_responses_scored`;
- menghitung `stai_mean` berdasarkan respons yang sudah dikoreksi;
- tetap menggunakan threshold operasional `STAI_INCREASE_THRESHOLD = 0.2` dengan aturan `>=`.

File `.pkl` hasil preprocessing lama perlu dibuat ulang dengan menjalankan ulang bagian preprocessing notebook, karena output lama masih dibuat menggunakan rata-rata mentah.

## 3. Audit S4 Setelah Reverse-Scoring

File [S4_quest.csv](../WESAD/S4/S4_quest.csv) memiliki urutan kondisi `Base`, `Fun`, `Medi 1`, `TSST`, `Medi 2`, `bRead`, `fRead`, dan `sRead`. Bagian STAI berisi lima baris respons untuk enam kondisi pertama; kolom pembacaan tidak memiliki respons STAI pada file ini.

Karena hanya lima respons yang tersedia, item keenam tidak ada pada audit S4. Setelah item positif yang tersedia dibalik—item 1 dan item 4—hasilnya adalah:

| Kondisi | Respons mentah | Respons setelah reverse-scoring | Rata-rata scored |
|---|---|---|---:|
| Base | 3, 3, 1, 1, 2 | 2, 3, 1, 4, 2 | 2,4 |
| Fun | 1, 1, 1, 3, 1 | 4, 1, 1, 2, 1 | 1,8 |
| Medi 1 | 1, 1, 1, 2, 1 | 4, 1, 1, 3, 1 | 2,0 |
| TSST | 4, 4, 2, 2, 3 | 1, 4, 2, 3, 3 | 2,6 |
| Medi 2 | 1, 1, 1, 1, 1 | 4, 1, 1, 4, 1 | 2,2 |

Dengan scoring yang dikoreksi, S4 memiliki:

```text
TSST - Base = 2,6 - 2,4 = +0,2
```

S4 masih diklasifikasikan sebagai meningkat bila aturan label menggunakan `>= 0,2`, tetapi kenaikannya tepat berada pada batas threshold dan jauh lebih kecil daripada kesimpulan lama `+1,0` berbasis rata-rata mentah. Nilai ini bukan skor STAI-S lengkap/klinis karena WESAD menggunakan versi shortened dan S4 hanya memiliki lima respons.

## 4. Implikasi Metodologis

- Kesimpulan bahwa TSST tidak meningkatkan STAI pada semua peserta tidak boleh dibuat hanya dari beberapa subjek.
- Hipotesis yang lebih tepat adalah bahwa respons STAI terhadap TSST bersifat heterogen antar-subjek.
- Seluruh subjek perlu dibuatkan tabel `Base`, `TSST`, dan `delta_STAI` setelah reverse-scoring.
- Pisahkan tiga target:
  1. `condition_label`: label protokol WESAD (`baseline`, `stress`, `amusement`);
  2. `delta_STAI`: perubahan kontinu dari baseline peserta;
  3. `STAI_increase`: label biner operasional setelah aturan scoring dan threshold ditetapkan.
- Korelasi yang relevan adalah korelasi pada level peserta antara `delta_STAI` dan `delta_feature`, bukan korelasi ratusan window yang mewarisi satu skor kuesioner.
- Model Alkurdi yang memakai label kondisi eksperimen tidak otomatis menjadi detektor anxiety berbasis STAI.

## 5. Tindak Lanjut

1. Jalankan ulang preprocessing seluruh subjek dengan notebook yang sudah dikoreksi.
2. Audit tabel skor Base–TSST per peserta.
3. Hitung proporsi peserta dengan `delta_STAI` positif, nol, dan negatif.
4. Uji sensitivitas threshold, termasuk threshold di sekitar `0,2`.
5. Lanjutkan korelasi `delta_STAI` dengan perubahan fitur fisiologis pada level peserta.

## 6. Perbandingan Skor STAI TSST dengan Kondisi Referensi

Notebook [Utils.ipynb](../Scripts/2nd%20try/Utils.ipynb) membaca seluruh 15 subjek WESAD, menerapkan reverse-scoring shortened STAI, lalu membandingkan skor TSST dengan skor Base, Fun, Medi 1, dan Medi 2. Hanya pasangan kondisi dengan skor valid yang dihitung. Perbandingan memakai aturan `TSST >= kondisi referensi`, tetapi hasilnya sekarang dipisahkan menjadi **sama**, **lebih tinggi**, dan **sama atau lebih tinggi**.

| Perbandingan | Subjek valid | TSST = referensi | TSST > referensi | TSST ≥ referensi | TSST < referensi |
|---|---:|---:|---:|---:|---:|
| TSST vs Base | 15 | 3 | 6 | 9 (60,0%) | 6 |
| TSST vs Fun | 15 | 3 | 11 | 14 (93,3%) | 1 |
| TSST vs Medi 1 | 15 | 1 | 13 | 14 (93,3%) | 1 |
| TSST vs Medi 2 | 15 | 3 | 11 | 14 (93,3%) | 1 |

Daftar subjek juga diekspor oleh notebook untuk setiap kategori. Hasil ini menunjukkan bahwa skor STAI TSST lebih tinggi atau sama dengan Base pada 9 dari 15 subjek, tetapi pada enam subjek lainnya skor TSST lebih rendah. Terhadap kondisi Fun dan kedua kondisi meditasi, hampir seluruh subjek memiliki skor TSST lebih tinggi atau sama. Karena itu, klaim bahwa TSST meningkatkan STAI harus dibatasi pada perbandingan yang jelas dan tidak boleh disimpulkan hanya dari agregat `TSST ≥ reference`; kesetaraan skor dan kenaikan skor perlu dilaporkan secara terpisah.

Output analisis tersimpan di:

- [stai_scored_condition_scores.csv](../Data/Analysis/stai_scored_condition_scores.csv)
- [stai_tsst_at_least_reference_counts.csv](../Data/Analysis/stai_tsst_at_least_reference_counts.csv)

---

## 7. Klarifikasi Target Model Alkurdi dan Hubungannya dengan STAI

### Target yang Sebenarnya Digunakan

Pemeriksaan terhadap kode pada repositori referensi `_forked` menunjukkan bahwa model WESAD menggunakan **label kondisi eksperimen**, bukan skor STAI sebagai ground truth anxiety. Label yang digunakan adalah:

| Label | Kondisi | Interpretasi |
|---:|---|---|
| `0` | Amusement | Kondisi menonton film/hiburan |
| `1` | Baseline | Kondisi netral atau istirahat |
| `2` | Stress / TSST | Kondisi stres yang diinduksi protokol |

Definisi tersebut terlihat pada [`_forked/fb_code/feature_extraction.py`](../_forked/fb_code/feature_extraction.py) dan notebook [`Feature Based Model WESAD.ipynb`](../_forked/Feature%20Based%20Model%20WESAD.ipynb), melalui pemetaan `label_dict = {'baseline': 1, 'stress': 2, 'amusement': 0}`.

Dengan demikian, model Alkurdi menjawab pertanyaan:

> “Berasal dari kondisi protokol WESAD yang mana jendela fisiologis ini?”

Model tersebut tidak secara langsung menjawab:

> “Apakah skor STAI peserta ini meningkat dan apakah peserta sedang mengalami anxiety secara subjektif?”

### Mengapa Judulnya Tetap Menggunakan Istilah *Anxiety Detection*?

Istilah *anxiety detection* dalam penelitian Alkurdi perlu dibaca sebagai istilah operasional dalam konteks eksperimen dan bukan sebagai bukti bahwa setiap subjek mengalami peningkatan anxiety yang terverifikasi oleh STAI. Kondisi TSST memang dirancang untuk memicu stres dan kemungkinan respons anxiety, tetapi stimulus yang sama tidak menghasilkan respons subjektif yang identik pada setiap peserta.

Oleh sebab itu, akurasi tinggi pada klasifikasi `baseline` versus `stress/TSST` hanya membuktikan bahwa sinyal fisiologis membedakan kondisi eksperimen tersebut. Akurasi itu belum membuktikan hubungan satu-ke-satu antara label `stress` dan peningkatan skor STAI individual.

### Catatan tentang RADWear

Notebook RADWear menggunakan pemetaan terpisah, yaitu `calibration`, `low anxiety`, dan `high anxiety`. Pemetaan ini berasal dari kondisi/rotasi studi RADWear dan tidak boleh disamakan begitu saja dengan skor STAI WESAD. Jadi, terdapat perbedaan antara label kondisi WESAD, label kondisi RADWear, dan label self-report STAI yang sedang dibangun dalam penelitian ini.

---

