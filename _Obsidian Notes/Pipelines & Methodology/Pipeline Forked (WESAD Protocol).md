---
tags:
  - skripsi/pipeline
  - dataset/wesad
  - metodologi
aliases:
  - Pipeline Forked
  - WESAD Forked Pipeline
  - Pipeline Forked (WESAD Protocol)
---
# Pipeline Forked (WESAD Protocol-Based)

> **Indeks Vault:** [[MOC - Skripsi Repository]]  
> **Catatan Terkait:** [[2026-10-06 Research Note]], [[Pipeline STAI User (v2_stai)]], [[Perbandingan Pipeline STAI vs Forked]], [[Forked Repository Overview]]

## 1. Ikhtisar (Overview)
Pipeline `_forked` merupakan pendekatan pemrosesan data berbasis anotasi protokol standar dataset WESAD (Wearable Stress and Affect Detection) yang diadaptasi dari repositori disertasi upstream `anxietyFB` (Alkurdi et al.). 

- **Pertanyaan Utama:** *“Jendela ini berasal dari kondisi protokol eksperimen laboratorium yang mana?”*
- **Sifat Anotasi:** Objektif berbasis waktu pelaksanaan stimulus terpandu dalam eksperimen lab.

---

## 2. Karakteristik & Metodologi

### A. Mekanisme Pelabelan
- **Sumber Label:** Menggunakan file anotasi protokol resmi WESAD.
- **Kategori Kondisi:** Membagi data fisiologis ke dalam segmen kondisi protokol eksperimen:
  - *Baseline* (kondisi netral/istirahat)
  - *TSST (Trier Social Stress Test)* (kondisi stres terinduksi)
  - *Amusement* (kondisi relaksasi/hiburan video komedi)
  - *Meditation*, dsb.
- **Asumsi:** Peserta diasumsikan mengalami kondisi afektif tertentu (misal: stres) secara seragam selama stimulus protokol berlangsung.

### B. Segmentasi & Windowing
- Mengikuti batasan waktu protokol eksperimen WESAD.
- Menggunakan jendela 60 detik non-overlapping (*tumbling window*, stride 60 detik) pada kode asli forked, menghasilkan sekitar 37–40 jendela per subjek.
- Penanda waktu (*timestamps*) disinkronkan melalui datetime index outer join pada basis frekuensi tertinggi (700 Hz).

---

## 3. Kelebihan & Keterbatasan

> [!success] Kelebihan
> - **Reproduksibilitas Tinggi:** Mengikuti konvensi baku penelitian literatur WESAD sebelumnya (Schmidt et al. 2018, Alkurdi et al. 2025).
> - **Keselarasan Label Eksperimental:** Batas waktu stimulus terkontrol secara presisi di laboratorium.

> [!warning] Keterbatasan
> - **Mengabaikan Variasi Subjektif:** Mengasumsikan respons fisiologis setiap individu identik terhadap stimulus yang sama (padahal tidak semua peserta otomatis stres saat TSST).
> - **Bukan Penilaian Klinis/Mandiri:** Label merefleksikan adanya stimulus eksternal, bukan evaluasi tingkat kecemasan psikologis yang benar-benar dirasakan peserta.

---

## 4. Tautan Terkait
- Catatan komparasi utama: [[2026-10-06 Research Note]]
- Alternatif pipeline berbasis kuesioner: [[Pipeline STAI User (v2_stai)]]
- Komparasi mendalam: [[Perbandingan Pipeline STAI vs Forked]]
- Modul pustaka: [[Modul Library fb_code]]
- Gambaran umum repositori: [[Forked Repository Overview]]
