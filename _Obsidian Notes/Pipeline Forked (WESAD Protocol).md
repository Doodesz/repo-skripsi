---
tags:
  - skripsi/pipeline
  - dataset/wesad
  - metodologi
aliases:
  - Pipeline Forked
  - WESAD Forked Pipeline
---
# Pipeline Forked (WESAD Protocol-Based)

## 1. Ikhtisar (Overview)
Pipeline `_forked` merupakan pendekatan pemrosesan data berbasis anotasi protokol standar dataset WESAD (Wearable Stress and Affect Detection). 

- **Pertanyaan Utama:** *“Jendela ini berasal dari kondisi protokol eksperimen yang mana?”*
- **Sifat Anotasi:** Objektif berbasis waktu pelaksanaan stimulus terpandu dalam eksperimen lab.

---

## 2. Karakteristik & Metodologi

### A. Mekanisme Pelabelan
- **Sumber Label:** Menggunakan file anotasi protokol resmi WESAD.
- **Kategori Kondisi:** Membagi data fisiologis ke dalam segmen kondisi protokol eksperimen:
  - *Baseline* (kondisi netral/istirahat)
  - *TSST (Trier Social Stress Test)* (kondisi stres terinduksi)
  - *Amusement* (kondisi relaksasi/hiburan)
  - *Meditation*, dsb.
- **Asumsi:** Peserta diasumsikan mengalami kondisi afektif tertentu (misal: stres) secara seragam selama stimulus protokol berlangsung.

### B. Segmentasi & Windowing
- Mengikuti batasan waktu protokol eksperimen WESAD.
- Penanda waktu (*timestamps*) diambil langsung dari sinkronisasi perangkat dan pencatat waktu eksperimen lab.

---

## 3. Kelebihan & Keterbatasan

> [!success] Kelebihan
> - **Reproduksibilitas Tinggi:** Mengikuti konvensi baku penelitian literatur WESAD sebelumnya.
> - **Keselarasan Label Eksperimental:** Batas waktu stimulus terkontrol secara presisi di laboratorium.

> [!warning] Keterbatasan
> - **Mengabaikan Variasi Subjektif:** Mengasumsikan respons fisiologis setiap individu identik terhadap stimulus yang sama (padahal tidak semua peserta otomatis stres saat TSST).
> - **Bukan Penilaian Klinis/Mandiri:** Label merefleksikan adanya stimulus eksternal, bukan evaluasi tingkat kecemasan psikologis yang benar-benar dirasakan peserta.

---

## 4. Tautan Terkait
- Catatan komparasi utama: [[Development Findings]]
- Alternatif pipeline berbasis kuesioner: [[Pipeline STAI User (v2_stai)]]
