---
tags:
  - skripsi/implementasi
  - wearable
  - samsung-health-sensor-sdk
  - ecg
  - bvp
  - development-findings
aliases:
  - Samsung Health Sensor SDK ECG Findings
---
# Research Note — 2026-10-29

## Samsung Health Sensor SDK: BVP dan ECG pada Galaxy Watch

### Konteks

Implementasi dilakukan pada aplikasi Wear OS berbasis Samsung Health Sensor SDK 1.4.1 di repositori [`Deployment/test1`](../Deployment/test1/). Target pengukuran adalah BVP/PPG hijau dan ECG melalui `HealthTracker`.

Catatan terkait:

- [[Development Findings 1]] — batasan interpretasi sinyal wearable dan validasi realistis.
- [[Katalog Fitur Fisiologis]] — konteks fitur fisiologis yang dapat diekstraksi dari sinyal.
- [[Pra-Pemrosesan Sinyal & Filtering]] — pemrosesan sinyal setelah data berhasil dikumpulkan.
- [[Wrangling & Sinkronisasi Multimodal]] — sinkronisasi data sensor wearable dengan sumber lain.
- [[MOC - Skripsi Repository]] — indeks utama catatan riset.

### Temuan teknis

1. Watch melaporkan tracker berikut sebagai tersedia:

   ```text
2026-10-29 17:49:27.480 11170-11170 HealthTrackerCapability com...sample.multisensormeasurement  I  supported List: [PPG_GREEN, ACCELEROMETER, ECG, PPG_IR, PPG_RED, SPO2, HEART_RATE, BIA, SWEAT_LOSS, SKIN_TEMPERATURE, SKIN_TEMPERATURE_CONTINUOUS, PPG_ON_DEMAND, PPG_CONTINUOUS, EDA_CONTINUOUS, MFBIA_ON_DEMAND]

2026-10-29 17:49:27.480 11170-11170 HealthTrackerCapability com...sample.multisensormeasurement  I  supported List : [PPG_GREEN, ACCELEROMETER, ACCELEROMETER_CONTINUOUS, ECG, ECG_ON_DEMAND, PPG_IR, PPG_RED, SPO2, SPO2_ON_DEMAND, HEART_RATE, HEART_RATE_CONTINUOUS, BIA, BIA_ON_DEMAND, SWEAT_LOSS, SKIN_TEMPERATURE, SKIN_TEMPERATURE_ON_DEMAND, SKIN_TEMPERATURE_CONTINUOUS, PPG_ON_DEMAND, PPG_CONTINUOUS, EDA_CONTINUOUS, MF_BIA_ON_DEMAND]

   ```

List lengkap:
- PPG (IR, RED, GREEN, ON_DEMAND, CONTINUOUS)
- ACC (+ CONTINUOUS)
- ECG (+ ON_DEMAND)
- SPO2 (+ ON_DEMAND)
- HR (+ CONTINOUS)
- BIA (+ BIA_ON_DEMAND, MFBIA_ON_DEMAND)
- SWEAT_LOSS
- SKIN_TEMPERATURE (+ CONTINUOUS, ON_DEMAND)
- EDA (CONTINUOUS)

2. Aplikasi berhasil membuat tracker ECG dan memasang listener:

   ```text
   getHealthTacker of type ECG_ON_DEMAND called
   createHealthTracker of trackerType ECG
   setEventListener
   ```

3. Setelah listener dipasang, service Samsung mengembalikan:

   ```text
   Exception occurred: 0
   PERMISSION_ERROR
   ```

4. Untuk BVP, aplikasi menggunakan `HealthTrackerType.PPG_GREEN` dan `ValueKey.PpgGreenSet`. BVP dapat mencapai tahap tracker/listener, tetapi permission error dapat muncul ketika body-sensor authorization belum tersedia.

### Analisis penyebab

Masalah bukan lagi pada pemilihan tracker atau deklarasi Kotlin. Capability service memang meng-advertise ECG, dan aplikasi berhasil mencapai proses `createHealthTracker`. Namun, capability list hanya menunjukkan bahwa API/hardware tracker diekspos; list tersebut tidak menjamin bahwa konfigurasi perangkat saat ini diizinkan untuk mengalirkan data ECG.

ECG dan blood pressure pada Galaxy Watch dibatasi oleh ekosistem Samsung Health/Samsung Health Monitor, termasuk kemungkinan ketergantungan pada:

- model dan firmware watch;
- region/regulatory availability;
- authorization Samsung Health/Samsung Health Monitor;
- pasangan watch dengan ponsel Samsung Galaxy yang kompatibel.

Ponsel non-Samsung dapat terhubung ke watch dan tetap membuat tracker ECG, tetapi service dapat menolak streaming dengan `PERMISSION_ERROR`. Hal ini konsisten dengan aplikasi resmi di watch yang menyatakan ECG dan blood pressure memerlukan ponsel Samsung.

### Kesimpulan

Untuk konfigurasi pengujian saat ini, implementasi aplikasi sudah mencapai batas integrasi SDK:

- `ECG_ON_DEMAND` tersedia di capability list;
- tracker ECG berhasil dibuat;
- listener berhasil dipasang;
- data ditolak oleh service dengan `PERMISSION_ERROR`;
- tidak ada permission dialog ECG terpisah yang dapat dipicu oleh aplikasi.

Hipotesis utama: ECG/BIA memerlukan otorisasi ekosistem Samsung dan/atau ponsel Samsung Galaxy, bukan sekadar `BODY_SENSORS` atau permission aplikasi biasa.

## Verifikasi berikutnya

Uji APK yang sama pada watch yang dipasangkan dengan ponsel Samsung Galaxy yang kompatibel, setelah Samsung Health dan Samsung Health Monitor diperbarui. Jika ECG menghasilkan data pada konfigurasi tersebut, batasan ponsel non-Samsung terkonfirmasi.

Jika pengujian Samsung tetap menghasilkan `PERMISSION_ERROR`, lanjutkan pemeriksaan model watch, region, firmware, versi Samsung Health Sensor SDK, dan status authorization Samsung Health Monitor.

### Perbandingan setelah memakai ponsel Samsung Galaxy

Pengujian ulang pada 29 Oktober 2026 menggunakan ponsel Samsung Galaxy tidak mengubah hasil otorisasi sensor secara berarti. Perbandingan log menunjukkan:

| Aspek | Pengujian sebelumnya | Pengujian dengan Samsung Galaxy |
|---|---|---|
| Koneksi Health Tracking Service | Berhasil | Berhasil |
| Capability list | Memuat `ECG`, `ECG_ON_DEMAND`, `PPG_GREEN` | Memuat tracker yang sama |
| Pembuatan tracker | Berhasil | Berhasil |
| Pemasangan listener | Berhasil | Berhasil |
| Akses data ECG/BVP | Ditolak | Tetap ditolak |
| Error service | `PERMISSION_ERROR` | Tetap `PERMISSION_ERROR` |

Pada log terbaru, service juga berhasil menjalankan `HEART_RATE` dan mengirim callback data. Ini menunjukkan bahwa koneksi Samsung phone bukan masalah untuk akses sensor dasar, tetapi belum membuktikan bahwa otorisasi ECG/BVP telah diprovisioning. Capability list hanya menunjukkan tracker yang diekspos API, bukan entitlement untuk streaming data pada konfigurasi perangkat tersebut.

Keterbatasan diagnosis sebelumnya adalah semua listener mengirim `MessageState.PermissionError` yang sama. Dengan perubahan implementasi, setiap tracker sekarang mencatat error secara spesifik:

```text
Starting ECG tracker: ECG_ON_DEMAND
ECG tracker error: PERMISSION_ERROR
Starting BVP tracker: PPG_GREEN
BVP tracker error: PERMISSION_ERROR
Starting heart-rate tracker: HEART_RATE_CONTINUOUS
HEART_RATE tracker error: PERMISSION_ERROR
```

Aplikasi juga menampilkan tracker yang ditolak pada pesan error, sehingga pengujian berikutnya dapat membedakan kegagalan ECG, BVP, SpO2, dan heart rate. Perubahan ini hanya diagnostik; tidak dimaksudkan sebagai bypass terhadap otorisasi Samsung Health/Samsung Health Monitor.

Hipotesis yang tersisa adalah provisioning Samsung Health Monitor, region/regulatory availability, model/firmware watch, atau otorisasi third-party ECG/BVP pada service Samsung. Ponsel Samsung tampaknya diperlukan untuk sebagian fitur resmi, tetapi pada pengujian ini belum cukup untuk menghilangkan penolakan service.

### Hasil diagnostik logging pada pengujian terbaru

Logging tracker-specific pada 29 Oktober 2026 mengonfirmasi bahwa penolakan terjadi khusus pada ECG:

```text
Starting ECG tracker: ECG_ON_DEMAND
createHealthTracker of trackerType ECG
setEventListener
Exception occurred: 0
ECG tracker error: PERMISSION_ERROR
```

Urutan tersebut terjadi dua kali. Sementara itu, heart rate berhasil dibuat dan mengirim data callback:

```text
Starting heart-rate tracker: HEART_RATE_CONTINUOUS
createHealthTracker of trackerType HEART_RATE
Data received from Tracker Service and sending data ...
```

Tidak ada `Starting BVP tracker` atau `BVP tracker error` pada potongan log terbaru, sehingga hasil ini belum menguji BVP dalam sesi yang sama. SpO2 juga berhasil dibuat kemudian. Dengan demikian, log terbaru mempersempit masalah: koneksi service dan akses heart rate berfungsi, tetapi service menolak tracker ECG setelah tracker dibuat dan listener dipasang. Tidak ada indikasi bahwa aplikasi gagal menemukan `ECG_ON_DEMAND`.

Capability list tetap memuat `ECG` dan `ECG_ON_DEMAND`, tetapi entitlement streaming ECG tetap ditolak. Tidak adanya setup authorization di watch dan kegagalan yang tetap terjadi setelah dipasangkan ke Samsung Galaxy memperkuat hipotesis bahwa akses ECG dikontrol oleh provisioning Samsung Health/Samsung Health Monitor, region/regulatory policy, model/firmware, atau authorization third-party—bukan permission Android biasa.

### Klarifikasi BVP versus ECG dan implementasi kontrol BVP

BVP dan ECG bukan sensor yang sama dan bukan satu tombol yang sama. Pada aplikasi, BVP menggunakan `HealthTrackerType.PPG_GREEN`, sedangkan ECG menggunakan `HealthTrackerType.ECG_ON_DEMAND`/tracker ECG. Nilai BVP memang sudah diproses melalui `bvpTrackerListener` dan ditampilkan pada `ResultColumn`, tetapi sebelumnya tidak ada kontrol UI untuk memulai dan menghentikan tracker BVP secara manual. BVP juga sempat dimulai otomatis bersama heart rate, sehingga error BVP sulit dibedakan dari error sensor lain.

Aplikasi kini memiliki tombol terpisah **Start BVP/Stop**. Tombol tersebut:

- memeriksa `BODY_SENSORS`;
- memulai `PPG_GREEN` melalui `MainViewModel.startBvpTracking()`;
- menghentikan tracker melalui `stopBvpTracking()`;
- menampilkan nilai BVP yang sudah tersedia pada layar utama;
- menghasilkan logging tracker-specific (`Starting BVP tracker: PPG_GREEN` dan `BVP tracker error: ...`).

Dengan demikian, pengujian berikutnya dapat memisahkan apakah BVP dapat digunakan walaupun ECG ditolak oleh service Samsung.

### Verifikasi BVP berhasil dan hasilnya sama dengan ECG

Pengujian terbaru akhirnya memulai BVP secara definitif. Log menunjukkan:

```text
Starting BVP tracker: PPG_GREEN
getHealthTacker of type PPG_GREEN called
createHealthTracker of trackerType PPG_GREEN
setEventListener
Exception occurred: 0
BVP tracker error: PERMISSION_ERROR
```

Urutan ini terjadi dua kali. ECG pada sesi yang sama juga mengalami kegagalan identik:

```text
Starting ECG tracker: ECG_ON_DEMAND
createHealthTracker of trackerType ECG
setEventListener
Exception occurred: 0
ECG tracker error: PERMISSION_ERROR
```

Heart rate tetap berhasil menerima callback data. BVP bukan lagi sekadar belum teruji: service Samsung memang menolak akses BVP/`PPG_GREEN` setelah tracker dibuat. Karena BVP dan ECG gagal pada tahap service yang sama, sedangkan heart rate berfungsi, temuan ini memperkuat dugaan adanya pembatasan bersama untuk sensor optik/ECG tertentu—misalnya provisioning Samsung Health/Samsung Health Monitor, region/regulatory policy, model/firmware, atau entitlement third-party.

Baris `unsetListener of trackerType PPG_GREEN` juga menunjukkan bahwa aplikasi membersihkan tracker setelah kegagalan, bukan crash sebelum tracker dimulai.

### Implikasi untuk skripsi

- Jangan menyimpulkan bahwa ECG tidak didukung hanya dari capability list.
- Jangan menyebutnya sebagai bug permission Android biasa; error terjadi setelah tracker dibuat dan listener dipasang.
- Catat `PERMISSION_ERROR` sebagai batasan perangkat/ekosistem pada eksperimen.
- BVP/PPG hijau tetap dapat diperlakukan sebagai jalur sensor yang terpisah dari ECG/BIA.
- Untuk validasi model, pastikan data BVP/ECG yang terkumpul benar-benar memiliki timestamp dan status sensor yang valid sebelum preprocessing.

### Referensi internal kode

- [`MainActivity.kt`](../Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/presentation/MainActivity.kt)
- [`TrackingManager.kt`](../Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/tracking/TrackingManager.kt)
- [`Bvp.kt`](../Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/Bvp.kt)
- [`Ecg.kt`](../Deployment/test1/app/src/main/java/com/samsung/health/sensorsdksample/multisensormeasurement/data/Ecg.kt)
