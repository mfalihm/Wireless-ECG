# Wireless ECG Monitoring System

Aplikasi desktop berbasis Python untuk pemantauan elektrokardiogram (ECG) nirkabel secara real-time. Aplikasi ini dirancang untuk memvisualisasikan sinyal 3-Lead (Lead I, Lead II, Lead III), menghitung detak jantung (BPM), serta mendeteksi gejala Sleep Apnea menggunakan model Deep Learning (CNN).

## 🌟 Fitur Utama

* **Real-time Monitoring:** Visualisasi sinyal ECG (Lead I, II, III) dengan responsivitas tinggi.
* **Signal Processing:** Filter digital Bandpass (0.5 - 40Hz) dan Notch filter untuk membersihkan noise.
* **Heart Rate Calculation:** Deteksi puncak R-peak otomatis untuk menghitung BPM.
* **AI Integration:** Prediksi Sleep Apnea real-time menggunakan model CNN (input buffer 6000 sampel).
    > *Catatan: Fitur AI Integration masih dalam tahap pengembangan.*
* **Data Logging:**
    * Penyimpanan data mentah ke format `.csv`.
    * Penyimpanan data terolah ke format `.csv`.
    * Penyimpanan data anotasi AI ke format `.csv`.
    * Pembuatan laporan medis otomatis ke format `.docx` (Word).
* **Konektivitas:** Mendukung komunikasi serial (USB/Bluetooth) dengan baudrate yang dapat disesuaikan pada kode.

## 🛠️ Requirements (Prasyarat Sistem)

* Program ini berjalan pada Python 3.10 - 3.11 (versi 3.12+ belum disarankan karena kompatibilitas TensorFlow).
* Disarankan menggunakan virtual environment.

### Dependencies Utama
Daftar library yang digunakan dalam proyek ini:
* `PyQt6` (Antarmuka pengguna)
* `numpy` (Komputasi numerik)
* `scipy` (Pemrosesan sinyal digital)
* `pyserial` (Komunikasi serial)
* `pyqtgraph` (Plotting grafik performa tinggi)
* `tensorflow` (Inference model AI)
* `python-docx` (Pembuatan laporan Word)
* `pandas` (Manipulasi data CSV)

### ℹ️ Versi Referensi (Tested Environment)
Aplikasi ini terakhir dikembangkan dan diuji stabil pada lingkungan berikut (Januari 2026). Informasi ini disertakan sebagai referensi debug jika terjadi kendala kompatibilitas di masa depan:

| Library | Versi Pengembangan |
| :--- | :--- |
| **Python** | 3.11.0 |
| **PyQt6** | 6.10.1 |
| **NumPy** | 2.2.6 |
| **SciPy** | 1.16.3 |
| **TensorFlow** | 2.20.0 |
| **PyQtGraph** | 0.14.0 |
| **Python-docx** | 1.2.0 |

> *Catatan: Pengguna **tidak wajib** menginstal versi persis di atas. File `requirements.txt` sudah dikonfigurasi agar `pip` otomatis memilih versi yang kompatibel dengan sistem Anda.*

## 📦 Instalasi

1.  **Clone atau Download** repository ini.
2.  **Buka Terminal/CMD** dan arahkan ke direktori folder proyek ini.
3.  **Buat Virtual Environment** (disarankan agar instalasi bersih):
    ```bash
    python -m venv venv
    
    # Aktivasi di Windows:
    venv\Scripts\activate
    
    # Aktivasi di Mac/Linux/Raspberry Pi:
    source venv/bin/activate
    ```
4.  **Install Dependencies**:
    Jalankan perintah berikut untuk menginstal seluruh library yang dibutuhkan secara otomatis:
    ```bash
    pip install -r requirements.txt
    ```

### ⚠️ Khusus Pengguna Raspberry Pi (Linux)
Jika terjadi error saat menjalankan aplikasi atau serial tidak terdeteksi, lakukan langkah berikut di terminal (di luar venv):

1.  **Beri Izin Akses Serial** (Wajib):
    ```bash
    sudo usermod -a -G dialout $USER
    ```
    *Setelah itu, Restart Raspberry Pi Anda.*

2.  **Install Library Tambahan** (Jika PyQt gagal berjalan):
    ```bash
    sudo apt-get install libxcb-xinerama0 libgl1-mesa-glx
    ```

## 🚀 Cara Penggunaan

### 1. Persiapan Hardware & Koneksi
* Pastikan modul ECG (ESP32 + AD8232/CJMCU) sudah menyala.
* **Untuk Pengguna Windows:**
    * Lakukan **Pairing** Bluetooth di menu Settings Windows.
    * Cek di Device Manager untuk mengetahui nomor COM Port (misal: `COM3` atau `COM4`).
* **Untuk Pengguna Raspberry Pi:**
    * Lakukan Pairing via terminal (`bluetoothctl`).
    * Buat Virtual Port dengan perintah: `sudo rfcomm bind 0 [MAC_ADDRESS]`.
    * Port akan terdeteksi sebagai `/dev/rfcomm0`.

### 2. Jalankan Aplikasi
```bash
# Windows
python main.py

# Raspberry Pi / Linux
python3 main.py