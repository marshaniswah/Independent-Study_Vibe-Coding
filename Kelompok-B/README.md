# Student Graduation Prediction — Kelompok B

## 1. Deskripsi Proyek
Aplikasi **Student Graduation Prediction** merupakan solusi berbasis Machine Learning untuk memprediksi apakah seorang siswa akan lulus atau tidak berdasarkan data akademik. Proyek ini dikembangkan dengan pendekatan **Spec‑First** oleh Kelompok B.

## 2. Latar Belakang dan Tujuan
- **Latar Belakang**: Menentukan siswa yang berpotensi tidak lulus dapat membantu guru dan pihak sekolah memberikan pendampingan tambahan secara tepat waktu.
- **Tujuan**:
  - Mengembangkan model prediksi kelulusan menggunakan Logistic Regression.
  - Menyediakan antarmuka web sederhana dengan Gradio untuk melakukan prediksi secara interaktif.
  - Memastikan proses preprocessing data bebas dari *data leakage*.

## 3. Fitur Aplikasi
- Memuat model Logistic Regression yang sudah dilatih.
- Input lima fitur akademik: jam belajar, kehadiran, nilai UTS, nilai tugas, dan program studi.
- Menampilkan hasil prediksi (**Lulus** atau **Tidak Lulus**) beserta probabilitas kelulusan.
- Validasi input otomatis (rentang nilai, program yang valid, dll.).
- Menyajikan interpretasi bahwa hasil hanyalah prediksi model, bukan keputusan resmi.

## 4. Dataset dan Atribut
Dataset berada di `data/siswa.csv` dengan kolom:
- `id_siswa` – identifier (tidak dipakai sebagai fitur).
- `program` – jurusan siswa (IPA, IPS, BAHASA) – akan dinormalisasi.
- `jam_belajar` – jumlah jam belajar siswa.
- `kehadiran` – persentase kehadiran.
- `nilai_uts` – nilai ujian tengah semester.
- `nilai_tugas` – nilai tugas.
- `lulus` – target (0 = Tidak Lulus, 1 = Lulus).
- `nilai_remedial` – *target leakage*; tidak dipakai dalam model.

## 5. Metodologi
1. **Preprocessing** (`src/preprocessing.py`)
   - Membaca dataset secara read‑only.
   - Validasi kehadiran semua kolom wajib dan nilai target.
   - Menghapus **exact duplicate rows** dengan `drop_duplicates()`.
   - Normalisasi kolom `program` (strip & uppercase).
   - Menentukan fitur numerik (`jam_belajar`, `kehadiran`, `nilai_uts`, `nilai_tugas`) dan kategorikal (`program`).
   - Membagi data menjadi train‑test secara stratified (`train_test_split`) dengan `test_size=0.2`, `random_state=42`.
2. **Model** (`src/train_model.py`)
   - Pipeline menggabungkan preprocessing (imputer, scaler, one‑hot encoder) dan **Logistic Regression** (`random_state=42`, `max_iter=1000`).
   - Model dilatih pada data training dan dievaluasi pada data testing.
3. **Evaluasi** (`src/train_model.py`)
   - Menghitung Accuracy, Precision, Recall, F1‑Score untuk kedua kelas, Confusion Matrix, dan Classification Report.
   - Hasil evaluasi disimpan di `outputs/model_evaluation_report.txt`.
4. **Aplikasi Gradio** (`src/app.py`)
   - Memuat model terlatih (`outputs/logistic_regression_model.joblib`).
   - Memvalidasi input, membangun DataFrame dengan urutan kolom yang sama, melakukan prediksi, dan menampilkan hasil serta interpretasi.

## 6. Hasil Evaluasi Model
| Metrik | Nilai |
|--------|-------|
| **Accuracy** | 67.50 % |
| **Precision (Tidak Lulus)** | 0.6774 |
| **Recall (Tidak Lulus)** | 0.4200 |
| **F1‑Score (Tidak Lulus)** | 0.5185 |
| **Precision (Lulus)** | 0.6742 |
| **Recall (Lulus)** | 0.8571 |
| **F1‑Score (Lulus)** | 0.7547 |

*Catatan*: Evaluasi ini menggunakan data testing yang tidak terkontaminasi oleh leakage.

## 7. Struktur Folder Proyek
```
Kelompok-B-Vibe-Coding/
│   README.md               # <--- file ini
│   SPEC.md
│   EDA_REPORT.md
│   requirements.txt
│
├───data/
│       siswa.csv
│
├───src/
│       app.py
│       preprocessing.py
│       train_model.py
│       data_audit.py
│
└───outputs/
        data_audit_report.txt
        model_evaluation_report.txt
        logistic_regression_model.joblib
        ... (visualisasi PNG)
```

## 8. Persyaratan Instalasi dan Cara Menjalankan
1. Pastikan Python 3 terinstal.
2. Install semua dependensi dari file `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
3. **Latih model** (jika belum ada) dengan menjalankan:
   ```bash
   python src/train_model.py
   ```
   Ini akan menghasilkan `outputs/logistic_regression_model.joblib` dan laporan evaluasi.
4. Jalankan aplikasi Gradio:
   ```bash
   python src/app.py
   ```
   Aplikasi akan tersedia pada alamat localhost (biasanya `http://127.0.0.1:7860`).

## 9. Cara Menggunakan Aplikasi
1. Buka URL yang ditampilkan di terminal setelah menjalankan `src/app.py`.
2. Masukkan nilai untuk:
   - **Jam Belajar** (angka, ≥ 0)
   - **Kehadiran** (0‑100 %)
   - **Nilai UTS** (0‑100)
   - **Nilai Tugas** (0‑100)
   - **Program** (dropdown: IPA, IPS, BAHASA)
3. Klik **“Prediksi Kelulusan”**.
4. Hasil akan menampilkan:
   - Status prediksi (**Lulus** / **Tidak Lulus**)
   - Probabilitas kelulusan dalam persen
   - Pesan interpretasi bahwa hasil adalah prediksi model.

## 10. Keterbatasan Model
- Model hanya menggunakan fitur numerik dan `program`; kolom `id_siswa` dan `nilai_remedial` tidak dipakai karena risiko *identifier* dan *target leakage*.
- Performanya bergantung pada kualitas data dan distribusi kelas yang masih sedikit tidak seimbang.
- Probabilitas yang dihasilkan mencerminkan kepercayaan model pada data yang serupa dengan data training, bukan jaminan kelulusan sebenarnya.
- Tidak ada mekanisme interpretasi fitur (mis. SHAP) dalam versi saat ini.

---
*Dokumen ini disusun berdasarkan SPEC.md, EDA_REPORT.md, kode sumber, dan hasil evaluasi aktual.*
