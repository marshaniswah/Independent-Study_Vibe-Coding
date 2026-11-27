
# SPECIFICATION
## Student Graduation Prediction — Kelompok B

### 1. Project Overview
Student Graduation Prediction merupakan aplikasi berbasis Machine Learning untuk memprediksi kelulusan siswa berdasarkan data akademik.

Pengembangan Kelompok B menggunakan pendekatan Spec-First, yaitu menyusun spesifikasi proyek sebelum melakukan implementasi kode dengan bantuan AI melalui Antigravity IDE.

### 2. Problem Statement
Proses identifikasi siswa yang berpotensi tidak lulus dapat membantu pihak pendidikan menentukan siswa yang membutuhkan perhatian atau pendampingan tambahan.

Oleh karena itu, diperlukan aplikasi yang dapat memprediksi status kelulusan berdasarkan data akademik siswa.

### 3. Project Objectives
- Mengembangkan aplikasi prediksi kelulusan siswa.
- Menerapkan Machine Learning menggunakan Python.
- Membandingkan hasil prediksi dengan data aktual.
- Menyediakan antarmuka sederhana untuk melakukan prediksi.

### 4. Target Users
- Guru
- Pihak sekolah
- Pengelola data akademik

### 5. Dataset Description

Dataset: `data/siswa.csv`

Kolom dataset:
- id_siswa: Identitas siswa.
- program: Program atau jurusan siswa.
- jam_belajar: Jumlah jam belajar.
- kehadiran: Persentase atau nilai kehadiran.
- nilai_uts: Nilai ujian tengah semester.
- nilai_tugas: Nilai tugas.
- lulus: Target status kelulusan.
- nilai_remedial: Nilai remedial siswa.

Target:
- 0 = Tidak lulus
- 1 = Lulus

### 6. Functional Requirements
- Membaca dataset siswa.
- Melakukan pemeriksaan dan preprocessing data.
- Melakukan prediksi kelulusan siswa.
- Menampilkan hasil prediksi.
- Menampilkan evaluasi performa model.
- Menyediakan antarmuka menggunakan Gradio.

### 7. User Flow
1. Pengguna membuka aplikasi.
2. Pengguna memasukkan data akademik siswa.
3. Sistem memproses input.
4. Model melakukan prediksi.
5. Sistem menampilkan hasil prediksi kelulusan.

### 8. Development Plan
1. Menyusun spesifikasi proyek.
2. Melakukan dataset audit.
3. Melakukan Exploratory Data Analysis (EDA).
4. Melakukan preprocessing.
5. Membangun baseline model Logistic Regression.
6. Mengevaluasi model.
7. Mengembangkan antarmuka Gradio.
8. Melakukan pengujian dan audit kode.

### 9. Model Evaluation
Model akan dievaluasi menggunakan:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 10. Data Leakage Risk
Kolom `nilai_remedial` perlu diperiksa lebih lanjut untuk mengetahui apakah informasi tersebut tersedia sebelum prediksi dilakukan atau justru muncul setelah proses remedial.

Kolom `id_siswa` juga tidak langsung digunakan sebagai fitur prediksi karena merupakan identifier.

### 11. Technical Requirements
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Gradio

### 12. Development Approach
Kelompok B menggunakan pendekatan Spec-First. Setiap fitur dikembangkan secara bertahap berdasarkan spesifikasi, menggunakan bantuan LLM melalui Antigravity IDE.

Setiap prompt, hasil kode, error, dan perbaikan akan didokumentasikan dalam prompt log.

### 13. Project Limitations
- Prediksi hanya berdasarkan fitur yang tersedia pada dataset.
- Hasil prediksi bukan keputusan kelulusan resmi.
- Performa model bergantung pada kualitas dan karakteristik dataset.
