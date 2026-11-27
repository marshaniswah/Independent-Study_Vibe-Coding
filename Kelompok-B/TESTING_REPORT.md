# Testing Report – Student Graduation Prediction
**Kelompok B | Pendekatan Spec-First**

## 1. Tujuan Pengujian

Pengujian dilakukan untuk memastikan aplikasi Student Graduation Prediction dapat menerima input data siswa, menghasilkan prediksi kelulusan, dan menolak input yang tidak sesuai dengan ketentuan validasi.

## 2. Lingkungan Pengujian

| Komponen | Keterangan |
|---|---|
| Aplikasi | Student Graduation Prediction |
| Framework antarmuka | Gradio |
| Model | Logistic Regression |
| Platform | Windows |
| Alamat lokal | http://127.0.0.1:7860 |
| Modul aplikasi | `src/app.py` |

## 3. Skenario dan Hasil Pengujian

### Test Case 1 – Input Valid

**Data pengujian:**

| Parameter | Nilai |
|---|---:|
| Jam Belajar | 8.5 |
| Kehadiran | 90 |
| Nilai UTS | 82 |
| Nilai Tugas | 85 |
| Program | IPA |

**Hasil yang diharapkan:**
- Aplikasi menerima seluruh input.
- Model menghasilkan prediksi kelulusan.
- Aplikasi menampilkan probabilitas prediksi.

**Hasil aktual:**
- Prediksi: Lulus
- Probabilitas Lulus: 79,91%
- Tidak ditemukan pesan kesalahan.

**Status: BERHASIL**

### Test Case 2 – Kehadiran Tidak Valid

**Data pengujian:**

| Parameter | Nilai |
|---|---:|
| Jam Belajar | 8 |
| Kehadiran | -10 |
| Nilai UTS | 82 |
| Nilai Tugas | 85 |
| Program | IPA |

**Hasil yang diharapkan:**

Aplikasi menolak input karena nilai kehadiran berada di bawah rentang 0–100.

**Hasil aktual:**

Aplikasi menampilkan status `Input Tidak Valid` dan pesan kesalahan. Prediksi tidak dilakukan.

**Status: BERHASIL**

### Test Case 3 – Nilai UTS Tidak Valid

**Data pengujian:**

| Parameter | Nilai |
|---|---:|
| Jam Belajar | 8 |
| Kehadiran | 90 |
| Nilai UTS | 150 |
| Nilai Tugas | 85 |
| Program | IPA |

**Hasil yang diharapkan:**

Aplikasi menolak input karena nilai UTS melebihi rentang 0–100.

**Hasil aktual:**

Aplikasi menampilkan status `Input Tidak Valid` dan pesan kesalahan. Prediksi tidak dilakukan.

**Status: BERHASIL**

## 4. Rekapitulasi Pengujian

| No. | Skenario | Hasil | Status |
|---|---|---|---|
| 1 | Seluruh input valid | Prediksi Lulus, probabilitas 79,91% | Berhasil |
| 2 | Kehadiran -10 | Input Tidak Valid | Berhasil |
| 3 | Nilai UTS 150 | Input Tidak Valid | Berhasil |

**Total pengujian:** 3 test case

**Pengujian berhasil:** 3 test case

**Pengujian gagal:** 0 test case

### Pengujian Tambahan pada Aplikasi Gradio

| No | Skenario | Input | Hasil Aktual | Status |
|---|---|---|---|---|
| 1 | Prediksi Lulus | Jam belajar 8.5, kehadiran 90, UTS 85, tugas 82, IPA | Lulus (80.86%) | Berhasil |
| 2 | Prediksi Tidak Lulus | Jam belajar 2, kehadiran 45, UTS 35, tugas 40, IPS | Tidak Lulus (1.90%) | Berhasil |
| 3 | Validasi Input | Kehadiran -10, UTS 150 | Input Tidak Valid | Berhasil |

## 5. Kesimpulan

Berdasarkan tiga pengujian yang telah dilakukan, aplikasi Student Graduation Prediction berhasil menjalankan fungsi prediksi pada input yang valid dan menolak input yang tidak memenuhi ketentuan validasi.

Hasil pengujian menunjukkan bahwa antarmuka Gradio dan mekanisme validasi input telah berfungsi sesuai skenario pengujian.

Pengujian ini masih terbatas pada validasi input dan satu contoh prediksi. Hasil tersebut belum membuktikan bahwa model memiliki akurasi yang tinggi pada seluruh kondisi data siswa. Evaluasi performa model tetap mengacu pada `model_evaluation_report.txt`.