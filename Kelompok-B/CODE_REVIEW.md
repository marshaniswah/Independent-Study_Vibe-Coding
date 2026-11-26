# Code Review — Kelompok B

Berdasarkan pemeriksaan kode contoh AI pada tugas, ditemukan empat kesalahan berikut:

| No. | Kesalahan | Dampak | Perbaikan |
|---|---|---|---|
| 1 | Missing value diisi dengan `fillna(0)` tanpa mempertimbangkan karakteristik setiap fitur. | Nilai nol dapat mengubah distribusi data dan menimbulkan bias. | Menggunakan `SimpleImputer` dengan median untuk fitur numerik dan modus untuk fitur kategorikal. |
| 2 | `StandardScaler` diterapkan sebelum pembagian data training dan testing. | Menyebabkan data leakage karena informasi data testing ikut memengaruhi preprocessing. | Memasukkan scaler ke dalam Pipeline agar hanya di-fit pada data training. |
| 3 | Pembagian data tidak menggunakan `stratify` dan `random_state`. | Proporsi kelas dapat berubah dan hasil eksperimen sulit direproduksi. | Menggunakan `train_test_split` dengan `stratify=y` dan `random_state=42`. |
| 4 | `id_siswa` dan `nilai_remedial` digunakan sebagai fitur prediksi. | ID tidak memberikan informasi akademik yang relevan, sedangkan nilai remedial berpotensi menyebabkan target leakage. | Mengecualikan kedua kolom dari fitur model. |

Keempat perbaikan diterapkan pada implementasi Kelompok B dengan pendekatan spec-first.