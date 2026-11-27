# Laporan Data Audit dan Exploratory Data Analysis (EDA)
**Proyek:** Student Graduation Prediction — Kelompok B  
**Pendekatan:** Spec-First  
**Sumber Data:** [data/siswa.csv](data/siswa.csv)  
**Skrip Audit:** [src/data_audit.py](src/data_audit.py)  
**Laporan Audit Aktual:** [outputs/data_audit_report.txt](outputs/data_audit_report.txt)

---

## 1. Dataset Overview

Dataset [data/siswa.csv](data/siswa.csv) berisi data akademik siswa yang digunakan sebagai dasar pengembangan model prediksi kelulusan pada proyek **Student Graduation Prediction — Kelompok B** sebagaimana tercantum dalam [SPEC.md](SPEC.md). Berdasarkan hasil audit pada [outputs/data_audit_report.txt](outputs/data_audit_report.txt), dataset memiliki dimensi sebagai berikut:

- **Jumlah Baris:** `615` baris
- **Jumlah Kolom:** `8` kolom

Rincian struktur kolom, tipe data hasil pembacaan `pandas`, jumlah data non-kosong (*non-null*), dan deskripsinya disajikan pada tabel berikut:

| No | Nama Kolom | Tipe Data (`dtype`) | Non-Null Count | Deskripsi |
|---:|---|---|---:|---|
| 1 | `id_siswa` | `str` | 615 | Identifier siswa |
| 2 | `program` | `str` | 615 | Program atau jurusan siswa (`IPA`, `IPS`, `BAHASA`) |
| 3 | `jam_belajar` | `float64` | 579 | Jumlah jam belajar siswa |
| 4 | `kehadiran` | `float64` | 615 | Persentase atau nilai kehadiran siswa |
| 5 | `nilai_uts` | `float64` | 591 | Nilai Ujian Tengah Semester (UTS) |
| 6 | `nilai_tugas` | `float64` | 615 | Nilai tugas siswa |
| 7 | `lulus` | `int64` | 615 | Target status kelulusan (`0` = Tidak Lulus, `1` = Lulus) |
| 8 | `nilai_remedial` | `float64` | 251 | Nilai remedial siswa |

**Catatan Identifier:** Kolom `id_siswa` berfungsi sebagai identifier siswa, tetapi bukan identifier yang unik pada seluruh baris saat ini karena hanya terdapat `600` nilai unik dari `615` baris akibat adanya duplikasi.

---

## 2. Missing Value Analysis

Pemeriksaan nilai hilang (*missing values*) terhadap `615` baris dataset menunjukkan tiga kolom memiliki data kosong, sedangkan lima kolom lainnya terisi penuh:

| Kolom | Jumlah Missing | Persentase Missing (%) | Jumlah Terisi |
|---|---:|---:|---:|
| `id_siswa` | 0 | 0.00% | 615 |
| `program` | 0 | 0.00% | 615 |
| `jam_belajar` | 36 | 5.85% | 579 |
| `kehadiran` | 0 | 0.00% | 615 |
| `nilai_uts` | 24 | 3.90% | 591 |
| `nilai_tugas` | 0 | 0.00% | 615 |
| `lulus` | 0 | 0.00% | 615 |
| `nilai_remedial` | 364 | 59.19% | 251 |

**Interpretasi:**
- Kolom fitur akademik `jam_belajar` (`5.85%`) dan `nilai_uts` (`3.90%`) memiliki proporsi *missing value* yang relatif kecil dan masih dapat ditangani pada tahap *preprocessing* melalui metode imputasi yang sesuai.
- Kolom `nilai_remedial` memiliki *missing value* sebesar `59.19%` (`364` dari `615` baris). Kekosongan ini bukan bersifat acak, melainkan berkaitan langsung dengan struktur target kelulusan sebagaimana dibahas pada bagian **Data Leakage Analysis**.

---

## 3. Duplicate Analysis

Pemeriksaan duplikasi dilakukan pada level seluruh baris (*exact duplicates*) dan pada kolom identifier `id_siswa`:

- **Jumlah Baris Duplikat Penuh (*Exact Duplicates*):** `15` baris
- **Jumlah Duplikat pada `id_siswa`:** `15` entri
- **Jumlah Nilai Unik `id_siswa`:** `600` nilai unik dari `615` total baris
- **Contoh `id_siswa` pada Baris Duplikat:** `S0000`, `S0023`, `S0066`, `S0165`, `S0201`, `S0232`, `S0329`, `S0334`, `S0404`, `S0415`

**Interpretasi:**  
Kolom `id_siswa` merupakan identifier siswa, tetapi bukan identifier yang unik pada seluruh baris saat ini karena hanya terdapat `600` nilai unik dari `615` baris akibat adanya duplikasi. Jumlah baris duplikat penuh (`15`) sama persis dengan jumlah duplikasi pada `id_siswa` (`15`), yang mengindikasikan adanya baris rekaman yang terulang identik di dalam dataset. Kehadiran baris duplikat berisiko menyebabkan bias bobot pada proses pelatihan serta menimbulkan kebocoran data evaluasi (*data leakage*) apabila baris yang identik terbagi ke dalam *training set* dan *test set*. Meskipun demikian, setiap baris duplikat perlu diperiksa terlebih dahulu sebelum dihapus untuk memastikan penyebab kemunculannya.

---

## 4. Categorical Analysis

Analisis fitur kategorikal berfokus pada kolom `program` (sedangkan `id_siswa` merupakan kolom identifier yang tidak diperlakukan sebagai fitur kategorikal prediktif).

Pada data mentah, kolom `program` terdeteksi memiliki **6 nilai unik** akibat inkonsistensi huruf kapital dan adanya spasi di akhir teks (*trailing whitespace*):

| Nilai Mentah `program` | Jumlah Baris |
|---|---:|
| `BAHASA` | 197 |
| `IPA` | 195 |
| `IPS` | 181 |
| `ipa ` | 17 |
| `ips ` | 14 |
| `bahasa ` | 11 |

Temuan inkonsistensi format pada kolom `program`: `['ipa ', 'ips ', 'bahasa ']`.

Ketika dilakukan simulasi normalisasi di memori menggunakan fungsi `strip()` dan `upper()` (tanpa mengubah dataset asli), kategori menyatu menjadi **3 nilai unik** yang konsisten dengan distribusi dan tingkat kelulusan sebagai berikut:

| Program (Ternormalisasi) | Jumlah Siswa | Tidak Lulus (`0`) | Lulus (`1`) | Tingkat Kelulusan (%) |
|---|---:|---:|---:|---:|
| `IPA` | 212 | 96 | 116 | 54.72% |
| `BAHASA` | 208 | 80 | 128 | 61.54% |
| `IPS` | 195 | 75 | 120 | 61.54% |
| **Total** | **615** | **251** | **364** | **59.19%** |

Visualisasi tingkat kelulusan per program tersimpan pada [outputs/hubungan_program_vs_lulus.png](outputs/hubungan_program_vs_lulus.png). Tanpa normalisasi teks pada tahap *preprocessing*, model akan menganggap `'IPA'` dan `'ipa '` sebagai dua kategori yang berbeda.

---

## 5. Target Distribution

Distribusi kolom target `lulus` pada `615` baris data adalah sebagai berikut:

| Kelas Target (`lulus`) | Label | Jumlah Siswa | Persentase (%) |
|---:|---|---:|---:|
| `0` | Tidak Lulus | 251 | 40.81% |
| `1` | Lulus | 364 | 59.19% |

Visualisasi distribusi target tersimpan pada [outputs/distribusi_target_lulus.png](outputs/distribusi_target_lulus.png).

**Interpretasi:**  
Proporsi kelas target antara siswa yang lulus (`59.19%`) dan tidak lulus (`40.81%`) tergolong cukup berimbang (*mildly imbalanced*), dengan selisih `18.38` poin persentase. Kondisi ini memadai untuk pemodelan klasifikasi biner standar, namun evaluasi performa model tetap perlu memperhatikan metrik selain *Accuracy*—seperti *Precision*, *Recall*, *F1-Score*, dan *Confusion Matrix*—sebagaimana tercantum dalam [SPEC.md](SPEC.md).

---

## 6. Numerical Feature Analysis

### 6.1 Statistik Deskriptif

Ringkasan statistik deskriptif seluruh kolom numerik dan target pada dataset disajikan pada tabel di bawah ini:

| Kolom | Count | Mean | Std | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `jam_belajar` | 579.0 | 7.41 | 3.26 | 1.1 | 5.1 | 7.0 | 9.3 | 23.2 |
| `kehadiran` | 615.0 | 83.31 | 9.50 | 55.0 | 77.0 | 84.0 | 90.0 | 100.0 |
| `nilai_uts` | 591.0 | 70.42 | 11.97 | 33.0 | 62.5 | 71.0 | 78.0 | 100.0 |
| `nilai_tugas` | 615.0 | 75.40 | 9.91 | 46.0 | 68.0 | 75.0 | 83.0 | 100.0 |
| `lulus` | 615.0 | 0.59 | 0.49 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| `nilai_remedial` | 251.0 | 59.94 | 8.20 | 39.0 | 54.0 | 60.0 | 65.0 | 88.0 |

*(Grafik sebaran distribusi dan boxplot fitur numerik prediktif tersedia pada [outputs/distribusi_fitur_numerik.png](outputs/distribusi_fitur_numerik.png).)*

### 6.2 Hubungan Awal Fitur Numerik Prediktif dengan Target `lulus`

Dengan mengecualikan `id_siswa` dan `nilai_remedial`, perbandingan statistik fitur numerik prediktif (`jam_belajar`, `kehadiran`, `nilai_uts`, `nilai_tugas`) berdasarkan status kelulusan adalah sebagai berikut:

| Fitur | Kelas `lulus = 0` (Tidak Lulus) | Kelas `lulus = 1` (Lulus) | Korelasi Pearson dengan `lulus` |
|---|---|---|---:|
| `nilai_uts` | n=245, mean=`67.28`, median=`68.0`, std=`11.77` | n=346, mean=`72.65`, median=`73.0`, std=`11.61` | `0.221` |
| `jam_belajar` | n=238, mean=`6.72`, median=`6.2`, std=`3.13` | n=341, mean=`7.89`, median=`7.4`, std=`3.27` | `0.177` |
| `nilai_tugas` | n=251, mean=`73.83`, median=`74.0`, std=`10.06` | n=364, mean=`76.48`, median=`77.0`, std=`9.66` | `0.131` |
| `kehadiran` | n=251, mean=`81.94`, median=`82.0`, std=`9.61` | n=364, mean=`84.26`, median=`85.0`, std=`9.32` | `0.120` |

Visualisasi hubungan fitur numerik dengan target tersedia pada [outputs/hubungan_fitur_numerik_vs_lulus.png](outputs/hubungan_fitur_numerik_vs_lulus.png) dan matriks korelasi pada [outputs/korelasi_fitur_numerik.png](outputs/korelasi_fitur_numerik.png).

**Interpretasi:**
- Secara deskriptif, rata-rata dan median keempat fitur prediktif (`nilai_uts`, `jam_belajar`, `nilai_tugas`, dan `kehadiran`) pada kelompok siswa yang lulus (`lulus = 1`) lebih tinggi dibandingkan kelompok tidak lulus (`lulus = 0`).
- Seluruh fitur prediktif menunjukkan korelasi Pearson positif lemah terhadap `lulus`, dengan nilai korelasi tertinggi pada `nilai_uts` (`0.221`) dan terendah pada `kehadiran` (`0.120`).
- **Catatan Penting (Korelasi Bukan Kausalitas):** Nilai korelasi hanya mengukur tingkat asosiasi linear antarvariabel dalam data historis yang diamati. **Korelasi tidak menunjukkan hubungan sebab-akibat (*correlation does not imply causation*)**; tingginya nilai suatu fitur belum tentu secara langsung menyebabkan kelulusan karena terdapat kemungkinan pengaruh faktor lain maupun interaksi antarfaktor akademik yang tidak tercatat dalam dataset.

---

## 7. Data Leakage Analysis

Audit khusus dilakukan terhadap kolom `nilai_remedial` untuk memeriksa risiko *target leakage* sesuai Bagian 10 pada [SPEC.md](SPEC.md). Hasil tabulasi silang antara keberadaan nilai `nilai_remedial` dengan target `lulus` adalah sebagai berikut:

| Status `nilai_remedial` | Lulus (`1`) | Tidak Lulus (`0`) | Total |
|---|---:|---:|---:|
| Missing (`NaN`) | 364 | 0 | 364 |
| Terisi (`Non-NaN`) | 0 | 251 | 251 |
| **Total** | **364** | **251** | **615** |

Ringkasan *missingness* dan statistik `nilai_remedial` per kelas target:
- **Kelas `lulus = 0` (Tidak Lulus):** dari `251` baris, seluruh `251` baris terisi (`0` missing / `0.0%` missing), dengan statistik `mean = 59.94`, `std = 8.2`, `min = 39.0`, `25% = 54.0`, `50% = 60.0`, `75% = 65.0`, `max = 88.0`.
- **Kelas `lulus = 1` (Lulus):** dari `364` baris, `0` baris terisi (`364` missing / `100.0%` missing).
- **Indikator Leakage:**
  - Jumlah siswa Lulus (`1`) yang memiliki `nilai_remedial`: `0`
  - Jumlah siswa Tidak Lulus (`0`) yang `nilai_remedial`-nya kosong: `0`

Visualisasi temuan ini dapat dilihat pada [outputs/audit_leakage_nilai_remedial.png](outputs/audit_leakage_nilai_remedial.png).

**Interpretasi:**  
Pola pada dataset menunjukkan **indikasi sangat kuat *target leakage*** karena keberadaan kolom `nilai_remedial` memisahkan kelas target secara sempurna (`100%` siswa `lulus = 0` memiliki nilai terisi dan `100%` siswa `lulus = 1` bernilai kosong). Meskipun pola ini konsisten dengan kemungkinan informasi *post-outcome*, urutan pencatatan nilai remedial perlu dikonfirmasi berdasarkan proses akademik untuk memastikan kapan nilai tersebut tersedia. Apabila fitur ini dimasukkan ke dalam pelatihan model sebelum ada kepastian bahwa nilainya tersedia pada saat prediksi dilakukan, model berisiko memperoleh performa semu yang tinggi hanya dengan mendeteksi keberadaan `nilai_remedial`, namun gagal pada skenario prediksi nyata sebelum kelulusan ditetapkan.

---

## 8. Data Quality Findings

Berdasarkan keseluruhan proses audit dan EDA pada [outputs/data_audit_report.txt](outputs/data_audit_report.txt), ditemukan beberapa isu kualitas data berikut:

1. **Indikasi Target Leakage Kritis pada `nilai_remedial`:** Keberadaan `nilai_remedial` memisahkan kelas target secara sempurna (`251` dari `251` siswa `lulus = 0` terisi, dan `364` dari `364` siswa `lulus = 1` kosong).
2. **Keberadaan Kolom Identifier (`id_siswa`):** Kolom `id_siswa` merupakan identifier siswa (`600` nilai unik dari `615` baris akibat duplikasi) yang tidak memiliki makna prediktif akademik.
3. **Duplikasi Data:** Terdapat `15` baris duplikat penuh (*exact duplicates*) dan `15` duplikat pada `id_siswa`.
4. **Inkonsistensi Format Kategorikal (`program`):** Terdapat variasi huruf kecil dan spasi di akhir (`'ipa '`, `'ips '`, `'bahasa '`) yang memecah `3` kategori jurusan menjadi `6` label mentah.
5. **Missing Values pada Fitur Prediktif:** Ditemukan nilai kosong pada `jam_belajar` (`36` baris / `5.85%`) dan `nilai_uts` (`24` baris / `3.90%`).
6. **Rentang Nilai dan Skala Fitur yang Berbeda:** Fitur prediktif memiliki skala yang berbeda-beda (contoh: `jam_belajar` berkisar `1.1`–`23.2`, sedangkan `nilai_uts` berkisar `33.0`–`100.0` dan `kehadiran` `55.0`–`100.0`), serta terdapat nilai ekstrem seperti `jam_belajar` maksimum `23.2` dan `nilai_uts` minimum `33.0`.

---

## 9. Recommendations

Untuk tahap *preprocessing* dan pemodelan (*Logistic Regression* sesuai [SPEC.md](SPEC.md)) berikutnya, direkomendasikan langkah-langkah berikut:

1. **Pengecualian Fitur `id_siswa` dan `nilai_remedial`:**
   - **Keluarkan `id_siswa`** dari daftar fitur prediksi karena hanya berfungsi sebagai identifier siswa dan berisiko menyebabkan *overfitting* apabila dikodekan.
   - **Keluarkan `nilai_remedial`** dari fitur pelatihan maupun prediksi karena menunjukkan indikasi sangat kuat *target leakage* (keberadaannya memisahkan kelas target secara sempurna) dan urutan pencatatannya dalam proses akademik perlu dikonfirmasi terlebih dahulu.
2. **Pemeriksaan Duplikasi Sebelum Penghapusan:**
   - Lakukan pemeriksaan terlebih dahulu terhadap `15` baris duplikat (beserta `id_siswa` yang berulang) untuk memastikan bahwa entri tersebut memang duplikasi pencatatan identik sebelum menghapusnya.
   - Jika terbukti duplikasi pencatatan, lakukan penghapusan baris duplikat **sebelum** melakukan *train-test split* agar baris yang sama tidak muncul sekaligus di data latih dan data uji.
3. **Standardisasi Fitur Kategorikal (`program`):**
   - Lakukan pembersihan teks pada `program` dengan `strip().upper()` sehingga seluruh label konsisten menjadi `IPA`, `IPS`, dan `BAHASA`, lalu terapkan *One-Hot Encoding* untuk mengubahnya menjadi fitur numerik bagi model.
4. **Penanganan Missing Values Tanpa Kebocoran Data:**
   - Tangani missing values pada `jam_belajar` (`36` baris) dan `nilai_uts` (`24` baris) menggunakan imputasi statistik (misalnya *median* atau *mean*). Proses `fit` imputasi wajib dilakukan hanya pada data latih (*training set*) di dalam `Pipeline`.
5. **Pemisahan Data dan *Feature Scaling* untuk Modeling:**
   - Gunakan *stratified train-test split* agar rasio target (`59.19%` lulus vs `40.81%` tidak lulus) terjaga seimbang pada data latih dan data uji.
   - Terapkan penskalaan fitur (misalnya `StandardScaler`) di dalam `Pipeline` *preprocessing* mengingat skala antarfitur numerik berbeda dan model baseline yang direncanakan adalah *Logistic Regression*.
   - Evaluasi model menggunakan `Accuracy`, `Precision`, `Recall`, `F1-Score`, dan `Confusion Matrix` sesuai spesifikasi proyek.
