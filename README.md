# Independent Study Vibe Coding: Mini Data App Prediksi Kelulusan

Repositori ini berisi pengerjaan tugas **Independent Study: Vibe Coding** pada mata kuliah **Pemrograman untuk AI dan Data Science** (Program Magister AI). Proyek ini membandingkan dua pendekatan *prompt engineering* dalam membangun aplikasi prediksi kelulusan siswa (`lulus`: `0`/`1`) dari data sintetis `siswa.csv`: pendekatan **Pure Vibe (P1)** pada **Kelompok A** dan pendekatan **Spec-First (P2)** pada **Kelompok B**.

## Anggota & Pembagian Bagian
| Bagian | Pendekatan Prompt | Anggota | NIM | Folder |
|---|---|---|---|---|
| **Kelompok A** | Pure Vibe (**P1**) | Marshaniswah Syamsul | `28426001` | [`Kelompok-A/`](Kelompok-A/) |
| **Kelompok B** | Spec-First (**P2**) | Hasna Putri Priswati | `28426002` | [`Kelompok-B/`](Kelompok-B/) |

## Ringkasan Tugas
Kedua bagian mengerjakan seluruh alur praktikum (Tahap 0–5):
1. **Setup & Audit Data:** Membangkitkan dataset sintetis `siswa.csv` dan mendeteksi 4 masalah kualitas data (15 baris duplikat, nilai hilang pada `jam_belajar` dan `nilai_uts`, label `program` tidak konsisten, serta indikasi *target leakage* pada `nilai_remedial`).
2. **EDA & Model Baseline:** Membersihkan data tanpa *leakage*, membuang `id_siswa` dan `nilai_remedial`, melakukan *train-test split* 80/20 terstratifikasi (`random_state=42`) sebelum imputasi dan *scaling*, lalu melatih `Pipeline` (`ColumnTransformer` + `LogisticRegression`).
3. **Antarmuka Prediksi (Gradio):** Menyediakan antarmuka web interaktif dengan input `program`, `jam_belajar`, `kehadiran`, `nilai_uts`, dan `nilai_tugas` untuk menampilkan probabilitas lulus dan tidak lulus.
4. **Audit Kode Contoh AI:** Mengidentifikasi kesalahan kritis pada kode hasil AI (*target leakage*, *data leakage* saat preprocessing, encoding `id_siswa`, data kotor, ketiadaan seed/stratifikasi, dan metrik evaluasi yang hanya memakai akurasi).
5. **Log Prompt, Verifikasi & Refleksi:** Mencatat iterasi prompt ke `log_prompt.csv`, memverifikasi kompleksitas waktu Big-O, melengkapi checklist, dan menulis refleksi.

## Perbandingan Singkat & Hasil Baseline
Setelah pembersihan dan pencegahan *data leakage* diverifikasi pada kedua pendekatan, keduanya menghasilkan performa model baseline yang konsisten pada data uji (120 baris):

| Metrik (Kelas Lulus = 1) | Kelompok A (P1 — Pure Vibe) | Kelompok B (P2 — Spec-First) |
|---|---|---|
| **Accuracy** | `67,50%` (`0.6750`) | `67,50%` (`0.6750`) |
| **Precision** | `0.6742` | `0.6742` |
| **Recall** | `0.8571` | `0.8571` |
| **F1-Score** | `0.7547` | `0.7547` |
| **Struktur Pengerjaan** | Notebook terintegrasi (`IndependentStudy_Vibe_Coding.ipynb`) + `log_prompt.csv` | Notebook + proyek modular (`src/`, `outputs/`, `SPEC.md`, `EDA_REPORT.md`, `TESTING_REPORT.md`, `CODE_REVIEW.md`, `PROMPT_LOG.md`) |

## Struktur Repositori
```text
.
├── IndependentStudy_Vibe_Coding.ipynb   # Notebook gabungan (Kelompok A + B + Perbandingan) — siap kumpul Edunex
├── log_prompt.csv                       # Log prompt gabungan (10 iterasi: 5 Kelompok A + 5 Kelompok B)
├── siswa.csv                            # Dataset sintetis hasil Tahap 0
├── README.md
├── Kelompok-A/                          # Arsip pengerjaan individu Kelompok A
│   ├── IndependentStudy_Vibe_Coding.ipynb
│   ├── log_prompt.csv
│   └── siswa.csv
└── Kelompok-B/                          # Arsip pengerjaan modular Kelompok B
    ├── Praktikum_Prompt_Engineering_Kel_B.ipynb
    ├── log_prompt.csv
    ├── README.md
    ├── SPEC.md
    ├── EDA_REPORT.md
    ├── CODE_REVIEW.md
    ├── PROMPT_LOG.md
    ├── TESTING_REPORT.md
    ├── requirements.txt
    ├── data/
    │   └── siswa.csv
    ├── src/
    │   ├── data_audit.py
    │   ├── preprocessing.py
    │   ├── train_model.py
    │   └── app.py
    └── outputs/
        ├── *.txt (laporan audit, preprocessing, evaluasi)
        ├── *.png (visualisasi EDA & audit leakage)
        └── *.joblib (model terlatih)
```

## Cara Menjalankan

### Notebook Gabungan (Siap Kumpul Edunex)
1. Buka `IndependentStudy_Vibe_Coding.ipynb` di root repositori melalui Google Colab, Jupyter Notebook, atau VS Code.
2. Jalankan seluruh sel secara berurutan dari atas ke bawah (**Run All**) — seluruh bagian Kelompok A, Kelompok B, perbandingan hasil, antarmuka Gradio, audit kode Tahap 4, dan pembentukan `log_prompt.csv` berjalan mandiri tanpa unggah file eksternal.

### Proyek Modular Kelompok B (Opsional)
1. Buka `Kelompok-B/Praktikum_Prompt_Engineering_Kel_B.ipynb`, atau jalankan versi modularnya dari dalam folder `Kelompok-B/`:
   ```bash
   cd Kelompok-B
   pip install -r requirements.txt
   python src/train_model.py
   python src/app.py
   ```
2. Detail lengkap tersedia di [`Kelompok-B/README.md`](Kelompok-B/README.md).
