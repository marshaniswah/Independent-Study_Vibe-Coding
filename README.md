# Independent Study Vibe Coding: Mini Data App Prediksi Kelulusan

Tugas **Independent Study: Vibe Coding** mata kuliah **Pemrograman untuk AI dan Data Science** (Magister AI).

## Anggota Tim
- **Marshaniswah Syamsul** (`28426001`) — Kelompok A (Pure Vibe / P1)
- **Hasna Putri Priswati** (`28426002`) — Kelompok B (Spec-First / P2)

## Ringkasan
Notebook ini menyatukan pengerjaan Kelompok A (prompt bebas P1) dan Kelompok B (spesifikasi dulu P2) untuk memprediksi kelulusan siswa (`lulus`: 0/1) dari `siswa.csv`, lengkap dengan:
- Audit data dan EDA (deteksi duplikat, nilai hilang, label tidak konsisten, dan target leakage `nilai_remedial`)
- Model baseline `LogisticRegression` dalam `Pipeline` + `ColumnTransformer` (bebas data leakage, `random_state=42`)
- Perbandingan hasil Kelompok A vs Kelompok B (`Accuracy` 0.6750, `F1` Lulus 0.7547, `Recall` Tidak Lulus 0.4200)
- Antarmuka prediksi Gradio dengan validasi input
- Audit kode contoh AI (Tahap 4), log prompt gabungan 10 iterasi, checklist, dan refleksi

## Isi Repositori
- `IndependentStudy_Vibe_Coding.ipynb` — Notebook final (Kelompok A, Kelompok B, dan perbandingannya)
- `log_prompt.csv` — Log prompt gabungan (5 iterasi Kelompok A + 5 iterasi Kelompok B)
- `siswa.csv` — Dataset sintetis dari Tahap 0
- `README.md` — Dokumentasi singkat repositori

## Cara Menjalankan
1. Buka `IndependentStudy_Vibe_Coding.ipynb` di Google Colab, Jupyter Notebook, atau VS Code.
2. Jalankan seluruh sel dari atas ke bawah (**Run All**).
