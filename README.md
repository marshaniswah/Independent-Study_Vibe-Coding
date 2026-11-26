# Independent Study Vibe Coding: Mini Data App Prediksi Kelulusan

Repositori ini berisi pengerjaan tugas **Independent Study: Vibe Coding** pada mata kuliah **Pemrograman untuk AI dan Data Science** (Program Magister AI).

## Anggota Kelompok (Kelompok A)
- **Marshaniswah Syamsul** (`28426001`)
- **Hasna Putri Priswati** (`28426002`)

## Ringkasan Tugas
Membangun mini aplikasi data science untuk memprediksi kelulusan siswa (`lulus`: `0`/`1`) dari data sintetis `siswa.csv` dengan bantuan asisten AI, meliputi:
1. **Audit Data & EDA:** Mendeteksi baris duplikat, nilai hilang, ketidakkonsistenan label kategori, dan indikasi *target leakage* (`nilai_remedial`).
2. **Model Baseline:** Membangun pipeline klasifikasi (`ColumnTransformer` + `LogisticRegression`) dengan pemisahan data latih/uji 80/20 terstratifikasi (`random_state=42`) sebelum imputasi dan *scaling*, serta evaluasi metrik lengkap (`Accuracy`, `Precision`, `Recall`, `F1-Score`, dan `Confusion Matrix`).
3. **Antarmuka Prediksi (Gradio):** Menyediakan antarmuka web interaktif untuk memprediksi probabilitas lulus dan tidak lulus menggunakan objek `model` yang sudah dilatih.
4. **Audit Kode Contoh AI:** Mengidentifikasi kesalahan umum pada kode hasil AI (*target leakage*, *data leakage*, efisiensi fitur ID, reproduktibilitas, dan metrik evaluasi).
5. **Log Prompt & Verifikasi:** Mencatat iterasi prompt ke `log_prompt.csv`, mengisi checklist verifikasi, serta refleksi pembelajaran.

## Struktur Berkas
- `IndependentStudy_Vibe_Coding.ipynb` — Notebook utama berisi seluruh tahapan pengerjaan (Tahap 0–5).
- `siswa.csv` — Dataset sintetis yang dibangkitkan pada Tahap 0.
- `log_prompt.csv` — Catatan riwayat iterasi prompt pada Tahap 5.

## Cara Menjalankan
1. Pastikan Python beserta pustaka `numpy`, `pandas`, `scikit-learn`, `matplotlib`, dan `gradio` sudah terpasang (atau jalankan sel instalasi di Tahap 0).
2. Buka `IndependentStudy_Vibe_Coding.ipynb` melalui Jupyter Notebook, VS Code, atau Google Colab.
3. Jalankan seluruh sel secara berurutan dari atas ke bawah (**Run All**).
