"""
Aplikasi Web Gradio: Student Graduation Prediction — Kelompok B
Pendekatan: Spec-First

Aplikasi ini memuat model Logistic Regression terlatih dari
`outputs/logistic_regression_model.joblib` (tanpa melatih ulang model)
dan menyediakan antarmuka sederhana untuk memprediksi status kelulusan siswa
berdasarkan lima fitur input akademik.

Jalankan dari root folder proyek:
    python src/app.py
"""

import math
from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import pandas as pd

# Konfigurasi path relatif terhadap root folder proyek
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "outputs" / "logistic_regression_model.joblib"

# Nama dan urutan fitur harus sama persis dengan fitur saat training
FEATURE_COLUMNS = [
    "jam_belajar",
    "kehadiran",
    "nilai_uts",
    "nilai_tugas",
    "program",
]
VALID_PROGRAMS = ("IPA", "IPS", "BAHASA")


# Cache model di memori agar dimuat secara aman saat dibutuhkan (bukan saat import modul)
_TRAINED_MODEL: Any | None = None


def load_trained_model(model_path: Path = MODEL_PATH) -> Any:
    """
    Memuat pipeline model terlatih dari file `.joblib` dengan penanganan error.
    Model tidak dilatih ulang saat aplikasi dijalankan.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan di: {model_path}. "
            "Jalankan `python src/train_model.py` terlebih dahulu untuk menghasilkan model."
        )
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Gagal memuat model dari {model_path}. Pastikan file model valid: {exc}"
        ) from exc


def get_trained_model(model_path: Path = MODEL_PATH) -> Any:
    """Mengambil instance model terlatih dari cache atau memuatnya secara aman."""
    global _TRAINED_MODEL
    if _TRAINED_MODEL is None:
        _TRAINED_MODEL = load_trained_model(model_path)
    return _TRAINED_MODEL


def _is_missing_number(value: Any) -> bool:
    """Memeriksa apakah input numerik kosong, boolean, NaN, atau tidak berhingga."""
    if value is None or isinstance(value, bool) or (isinstance(value, str) and value.strip() == ""):
        return True
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return True
    return math.isnan(numeric) or math.isinf(numeric)


def validate_inputs(
    jam_belajar: Any,
    kehadiran: Any,
    nilai_uts: Any,
    nilai_tugas: Any,
    program: Any,
) -> tuple[float, float, float, float, str]:
    """
    Memvalidasi seluruh input pengguna sebelum dikirim ke model.

    Aturan validasi:
    - Semua input wajib diisi.
    - Jam belajar tidak boleh negatif (>= 0).
    - Kehadiran, nilai UTS, dan nilai tugas harus berada pada rentang 0–100.
    - Program harus merupakan salah satu dari: IPA, IPS, BAHASA.
    """
    errors: list[str] = []

    if _is_missing_number(jam_belajar):
        errors.append("Jam Belajar wajib diisi dengan angka yang valid.")
    if _is_missing_number(kehadiran):
        errors.append("Kehadiran wajib diisi dengan angka yang valid.")
    if _is_missing_number(nilai_uts):
        errors.append("Nilai UTS wajib diisi dengan angka yang valid.")
    if _is_missing_number(nilai_tugas):
        errors.append("Nilai Tugas wajib diisi dengan angka yang valid.")
    if program is None or str(program).strip() == "":
        errors.append("Program / Jurusan wajib dipilih.")

    if errors:
        raise ValueError(" | ".join(errors))

    jam_belajar_val = float(jam_belajar)
    kehadiran_val = float(kehadiran)
    nilai_uts_val = float(nilai_uts)
    nilai_tugas_val = float(nilai_tugas)
    program_val = str(program).strip().upper()

    if jam_belajar_val < 0:
        errors.append("Jam Belajar tidak boleh bernilai negatif (minimal 0).")
    if not (0 <= kehadiran_val <= 100):
        errors.append("Kehadiran harus berada pada rentang 0 sampai 100 (%).")
    if not (0 <= nilai_uts_val <= 100):
        errors.append("Nilai UTS harus berada pada rentang 0 sampai 100.")
    if not (0 <= nilai_tugas_val <= 100):
        errors.append("Nilai Tugas harus berada pada rentang 0 sampai 100.")
    if program_val not in VALID_PROGRAMS:
        errors.append(f"Program harus dipilih dari: {', '.join(VALID_PROGRAMS)}.")

    if errors:
        raise ValueError(" | ".join(errors))

    return jam_belajar_val, kehadiran_val, nilai_uts_val, nilai_tugas_val, program_val


def build_input_dataframe(
    jam_belajar: float,
    kehadiran: float,
    nilai_uts: float,
    nilai_tugas: float,
    program: str,
) -> pd.DataFrame:
    """
    Membentuk pandas DataFrame satu baris dengan nama dan urutan kolom
    yang sama persis seperti data training.
    """
    return pd.DataFrame(
        [
            {
                "jam_belajar": jam_belajar,
                "kehadiran": kehadiran,
                "nilai_uts": nilai_uts,
                "nilai_tugas": nilai_tugas,
                "program": program,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def predict_graduation(
    jam_belajar: Any,
    kehadiran: Any,
    nilai_uts: Any,
    nilai_tugas: Any,
    program: Any,
) -> tuple[str, str, str]:
    """
    Memvalidasi input, melakukan prediksi menggunakan model Logistic Regression
    terlatih, dan mengembalikan status kelulusan, probabilitas, serta interpretasi.
    """
    try:
        jb, kh, uts, tgs, prog = validate_inputs(
            jam_belajar=jam_belajar,
            kehadiran=kehadiran,
            nilai_uts=nilai_uts,
            nilai_tugas=nilai_tugas,
            program=program,
        )
    except ValueError as exc:
        # Tampilkan pesan validasi dengan jelas apabila input belum lengkap/tidak valid
        pesan_error = f"Input tidak valid: {exc}"
        return "Input Tidak Valid", "-", pesan_error

    input_df = build_input_dataframe(jb, kh, uts, tgs, prog)

    # Pipeline model otomatis menerapkan preprocessing (imputer, scaler, encoder) yang sudah di-fit
    try:
        model = get_trained_model()
        pred_class = int(model.predict(input_df)[0])
        prob_lulus = float(model.predict_proba(input_df)[0][1]) * 100.0
    except (FileNotFoundError, RuntimeError, IndexError, ValueError, TypeError) as exc:
        return (
            "Gagal Memprediksi",
            "-",
            f"Terjadi kesalahan saat memuat model atau memproses prediksi: {exc}",
        )
    except Exception as exc:
        return (
            "Gagal Memprediksi",
            "-",
            f"Terjadi kesalahan tak terduga saat prediksi: {exc}",
        )

    status_label = "Lulus" if pred_class == 1 else "Tidak Lulus"
    probabilitas_text = f"{prob_lulus:.2f}%"
    interpretasi_text = (
        f"Model memprediksi siswa berstatus **{status_label}** dengan estimasi "
        f"probabilitas lulus sebesar **{probabilitas_text}**. "
        "Catatan: Hasil ini merupakan prediksi statistik dari model Machine Learning "
        "(Logistic Regression) berdasarkan data historis, **bukan keputusan kelulusan resmi** "
        "dari pihak sekolah."
    )

    return status_label, probabilitas_text, interpretasi_text


def create_app() -> gr.Blocks:
    """Membangun antarmuka Gradio yang sederhana, rapi, dan mudah digunakan."""
    with gr.Blocks(title="Student Graduation Prediction — Kelompok B") as app:
        gr.Markdown(
            """
            # Student Graduation Prediction — Kelompok B
            Aplikasi prediksi status kelulusan siswa berbasis **Machine Learning (Logistic Regression)**
            yang dikembangkan dengan pendekatan *Spec-First*.

            Masukkan lima data akademik siswa di bawah ini, lalu tekan tombol **Prediksi Kelulusan**.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Data Akademik Siswa")
                jam_belajar_input = gr.Number(
                    label="Jam Belajar",
                    value=None,
                    placeholder="Contoh: 7.5 (minimal 0)",
                )
                kehadiran_input = gr.Number(
                    label="Kehadiran (%)",
                    value=None,
                    placeholder="Rentang 0 - 100",
                )
                nilai_uts_input = gr.Number(
                    label="Nilai UTS",
                    value=None,
                    placeholder="Rentang 0 - 100",
                )
                nilai_tugas_input = gr.Number(
                    label="Nilai Tugas",
                    value=None,
                    placeholder="Rentang 0 - 100",
                )
                program_input = gr.Dropdown(
                    choices=list(VALID_PROGRAMS),
                    label="Program / Jurusan",
                    value=None,
                )

                predict_button = gr.Button("Prediksi Kelulusan", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Hasil Prediksi")
                status_output = gr.Textbox(
                    label="Status Prediksi Kelulusan",
                    interactive=False,
                )
                probabilitas_output = gr.Textbox(
                    label="Probabilitas Lulus",
                    interactive=False,
                )
                interpretasi_output = gr.Markdown(
                    label="Interpretasi",
                    value="Silakan isi seluruh data input lalu klik **Prediksi Kelulusan**.",
                )

        # Contoh input cepat agar mahasiswa mudah menguji aplikasi
        gr.Examples(
            examples=[
                [8.5, 90.0, 82.0, 85.0, "IPA"],
                [3.0, 65.0, 50.0, 58.0, "IPS"],
                [6.5, 80.0, 72.0, 75.0, "BAHASA"],
            ],
            inputs=[
                jam_belajar_input,
                kehadiran_input,
                nilai_uts_input,
                nilai_tugas_input,
                program_input,
            ],
            label="Contoh Data Siswa",
        )

        predict_button.click(
            fn=predict_graduation,
            inputs=[
                jam_belajar_input,
                kehadiran_input,
                nilai_uts_input,
                nilai_tugas_input,
                program_input,
            ],
            outputs=[
                status_output,
                probabilitas_output,
                interpretasi_output,
            ],
        )

    return app


if __name__ == "__main__":
    demo = create_app()
    demo.launch()
