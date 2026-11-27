"""
Modul Data Preprocessing
Proyek: Student Graduation Prediction — Kelompok B (Pendekatan Spec-First)

Skrip ini melakukan pemuatan data, validasi kolom dan target, penghapusan
baris duplikat identik, normalisasi kolom kategorikal, pemisahan fitur/target
tanpa data leakage, pembagian train-test secara stratified, serta penyusunan
preprocessing pipeline (belum di-fit) untuk tahap pemodelan berikutnya.

Catatan:
- Dataset asli `data/siswa.csv` tidak diubah.
- Tidak melakukan pelatihan model ataupun membangun antarmuka Gradio.
- Imputasi, scaling, dan encoding hanya didefinisikan dalam pipeline agar
  di-fit khusus pada data training saat tahap modeling.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Konstanta kolom sesuai SPEC.md dan temuan EDA_REPORT.md
TARGET_COL = "lulus"
ID_COL = "id_siswa"
LEAKAGE_COL = "nilai_remedial"

NUMERIC_FEATURES = ["jam_belajar", "kehadiran", "nilai_uts", "nilai_tugas"]
CATEGORICAL_FEATURES = ["program"]
SELECTED_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXCLUDED_COLUMNS = [ID_COL, LEAKAGE_COL]

VALID_TARGET_VALUES = {0, 1}
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def get_project_paths() -> tuple[Path, Path, Path]:
    """Menentukan path root proyek, dataset input, dan direktori output secara aman."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "siswa.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, data_path, output_dir


def validate_raw_dataset(df: pd.DataFrame) -> None:
    """
    Memvalidasi dataset tidak kosong, memiliki seluruh kolom yang dibutuhkan,
    serta kolom target `lulus` valid (tidak kosong dan hanya berisi 0 atau 1).
    """
    if df.empty:
        raise ValueError("Dataset kosong dan tidak dapat diproses.")

    required_cols = set(SELECTED_FEATURES + [TARGET_COL, ID_COL])
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(
            f"Kolom wajib berikut tidak ditemukan pada dataset: {missing_cols}"
        )

    # Validasi nilai kolom target `lulus`
    if df[TARGET_COL].isna().any():
        missing_target_count = int(df[TARGET_COL].isna().sum())
        raise ValueError(
            f"Kolom target '{TARGET_COL}' memiliki {missing_target_count} nilai kosong (NaN)."
        )

    numeric_target = pd.to_numeric(df[TARGET_COL], errors="coerce")
    if numeric_target.isna().any():
        raise ValueError(
            f"Kolom target '{TARGET_COL}' mengandung nilai non-numerik yang tidak valid."
        )

    # Pastikan tidak ada nilai pecahan atau di luar himpunan {0, 1}
    if not (numeric_target % 1 == 0).all():
        raise ValueError(
            f"Kolom target '{TARGET_COL}' harus berupa bilangan bulat biner (0 atau 1)."
        )

    unique_targets = set(numeric_target.astype(int).unique())
    invalid_targets = unique_targets - VALID_TARGET_VALUES
    if invalid_targets:
        raise ValueError(
            f"Kolom target '{TARGET_COL}' memiliki nilai tidak valid: {sorted(invalid_targets)}. "
            f"Nilai yang diizinkan hanya {sorted(VALID_TARGET_VALUES)}."
        )

    # Validasi tipe data fitur numerik (nilai non-null harus numerik dan bukan boolean)
    for col in NUMERIC_FEATURES:
        non_null_values = df[col].dropna()
        if non_null_values.empty:
            continue
        if non_null_values.map(lambda val: isinstance(val, bool)).any():
            raise ValueError(
                f"Kolom numerik '{col}' tidak boleh berisi nilai boolean."
            )
        parsed_numeric = pd.to_numeric(non_null_values, errors="coerce")
        if parsed_numeric.isna().any():
            raise ValueError(
                f"Kolom numerik '{col}' mengandung nilai non-numerik yang tidak valid."
            )

    # Validasi tipe data fitur kategorikal (nilai non-null harus berupa string)
    for col in CATEGORICAL_FEATURES:
        non_null_cat = df[col].dropna()
        if not non_null_cat.empty and not non_null_cat.map(lambda val: isinstance(val, str)).all():
            raise ValueError(
                f"Kolom kategorikal '{col}' harus bertipe string pada seluruh nilai non-kosong."
            )


def load_and_clean_data(
    data_path: Path | None = None,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Membaca dataset `data/siswa.csv` (read-only), memvalidasi kolom & target,
    menghapus hanya baris duplikat identik (`drop_duplicates()`), serta
    menormalisasi kolom `program` (strip + uppercase).

    Parameters
    ----------
    data_path : Path | None, optional
        Path ke berkas CSV. Jika None, otomatis menggunakan `data/siswa.csv`.
    return_metadata : bool, default=False
        Jika True, mengembalikan `(df_clean, metadata)` untuk keperluan laporan.
    """
    if data_path is None:
        _, data_path, _ = get_project_paths()

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan pada path: {data_path}")

    try:
        df_raw = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Berkas dataset kosong: {data_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Gagal membaca dataset dari {data_path}: {exc}") from exc

    # Validasi struktur kolom dan target sebelum pembersihan
    validate_raw_dataset(df_raw)

    rows_before = int(len(df_raw))
    exact_duplicates = int(df_raw.duplicated().sum())
    id_duplicates_before = int(df_raw[ID_COL].duplicated().sum())
    raw_program_values = df_raw["program"].value_counts(dropna=False).to_dict()

    # Hapus HANYA baris yang duplikat identik di seluruh kolom.
    # Baris dengan id_siswa sama tetapi isi berbeda (jika ada) tidak dihapus otomatis.
    df_clean = df_raw.drop_duplicates().reset_index(drop=True)
    rows_after = int(len(df_clean))
    id_duplicates_after = int(df_clean[ID_COL].duplicated().sum())

    # Normalisasi kolom kategorikal `program`: hapus spasi awal/akhir dan ubah ke huruf kapital
    df_clean["program"] = df_clean["program"].apply(
        lambda val: val.strip().upper() if isinstance(val, str) else val
    )
    clean_program_values = df_clean["program"].value_counts(dropna=False).to_dict()

    # Pastikan tipe target berupa integer biner (0 / 1)
    df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL]).astype(int)

    if not return_metadata:
        return df_clean

    metadata: dict[str, Any] = {
        "data_path": str(data_path),
        "rows_before": rows_before,
        "rows_after": rows_after,
        "exact_duplicates_removed": exact_duplicates,
        "id_duplicates_before": id_duplicates_before,
        "id_duplicates_after": id_duplicates_after,
        "raw_program_counts": raw_program_values,
        "clean_program_counts": clean_program_values,
    }
    return df_clean, metadata


def prepare_train_test_data(
    df_clean: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Memisahkan fitur (X) dan target (y), mengecualikan `id_siswa` dan `nilai_remedial`,
    lalu membagi data menjadi train dan test menggunakan stratified `train_test_split`.
    """
    validate_raw_dataset(df_clean)

    # Pilih hanya fitur yang aman dan valid; kecualikan identifier & fitur target leakage
    X = df_clean[SELECTED_FEATURES].copy()
    y = df_clean[TARGET_COL].astype(int).copy()

    # Guard eksplisit agar identifier dan kandidat target leakage tidak pernah masuk fitur
    for excluded_col in EXCLUDED_COLUMNS:
        if excluded_col in X.columns:
            raise AssertionError(
                f"Kolom terlarang '{excluded_col}' tidak boleh masuk ke matriks fitur."
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Membangun preprocessing pipeline scikit-learn (UNFITTED / belum di-fit).

    Pipeline ini dirancang untuk digabungkan ke pipeline modeling dan HANYA
    di-fit pada `X_train` saat tahap pelatihan model guna mencegah data leakage:
    - Fitur numerik: SimpleImputer(strategy='median') -> StandardScaler()
    - Fitur kategorikal: SimpleImputer(strategy='most_frequent') -> OneHotEncoder(handle_unknown='ignore')
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def _format_target_distribution(y: pd.Series) -> str:
    """Memformat distribusi jumlah dan persentase kelas target `lulus`."""
    counts = y.value_counts().sort_index()
    pcts = (y.value_counts(normalize=True).sort_index() * 100).round(2)
    summary = pd.DataFrame(
        {
            "label": ["Tidak Lulus (0)" if idx == 0 else "Lulus (1)" for idx in counts.index],
            "jumlah": counts.values,
            "persentase (%)": pcts.values,
        },
        index=counts.index,
    )
    return summary.to_string()


def build_preprocessing_report(
    metadata: dict[str, Any],
    df_clean: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> str:
    """Menyusun teks laporan ringkasan hasil preprocessing."""
    separator = "=" * 70
    missing_train = X_train.isna().sum().to_string()
    missing_test = X_test.isna().sum().to_string()

    lines = [
        "LAPORAN PREPROCESSING DATA — STUDENT GRADUATION PREDICTION (KELOMPOK B)",
        f"Sumber Dataset : {metadata['data_path']}",
        "",
        separator,
        "1. RINGKASAN DUPLIKASI & PEMBERSIHAN DATA",
        separator,
        f"Jumlah Baris Sebelum Penghapusan Duplikat : {metadata['rows_before']}",
        f"Jumlah Baris Duplikat Identik Ditemukan   : {metadata['exact_duplicates_removed']}",
        f"Jumlah Baris Setelah drop_duplicates()    : {metadata['rows_after']}",
        f"Duplikat id_siswa Sebelum Cleaning        : {metadata['id_duplicates_before']}",
        f"Duplikat id_siswa Setelah Cleaning        : {metadata['id_duplicates_after']}",
        "",
        "Normalisasi Kolom 'program' (strip + upper):",
        f"- Distribusi Mentah : {metadata['raw_program_counts']}",
        f"- Distribusi Bersih : {metadata['clean_program_counts']}",
        "",
        separator,
        "2. FITUR YANG DIGUNAKAN & KOLOM YANG DIKECUALIKAN",
        separator,
        f"Target (y)          : {TARGET_COL}",
        f"Fitur Numerik       : {NUMERIC_FEATURES}",
        f"Fitur Kategorikal   : {CATEGORICAL_FEATURES}",
        f"Total Fitur Input   : {len(SELECTED_FEATURES)} -> {SELECTED_FEATURES}",
        f"Kolom Dikecualikan  : {EXCLUDED_COLUMNS}",
        f"  - '{ID_COL}'       : identifier siswa (tidak bersifat prediktif).",
        f"  - '{LEAKAGE_COL}' : indikasi sangat kuat target leakage (tidak digunakan).",
        "",
        separator,
        "3. DISTRIBUSI TARGET (SETELAH PEMBERSIHAN DUPLIKAT)",
        separator,
        "Keseluruhan Data Bersih:",
        _format_target_distribution(df_clean[TARGET_COL]),
        "",
        separator,
        "4. HASIL PEMBAGIAN TRAIN-TEST SPLIT (STRATIFIED)",
        separator,
        f"Parameter Split : test_size={DEFAULT_TEST_SIZE}, random_state={DEFAULT_RANDOM_STATE}, stratify={TARGET_COL}",
        f"Ukuran X_train  : {X_train.shape} | Ukuran y_train : {y_train.shape}",
        f"Ukuran X_test   : {X_test.shape}  | Ukuran y_test  : {y_test.shape}",
        "",
        "Distribusi Target pada Data Training (y_train):",
        _format_target_distribution(y_train),
        "",
        "Distribusi Target pada Data Testing (y_test):",
        _format_target_distribution(y_test),
        "",
        separator,
        "5. STATUS MISSING VALUES & KEBIJAKAN ANTI DATA LEAKAGE",
        separator,
        "Missing Values pada X_train (belum diimputasi):",
        missing_train,
        "",
        "Missing Values pada X_test (belum diimputasi):",
        missing_test,
        "",
        "Rancangan Preprocessing Pipeline (BELUM DI-FIT):",
        str(preprocessor),
        "",
        "Catatan Penting:",
        "- Imputasi, scaling, dan encoding TIDAK di-fit pada seluruh dataset maupun di tahap ini.",
        "- Pipeline hanya akan di-fit pada data training (X_train) saat tahap pelatihan model.",
        "- Dataset asli data/siswa.csv tidak mengalami perubahan.",
    ]
    return "\n".join(lines) + "\n"


def save_preprocessing_report(report_text: str, output_dir: Path | None = None) -> Path:
    """Menyimpan laporan hasil preprocessing ke `outputs/preprocessing_report.txt`."""
    if output_dir is None:
        _, _, output_dir = get_project_paths()
    report_path = output_dir / "preprocessing_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def main() -> None:
    """Menjalankan alur preprocessing end-to-end (tanpa fitting transformator atau training model)."""
    _, data_path, output_dir = get_project_paths()

    # 1. Baca, validasi, bersihkan duplikat identik, dan normalisasi program
    df_clean, metadata = load_and_clean_data(data_path=data_path, return_metadata=True)

    # 2. Pisahkan fitur & target lalu lakukan stratified train-test split
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        df_clean,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # 3. Bangun preprocessing pipeline (unfitted) untuk tahap modeling berikutnya
    preprocessor = build_preprocessing_pipeline()

    # 4. Susun, tampilkan, dan simpan laporan ringkasan preprocessing
    report_text = build_preprocessing_report(
        metadata=metadata,
        df_clean=df_clean,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
    )
    report_path = save_preprocessing_report(report_text, output_dir=output_dir)

    print(report_text)
    print(f"Laporan preprocessing berhasil disimpan ke: {report_path}")


if __name__ == "__main__":
    main()
