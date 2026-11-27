"""
Modul Pelatihan & Evaluasi Baseline Model (Logistic Regression)
Proyek: Student Graduation Prediction — Kelompok B (Pendekatan Spec-First)

Skrip ini melatih model klasifikasi biner Logistic Regression menggunakan
scikit-learn Pipeline agar seluruh transformasi preprocessing (imputasi,
scaling, dan one-hot encoding) hanya di-fit pada data training (`X_train`),
sehingga tidak terjadi data leakage ke data testing (`X_test`).

Output yang dihasilkan:
- outputs/model_evaluation_report.txt
- outputs/logistic_regression_model.joblib
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

# Mendukung eksekusi langsung (`python src/train_model.py`) maupun impor sebagai modul (`src.train_model`)
try:
    from src.preprocessing import (
        DEFAULT_RANDOM_STATE,
        DEFAULT_TEST_SIZE,
        SELECTED_FEATURES,
        TARGET_COL,
        build_preprocessing_pipeline,
        get_project_paths,
        load_and_clean_data,
        prepare_train_test_data,
    )
except ImportError:
    from preprocessing import (
        DEFAULT_RANDOM_STATE,
        DEFAULT_TEST_SIZE,
        SELECTED_FEATURES,
        TARGET_COL,
        build_preprocessing_pipeline,
        get_project_paths,
        load_and_clean_data,
        prepare_train_test_data,
    )

# Parameter baseline Logistic Regression sesuai ketentuan proyek
MODEL_RANDOM_STATE = 42
MODEL_MAX_ITER = 1000


def build_model_pipeline() -> Pipeline:
    """
    Menggabungkan preprocessing pipeline dengan model Logistic Regression
    di dalam satu `sklearn.pipeline.Pipeline`.

    Dengan menggunakan Pipeline:
    - Saat `.fit(X_train, y_train)` dipanggil, preprocessor menghitung parameter
      imputasi, scaling, dan encoding HANYA dari data training.
    - Saat `.predict(X_test)` dipanggil, parameter dari data training tersebut
      otomatis diterapkan ke data testing tanpa melakukan fit ulang.
    """
    preprocessor = build_preprocessing_pipeline()
    classifier = LogisticRegression(
        random_state=MODEL_RANDOM_STATE,
        max_iter=MODEL_MAX_ITER,
    )

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return model_pipeline


def evaluate_model(
    model_pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Mengevaluasi performa model terlatih pada data testing (`X_test`, `y_test`)
    serta menyusun ringkasan distribusi prediksi dan perbandingan aktual vs prediksi.
    """
    # Prediksi kelas (0 = Tidak Lulus, 1 = Lulus) dan probabilitas kelas positif (1)
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    # Hitung metrik evaluasi klasifikasi biner untuk kelas 1 (Lulus) dan kelas 0 (Tidak Lulus)
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    precision_class_0 = float(precision_score(y_test, y_pred, pos_label=0, zero_division=0))
    recall_class_0 = float(recall_score(y_test, y_pred, pos_label=0, zero_division=0))
    f1_class_0 = float(f1_score(y_test, y_pred, pos_label=0, zero_division=0))

    # Confusion matrix dengan urutan kelas eksplisit [0, 1]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cls_report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Tidak Lulus (0)", "Lulus (1)"],
        digits=4,
        zero_division=0,
    )

    # Distribusi nilai aktual vs distribusi nilai prediksi pada data testing
    actual_counts = pd.Series(y_test.values).value_counts().reindex([0, 1], fill_value=0)
    pred_counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)

    distribution_df = pd.DataFrame(
        {
            "label": ["Tidak Lulus (0)", "Lulus (1)"],
            "aktual_jumlah": actual_counts.values,
            "aktual_pct (%)": (actual_counts.values / len(y_test) * 100).round(2),
            "prediksi_jumlah": pred_counts.values,
            "prediksi_pct (%)": (pred_counts.values / len(y_pred) * 100).round(2),
        },
        index=[0, 1],
    )

    # Tabel perbandingan per sampel antara nilai aktual dan hasil prediksi model
    comparison_df = pd.DataFrame(
        {
            "aktual": y_test.values,
            "prediksi": y_pred,
            "prob_lulus": y_prob.round(4),
            "status": [
                "Benar" if a == p else "Salah" for a, p in zip(y_test.values, y_pred)
            ],
        },
        index=X_test.index,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "precision_class_0": precision_class_0,
        "recall_class_0": recall_class_0,
        "f1_class_0": f1_class_0,
        "confusion_matrix": cm,
        "classification_report": cls_report,
        "distribution_df": distribution_df,
        "comparison_df": comparison_df,
    }


def format_evaluation_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    eval_results: dict[str, Any],
) -> str:
    """Memformat seluruh hasil pelatihan dan evaluasi model ke teks yang rapi."""
    separator = "=" * 70
    cm = eval_results["confusion_matrix"]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    cm_df = pd.DataFrame(
        cm,
        index=["Aktual: Tidak Lulus (0)", "Aktual: Lulus (1)"],
        columns=["Prediksi: Tidak Lulus (0)", "Prediksi: Lulus (1)"],
    )

    comparison_df: pd.DataFrame = eval_results["comparison_df"]
    correct_count = int((comparison_df["status"] == "Benar").sum())
    wrong_count = int((comparison_df["status"] == "Salah").sum())

    # Tampilkan 15 baris pertama perbandingan aktual vs prediksi agar ringkas & informatif
    sample_comparison = comparison_df.head(15).to_string()

    lines = [
        "LAPORAN EVALUASI BASELINE MODEL — LOGISTIC REGRESSION (KELOMPOK B)",
        "",
        separator,
        "1. RINGKASAN DATA & KONFIGURASI MODEL",
        separator,
        f"Fitur Digunakan     : {SELECTED_FEATURES}",
        f"Target              : {TARGET_COL} (0 = Tidak Lulus, 1 = Lulus)",
        f"Parameter Split     : test_size={DEFAULT_TEST_SIZE}, random_state={DEFAULT_RANDOM_STATE}, stratify={TARGET_COL}",
        f"Ukuran Data Train   : X_train={X_train.shape}, y_train={y_train.shape}",
        f"Ukuran Data Test    : X_test={X_test.shape}, y_test={y_test.shape}",
        f"Model Baseline      : LogisticRegression(random_state={MODEL_RANDOM_STATE}, max_iter={MODEL_MAX_ITER})",
        "Pencegahan Leakage  : Preprocessing (imputasi, scaling, encoding) di-fit HANYA pada X_train melalui Pipeline.",
        "",
        separator,
        "2. METRIK EVALUASI PADA DATA TESTING",
        separator,
        f"Accuracy               : {eval_results['accuracy']:.4f} ({eval_results['accuracy'] * 100:.2f}%)",
        "",
        "Metrik Kelas 1 (Lulus):",
        f"  Precision (Kelas 1)  : {eval_results['precision']:.4f}",
        f"  Recall    (Kelas 1)  : {eval_results['recall']:.4f}",
        f"  F1-Score  (Kelas 1)  : {eval_results['f1_score']:.4f}",
        "",
        "Metrik Kelas 0 (Tidak Lulus):",
        f"  Precision (Kelas 0)  : {eval_results['precision_class_0']:.4f}",
        f"  Recall    (Kelas 0)  : {eval_results['recall_class_0']:.4f}",
        f"  F1-Score  (Kelas 0)  : {eval_results['f1_class_0']:.4f}",
        "",
        separator,
        "3. CONFUSION MATRIX",
        separator,
        cm_df.to_string(),
        "",
        "Rincian Sel Confusion Matrix:",
        f"- True Negative  (TN : Aktual 0, Prediksi 0) = {tn}",
        f"- False Positive (FP : Aktual 0, Prediksi 1) = {fp}",
        f"- False Negative (FN : Aktual 1, Prediksi 0) = {fn}",
        f"- True Positive  (TP : Aktual 1, Prediksi 1) = {tp}",
        "",
        separator,
        "4. CLASSIFICATION REPORT",
        separator,
        eval_results["classification_report"].rstrip(),
        "",
        separator,
        "5. DISTRIBUSI PREDIKSI VS NILAI AKTUAL",
        separator,
        eval_results["distribution_df"].to_string(),
        "",
        f"Total Prediksi Tepat : {correct_count} dari {len(comparison_df)} data uji",
        f"Total Prediksi Salah : {wrong_count} dari {len(comparison_df)} data uji",
        "",
        separator,
        "6. CONTOH PERBANDINGAN AKTUAL VS PREDIKSI (15 DATA UJI PERTAMA)",
        separator,
        sample_comparison,
    ]
    return "\n".join(lines) + "\n"


def save_training_artifacts(
    model_pipeline: Pipeline,
    report_text: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Menyimpan laporan evaluasi (.txt) dan model terlatih (.joblib) ke folder outputs/."""
    report_path = output_dir / "model_evaluation_report.txt"
    model_path = output_dir / "logistic_regression_model.joblib"

    report_path.write_text(report_text, encoding="utf-8")
    joblib.dump(model_pipeline, model_path)

    return report_path, model_path


def main() -> None:
    """Menjalankan alur lengkap training baseline Logistic Regression dan evaluasinya."""
    _, data_path, output_dir = get_project_paths()

    # 1. Muat dan bersihkan data menggunakan fungsi dari src/preprocessing.py
    df_clean = load_and_clean_data(data_path=data_path)

    # 2. Bagi data menjadi training dan testing dengan parameter yang konsisten
    X_train, X_test, y_train, y_test = prepare_train_test_data(
        df_clean,
        test_size=DEFAULT_TEST_SIZE,
        random_state=DEFAULT_RANDOM_STATE,
    )

    # 3. Bangun sklearn Pipeline (preprocessor + LogisticRegression)
    model_pipeline = build_model_pipeline()

    # 4. Latih model (beserta preprocessor di dalamnya) HANYA menggunakan data training
    model_pipeline.fit(X_train, y_train)

    # 5. Evaluasi model pada data testing yang belum pernah disentuh saat fitting
    eval_results = evaluate_model(model_pipeline, X_test, y_test)

    # 6. Format laporan, tampilkan ke terminal, dan simpan laporan + model joblib
    report_text = format_evaluation_report(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        eval_results=eval_results,
    )
    report_path, model_path = save_training_artifacts(
        model_pipeline=model_pipeline,
        report_text=report_text,
        output_dir=output_dir,
    )

    print(report_text)
    print(f"Laporan evaluasi berhasil disimpan ke : {report_path}")
    print(f"Model terlatih berhasil disimpan ke   : {model_path}")


if __name__ == "__main__":
    main()
