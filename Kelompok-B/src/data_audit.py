"""
Modul Data Audit dan Exploratory Data Analysis (EDA)
Proyek: Student Graduation Prediction — Kelompok B (Pendekatan Spec-First)

Skrip ini melakukan audit kualitas data dan EDA terhadap dataset `data/siswa.csv`
sesuai spesifikasi pada `SPEC.md`, tanpa mengubah dataset asli dan tanpa
melakukan pelatihan model maupun membangun antarmuka Gradio.
"""

from pathlib import Path

import matplotlib

# Gunakan backend non-interaktif agar grafik dapat disimpan langsung ke file
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Konfigurasi gaya visualisasi agar konsisten dan mudah dibaca
sns.set_theme(style="whitegrid", palette="Set2")

# Konstanta nama kolom sesuai SPEC.md
TARGET_COL = "lulus"
ID_COL = "id_siswa"
LEAKAGE_COL = "nilai_remedial"
CATEGORICAL_COLS = ["program"]
PREDICTIVE_NUMERIC_COLS = ["jam_belajar", "kehadiran", "nilai_uts", "nilai_tugas"]


def get_project_paths() -> tuple[Path, Path, Path]:
    """Menentukan path root proyek, dataset input, dan direktori output."""
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "siswa.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, data_path, output_dir


def load_dataset(data_path: Path) -> pd.DataFrame:
    """1. Membaca dataset siswa.csv menggunakan pandas tanpa mengubah file asli."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan pada path: {data_path}")
    return pd.read_csv(data_path)


def format_section(title: str, body: str) -> str:
    """Memformat bagian laporan audit dengan pemisah yang rapi."""
    separator = "=" * 70
    return f"\n{separator}\n{title}\n{separator}\n{body}\n"


def audit_basic_structure(df: pd.DataFrame) -> str:
    """2, 3, 4. Memeriksa ukuran dataset, 5 baris pertama, dan tipe data kolom."""
    rows, cols = df.shape
    head_str = df.head(5).to_string()
    dtypes_df = pd.DataFrame(
        {
            "tipe_data": df.dtypes.astype(str),
            "non_null_count": df.notna().sum(),
        }
    )

    body = (
        f"Jumlah Baris : {rows}\n"
        f"Jumlah Kolom : {cols}\n"
        f"Daftar Kolom : {list(df.columns)}\n\n"
        f"Lima Data Pertama:\n{head_str}\n\n"
        f"Tipe Data Setiap Kolom:\n{dtypes_df.to_string()}"
    )
    return format_section("1. STRUKTUR DASAR, SAMPEL DATA, & TIPE DATA", body)


def audit_missing_and_duplicates(df: pd.DataFrame) -> str:
    """5 & 6. Memeriksa missing values (jumlah & persentase) serta data duplikat."""
    total_rows = len(df)
    missing_count = df.isna().sum()
    missing_pct = (missing_count / total_rows * 100).round(2)

    missing_summary = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_pct (%)": missing_pct,
        }
    )

    # Pemeriksaan duplikasi baris penuh dan duplikasi identifier siswa
    exact_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else 0

    dup_preview = ""
    if exact_duplicates > 0 and ID_COL in df.columns:
        dup_ids = df.loc[df.duplicated(keep=False), ID_COL].unique().tolist()
        dup_preview = f"\nContoh id_siswa pada baris duplikat: {dup_ids[:10]}"

    body = (
        f"Ringkasan Missing Values:\n{missing_summary.to_string()}\n\n"
        f"Jumlah Baris Duplikat Penuh (Exact Duplicates) : {exact_duplicates}\n"
        f"Jumlah Duplikat pada '{ID_COL}'                : {id_duplicates}"
        f"{dup_preview}"
    )
    return format_section("2. AUDIT MISSING VALUES & DATA DUPLIKAT", body)


def audit_categorical_and_descriptive(df: pd.DataFrame) -> str:
    """7 & 8. Memeriksa nilai unik kolom kategorikal dan statistik deskriptif."""
    cat_lines = []

    # Pemeriksaan nilai unik pada kolom kategorikal bisnis (mis. program)
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        raw_counts = df[col].value_counts(dropna=False)
        # Normalisasi di memori tanpa mengubah nilai NaN menjadi string "NAN"
        clean_series = df[col].apply(
            lambda val: val.strip().upper() if isinstance(val, str) else val
        )
        clean_counts = clean_series.value_counts(dropna=False)

        cat_lines.append(f"Kolom '{col}' (nilai mentah, n_unique={df[col].nunique(dropna=False)}):")
        cat_lines.append(raw_counts.to_string())
        cat_lines.append(
            f"\nKolom '{col}' setelah simulasi strip+upper (hanya di memori untuk audit, "
            f"n_unique={clean_series.nunique(dropna=False)}):"
        )
        cat_lines.append(clean_counts.to_string())

        # Catat nilai string yang mengandung spasi atau kapitalisasi tidak seragam
        inconsistent = [
            val
            for val in df[col].dropna().unique()
            if isinstance(val, str) and val != val.strip().upper()
        ]
        if inconsistent:
            cat_lines.append(
                f"Temuan inkonsistensi format pada '{col}': {inconsistent}"
            )

    # Catatan khusus untuk id_siswa sebagai identifier
    if ID_COL in df.columns:
        cat_lines.append(
            f"\nCatatan Kolom '{ID_COL}': jumlah nilai unik = {df[ID_COL].nunique()} "
            f"dari {len(df)} baris (identifier unik siswa; TIDAK digunakan sebagai fitur prediksi)."
        )

    # Statistik deskriptif untuk seluruh kolom numerik
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc_num = df[numeric_cols].describe().T
    desc_num["median"] = df[numeric_cols].median()
    desc_num = desc_num[["count", "mean", "std", "min", "25%", "median", "75%", "max"]].round(2)

    body = (
        "\n".join(cat_lines)
        + "\n\nStatistik Deskriptif Fitur Numerik & Target:\n"
        + desc_num.to_string()
    )
    return format_section("3. NILAI UNIK KATEGORIKAL & STATISTIK DESKRIPTIF", body)


def audit_target_distribution(df: pd.DataFrame, output_dir: Path) -> str:
    """9. Memeriksa dan memvisualisasikan distribusi kolom target `lulus`."""
    counts = df[TARGET_COL].value_counts().sort_index()
    pcts = (df[TARGET_COL].value_counts(normalize=True).sort_index() * 100).round(2)

    dist_df = pd.DataFrame(
        {
            "label": ["Tidak Lulus (0)" if idx == 0 else "Lulus (1)" for idx in counts.index],
            "jumlah": counts.values,
            "persentase (%)": pcts.values,
        },
        index=counts.index,
    )

    # Simpan grafik distribusi target
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["#e07a5f", "#3d85c6"]
    bars = ax.bar(dist_df["label"], dist_df["jumlah"], color=colors, edgecolor="black", width=0.55)

    for bar, pct in zip(bars, dist_df["persentase (%)"]):
        height = bar.get_height()
        ax.annotate(
            f"{int(height)} ({pct:.2f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Distribusi Target Kelulusan Siswa (lulus)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Status Kelulusan")
    ax.set_ylabel("Jumlah Siswa")
    ax.set_ylim(0, max(dist_df["jumlah"]) * 1.15)
    plt.tight_layout()

    plot_path = output_dir / "distribusi_target_lulus.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    body = (
        f"Distribusi Target '{TARGET_COL}':\n{dist_df.to_string()}\n\n"
        f"Grafik disimpan ke: {plot_path.name}"
    )
    return format_section("4. DISTRIBUSI TARGET KELULUSAN (lulus)", body)


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> Path:
    """10. Membuat grafik distribusi (histogram + KDE dan boxplot) fitur numerik."""
    # Hanya fitur numerik prediktif (tidak memasukkan id_siswa maupun target lulus)
    features = [col for col in PREDICTIVE_NUMERIC_COLS if col in df.columns]
    n_cols = len(features)

    fig, axes = plt.subplots(n_cols, 2, figsize=(12, 3.6 * n_cols))
    if n_cols == 1:
        axes = np.array([axes])

    for idx, col in enumerate(features):
        series = df[col].dropna()

        # Histogram + KDE di kolom kiri
        sns.histplot(series, kde=True, bins=20, ax=axes[idx, 0], color="#4c72b0")
        axes[idx, 0].axvline(series.mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean: {series.mean():.2f}")
        axes[idx, 0].axvline(series.median(), color="green", linestyle="-.", linewidth=1.2, label=f"Median: {series.median():.2f}")
        axes[idx, 0].set_title(f"Distribusi {col}", fontsize=11, fontweight="bold")
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel("Frekuensi")
        axes[idx, 0].legend(fontsize=8)

        # Boxplot di kolom kanan untuk melihat sebaran kuartil dan outlier
        sns.boxplot(x=series, ax=axes[idx, 1], color="#8172b3")
        axes[idx, 1].set_title(f"Boxplot {col}", fontsize=11, fontweight="bold")
        axes[idx, 1].set_xlabel(col)

    plt.suptitle("Distribusi Fitur Numerik Akademik Siswa", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    plot_path = output_dir / "distribusi_fitur_numerik.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def audit_feature_target_relationships(df: pd.DataFrame, output_dir: Path) -> str:
    """
    11 & 13. Memeriksa hubungan awal fitur terhadap target `lulus`.
    Kolom `id_siswa` dikecualikan secara eksplisit sebagai fitur prediksi.
    """
    # Pastikan fitur analisis tidak mengandung id_siswa dan memisahkan nilai_remedial (kandidat leakage)
    numeric_features = [
        col
        for col in PREDICTIVE_NUMERIC_COLS
        if col in df.columns and col not in {ID_COL, TARGET_COL, LEAKAGE_COL}
    ]

    # Statistik agregat fitur numerik berdasarkan status kelulusan
    grouped_stats = (
        df.groupby(TARGET_COL)[numeric_features]
        .agg(["count", "mean", "median", "std"])
        .round(2)
    )

    # Korelasi Pearson antara fitur numerik prediktif dengan target lulus
    corr_matrix = df[numeric_features + [TARGET_COL]].corr(numeric_only=True).round(3)
    target_corr = corr_matrix[TARGET_COL].drop(TARGET_COL).sort_values(ascending=False)

    # Visualisasi 1: Boxplot hubungan fitur numerik prediktif vs target lulus
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()
    for idx, col in enumerate(numeric_features):
        sns.boxplot(
            data=df,
            x=TARGET_COL,
            y=col,
            hue=TARGET_COL,
            palette={0: "#e07a5f", 1: "#3d85c6"},
            legend=False,
            ax=axes_flat[idx],
        )
        axes_flat[idx].set_title(f"{col} vs Status Kelulusan ({TARGET_COL})", fontsize=11, fontweight="bold")
        axes_flat[idx].set_xlabel("lulus (0 = Tidak Lulus, 1 = Lulus)")
        axes_flat[idx].set_ylabel(col)

    plt.suptitle("Hubungan Fitur Numerik terhadap Status Kelulusan", fontsize=14, fontweight="bold")
    plt.tight_layout()
    boxplot_path = output_dir / "hubungan_fitur_numerik_vs_lulus.png"
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)

    # Visualisasi 2: Heatmap korelasi fitur numerik prediktif & target
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Matriks Korelasi Fitur Numerik Prediktif & Target (Tanpa id_siswa)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    heatmap_path = output_dir / "korelasi_fitur_numerik.png"
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)

    # Hubungan kolom kategorikal `program` terhadap target `lulus`
    audit_df = df[[TARGET_COL]].copy()
    audit_df["program_bersih"] = df["program"].apply(
        lambda val: val.strip().upper() if isinstance(val, str) else val
    )

    prog_crosstab = pd.crosstab(audit_df["program_bersih"], audit_df[TARGET_COL], margins=True)
    prog_rate = (
        audit_df.groupby("program_bersih")[TARGET_COL]
        .agg(jumlah_siswa="count", tingkat_kelulusan="mean")
        .assign(tingkat_kelulusan_pct=lambda x: (x["tingkat_kelulusan"] * 100).round(2))
    )

    # Visualisasi 3: Tingkat kelulusan per program
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_data = prog_rate.reset_index().sort_values("tingkat_kelulusan_pct", ascending=False)
    bars = ax.bar(
        plot_data["program_bersih"],
        plot_data["tingkat_kelulusan_pct"],
        color="#55a868",
        edgecolor="black",
        width=0.5,
    )
    for bar, count in zip(bars, plot_data["jumlah_siswa"]):
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f}%\n(n={count})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title("Tingkat Kelulusan Berdasarkan Program (Audit Strip+Upper)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Program")
    ax.set_ylabel("Persentase Lulus (%)")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    prog_plot_path = output_dir / "hubungan_program_vs_lulus.png"
    fig.savefig(prog_plot_path, dpi=150)
    plt.close(fig)

    body = (
        f"Fitur numerik yang dianalisis (mengecualikan '{ID_COL}' dan '{LEAKAGE_COL}'): {numeric_features}\n\n"
        f"Statistik Fitur Numerik berdasarkan '{TARGET_COL}':\n{grouped_stats.to_string()}\n\n"
        f"Korelasi dengan Target '{TARGET_COL}':\n{target_corr.to_string()}\n\n"
        f"Tabulasi Silang 'program' (dinormalisasi di memori untuk audit) vs '{TARGET_COL}':\n"
        f"{prog_crosstab.to_string()}\n\n"
        f"Tingkat Kelulusan per Program:\n{prog_rate.to_string()}\n\n"
        f"Grafik disimpan ke:\n"
        f"- {boxplot_path.name}\n"
        f"- {heatmap_path.name}\n"
        f"- {prog_plot_path.name}"
    )
    return format_section("5. HUBUNGAN AWAL FITUR DENGAN TARGET (TANPA id_siswa)", body)


def audit_target_leakage(df: pd.DataFrame, output_dir: Path) -> str:
    """
    12 & 13. Memeriksa potensi target leakage pada `nilai_remedial`
    serta memverifikasi pengecualian `id_siswa` sebagai fitur prediksi.
    """
    leakage_series = df[LEAKAGE_COL]
    has_remedial = leakage_series.notna()

    # Tabulasi silang keberadaan nilai_remedial terhadap target lulus
    leakage_crosstab = pd.crosstab(
        has_remedial.map({False: "Missing (NaN)", True: "Terisi (Non-NaN)"}),
        df[TARGET_COL].map({0: "Tidak Lulus (0)", 1: "Lulus (1)"}),
        margins=True,
    )

    # Persentase missing values nilai_remedial per kelas target
    missing_by_target = (
        df.groupby(TARGET_COL)[LEAKAGE_COL]
        .agg(
            total_baris="size",
            terisi_count="count",
            missing_count=lambda s: int(s.isna().sum()),
            missing_pct=lambda s: round(float(s.isna().mean() * 100), 2),
        )
    )

    # Statistik deskriptif nilai_remedial untuk baris yang terisi per kelas target
    remedial_stats_by_target = (
        df.groupby(TARGET_COL)[LEAKAGE_COL]
        .describe()
        .round(2)
    )

    # Visualisasi audit target leakage pada nilai_remedial
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ct_plot = pd.crosstab(
        df[TARGET_COL].map({0: "Tidak Lulus (0)", 1: "Lulus (1)"}),
        has_remedial.map({False: "Missing (NaN)", True: "Terisi (Non-NaN)"}),
    )
    ct_plot.plot(kind="bar", stacked=True, ax=axes[0], color=["#999999", "#d95f02"], edgecolor="black")
    axes[0].set_title(f"Keberadaan '{LEAKAGE_COL}' per Kelas Target", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Status Kelulusan")
    axes[0].set_ylabel("Jumlah Siswa")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title=LEAKAGE_COL)

    non_null_remedial = df.dropna(subset=[LEAKAGE_COL])
    if not non_null_remedial.empty:
        sns.histplot(
            data=non_null_remedial,
            x=LEAKAGE_COL,
            hue=TARGET_COL,
            bins=20,
            kde=True,
            ax=axes[1],
            palette={0: "#e07a5f", 1: "#3d85c6"},
        )
        axes[1].set_title(f"Distribusi '{LEAKAGE_COL}' (Hanya Data Terisi)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel(LEAKAGE_COL)
        axes[1].set_ylabel("Frekuensi")

    plt.suptitle("Audit Potensi Target Leakage: nilai_remedial", fontsize=13, fontweight="bold")
    plt.tight_layout()
    leakage_plot_path = output_dir / "audit_leakage_nilai_remedial.png"
    fig.savefig(leakage_plot_path, dpi=150)
    plt.close(fig)

    # Deteksi mekanisme leakage secara objektif dari pola data
    lulus_1_terisi = int(df.loc[df[TARGET_COL] == 1, LEAKAGE_COL].notna().sum())
    lulus_0_missing = int(df.loc[df[TARGET_COL] == 0, LEAKAGE_COL].isna().sum())

    body = (
        f"Tabulasi Silang Keberadaan '{LEAKAGE_COL}' vs '{TARGET_COL}':\n"
        f"{leakage_crosstab.to_string()}\n\n"
        f"Ringkasan Missingness '{LEAKAGE_COL}' per Kelas Target:\n"
        f"{missing_by_target.to_string()}\n\n"
        f"Statistik '{LEAKAGE_COL}' per Kelas Target:\n"
        f"{remedial_stats_by_target.to_string()}\n\n"
        f"Indikator Target Leakage:\n"
        f"- Jumlah siswa Lulus (1) yang memiliki '{LEAKAGE_COL}'       : {lulus_1_terisi}\n"
        f"- Jumlah siswa Tidak Lulus (0) yang '{LEAKAGE_COL}'-nya kosong : {lulus_0_missing}\n"
        f"- Interpretasi: Jika keberadaan/nilai '{LEAKAGE_COL}' hanya muncul setelah "
        f"diketahui siswa tidak lulus (post-outcome), maka kolom ini merupakan target leakage "
        f"dan harus dikeluarkan dari fitur pelatihan model.\n\n"
        f"Kebijakan Fitur Identifier ('{ID_COL}'):\n"
        f"- Kolom '{ID_COL}' bertindak sebagai identitas unik siswa dan tidak memiliki makna "
        f"akademik prediktif, sehingga tidak digunakan sebagai fitur prediksi.\n\n"
        f"Grafik audit leakage disimpan ke: {leakage_plot_path.name}"
    )
    return format_section("6. AUDIT TARGET LEAKAGE (nilai_remedial) & PENGECUALIAN id_siswa", body)


def main() -> None:
    """Menjalankan seluruh rangkaian Data Audit dan EDA serta menyimpan hasilnya ke outputs/."""
    _, data_path, output_dir = get_project_paths()

    # 1. Memuat dataset asli (read-only)
    df = load_dataset(data_path)

    # Menjalankan seluruh tahapan audit dan pembuatan visualisasi EDA
    sections = [
        f"LAPORAN DATA AUDIT & EDA — STUDENT GRADUATION PREDICTION (KELOMPOK B)\n"
        f"Sumber Dataset: {data_path}",
        audit_basic_structure(df),
        audit_missing_and_duplicates(df),
        audit_categorical_and_descriptive(df),
        audit_target_distribution(df, output_dir),
    ]

    # 10. Grafik distribusi fitur numerik
    num_dist_path = plot_numeric_distributions(df, output_dir)
    sections.append(
        format_section(
            "4B. GRAFIK DISTRIBUSI FITUR NUMERIK",
            f"Grafik distribusi (histogram + KDE & boxplot) fitur numerik disimpan ke: {num_dist_path.name}",
        )
    )

    # 11, 12, 13. Hubungan fitur-target dan audit target leakage
    sections.append(audit_feature_target_relationships(df, output_dir))
    sections.append(audit_target_leakage(df, output_dir))

    # Menggabungkan laporan, menampilkan ke terminal, dan menyimpan ke file teks di outputs/
    full_report = "\n".join(sections)
    report_path = output_dir / "data_audit_report.txt"
    report_path.write_text(full_report, encoding="utf-8")

    print(full_report)
    print(f"\nLaporan audit lengkap berhasil disimpan ke: {report_path}")


if __name__ == "__main__":
    main()
