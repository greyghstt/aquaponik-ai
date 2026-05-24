from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import LABEL_ORDER, PLOTS_DIR


CLASS_COLORS = {
    "Normal": "#2E7D32",
    "DO rendah": "#1565C0",
    "Amonia tinggi": "#C62828",
    "Nitrit tinggi": "#AD1457",
    "Nutrisi rendah": "#EF6C00",
    "pH tidak stabil": "#6A1B9A",
    "Level air rendah": "#00838F",
    "Suhu air tinggi": "#D84315",
    "Risiko stres ikan": "#4E342E",
}


def _duration_days(df: pd.DataFrame) -> int:
    span_days = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400
    return int(round(span_days))


def _setup() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _format_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.grid(True, alpha=0.22)


def plot_sensor_trends(df: pd.DataFrame) -> Path:
    _setup()
    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11), sharex=True)
    time = df["timestamp"]

    axes[0].plot(time, df["suhu_air_c"], color="#D84315", label="Suhu air (C)", linewidth=1.8)
    axes[0].plot(time, df["suhu_udara_c"], color="#F9A825", label="Suhu udara (C)", linewidth=1.25, alpha=0.9)
    axes[0].set_ylabel("Temperatur")
    axes[0].legend(ncol=2, loc="upper left")

    axes[1].plot(time, df["do_mg_l"], color="#1565C0", label="DO (mg/L)", linewidth=1.8)
    axes[1].plot(time, df["amonia_mg_l"], color="#C62828", label="Amonia (mg/L)", linewidth=1.2)
    axes[1].plot(time, df["nitrit_mg_l"], color="#AD1457", label="Nitrit (mg/L)", linewidth=1.2)
    axes[1].axhline(5.0, color="#1565C0", linestyle="--", alpha=0.45)
    axes[1].set_ylabel("Kualitas air")
    axes[1].legend(ncol=3, loc="upper left")

    axes[2].plot(time, df["nitrat_mg_l"], color="#2E7D32", label="Nitrat (mg/L)", linewidth=1.8)
    axes[2].plot(time, df["ec_ms_cm"], color="#EF6C00", label="EC (mS/cm)", linewidth=1.5)
    axes[2].plot(time, df["ph_air"], color="#6A1B9A", label="pH", linewidth=1.3)
    axes[2].set_ylabel("Nutrisi / pH")
    axes[2].legend(ncol=3, loc="upper left")

    axes[3].plot(time, df["level_air_pct"], color="#00838F", label="Level air (%)", linewidth=1.8)
    axes[3].plot(time, df["kelembapan_pct"], color="#455A64", label="Kelembapan (%)", linewidth=1.1, alpha=0.8)
    axes[3].fill_between(time, 0, df["intensitas_cahaya_lux"] / 900, color="#FDD835", alpha=0.24, label="Cahaya/900")
    axes[3].set_ylabel("Greenhouse")
    axes[3].legend(ncol=3, loc="upper left")

    for ax in axes:
        _format_time_axis(ax)
    fig.suptitle(f"Tren Sensor Smart Greenhouse Aquaponik {_duration_days(df)} Hari", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = PLOTS_DIR / "tren_sensor_14_hari.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_dashboard_trends(df: pd.DataFrame) -> Path:
    _setup()
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 6.3), sharex=True)
    time = df["timestamp"]

    axes[0].plot(time, df["suhu_air_c"], color="#D84315", label="Suhu air (C)", linewidth=1.6)
    axes[0].plot(time, df["do_mg_l"], color="#1565C0", label="DO (mg/L)", linewidth=1.5)
    axes[0].axhline(5.0, color="#1565C0", linestyle="--", alpha=0.35)
    axes[0].set_ylabel("Ikan")
    axes[0].legend(ncol=3, loc="upper left")

    axes[1].plot(time, df["amonia_mg_l"], color="#C62828", label="Amonia", linewidth=1.2)
    axes[1].plot(time, df["nitrit_mg_l"], color="#AD1457", label="Nitrit", linewidth=1.2)
    axes[1].plot(time, df["ph_air"], color="#6A1B9A", label="pH", linewidth=1.2)
    axes[1].set_ylabel("Air")
    axes[1].legend(ncol=3, loc="upper left")

    axes[2].plot(time, df["nitrat_mg_l"], color="#2E7D32", label="Nitrat", linewidth=1.4)
    axes[2].plot(time, df["ec_ms_cm"], color="#EF6C00", label="EC", linewidth=1.2)
    axes[2].plot(time, df["level_air_pct"], color="#00838F", label="Level air (%)", linewidth=1.3)
    axes[2].set_ylabel("Nutrisi")
    axes[2].legend(ncol=3, loc="upper left")

    for ax in axes:
        _format_time_axis(ax)
    fig.suptitle(f"Ringkasan Tren Sensor {_duration_days(df)} Hari", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = PLOTS_DIR / "dashboard_tren_sensor.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_status_timeline(df: pd.DataFrame) -> Path:
    _setup()
    fig, ax = plt.subplots(figsize=(13.5, 3.6))
    label_to_y = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    y = [label_to_y[str(label)] for label in df["rf_prediction"]]
    colors = [CLASS_COLORS.get(str(label), "#607D8B") for label in df["rf_prediction"]]
    ax.scatter(df["timestamp"], y, c=colors, s=np.maximum(df["risk_score"] * 1.8, 12), alpha=0.82)
    ax.plot(df["timestamp"], y, color="#263238", linewidth=0.65, alpha=0.35)
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_yticklabels(LABEL_ORDER)
    ax.set_title("Status AI Random Forest terhadap Waktu")
    ax.set_xlabel("Waktu")
    ax.set_ylabel("Kelas AI")
    _format_time_axis(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "status_ai_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_feature_importance(feature_importance: pd.DataFrame) -> Path:
    _setup()
    top = feature_importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(top["feature"], top["importance"], color="#2A6F62")
    ax.set_title("Feature Importance Random Forest")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    out = PLOTS_DIR / "feature_importance_rf.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_class_distribution(df: pd.DataFrame) -> Path:
    _setup()
    target_column = "final_ai_class" if "final_ai_class" in df.columns else "ai_status"
    counts = df[target_column].astype(str).value_counts().reindex(LABEL_ORDER, fill_value=0)
    colors = [CLASS_COLORS[label] for label in counts.index]
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_title("Distribusi Kelas pada Dataset Training")
    ax.set_ylabel("Jumlah data")
    ax.tick_params(axis="x", rotation=32)
    ax.grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    out = PLOTS_DIR / "distribusi_kelas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(confusion: pd.DataFrame) -> Path:
    _setup()
    fig, ax = plt.subplots(figsize=(9.5, 8.4))
    matrix = confusion.to_numpy()
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set_xticks(range(len(confusion.columns)))
    ax.set_yticks(range(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=45, ha="right")
    ax.set_yticklabels(confusion.index)
    ax.set_title("Confusion Matrix Random Forest")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="#102027", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = PLOTS_DIR / "confusion_matrix.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def create_all_plots(
    poster_df: pd.DataFrame,
    training_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
    confusion: pd.DataFrame,
) -> dict[str, str]:
    paths = {
        "sensor_trends": plot_sensor_trends(poster_df),
        "dashboard_trends": plot_dashboard_trends(poster_df),
        "status_timeline": plot_status_timeline(poster_df),
        "feature_importance": plot_feature_importance(feature_importance),
        "class_distribution": plot_class_distribution(training_df),
        "confusion_matrix": plot_confusion_matrix(confusion),
    }
    return {key: str(path) for key, path in paths.items()}
