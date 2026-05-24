from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import LABEL_ORDER, OUTPUT_DIR, PLOTS_DIR, RANGES


def verify_simulation(df: pd.DataFrame, training_df: pd.DataFrame, metrics: dict) -> dict:
    checks: dict[str, object] = {}
    checks["poster_rows"] = int(len(df))
    checks["training_rows"] = int(len(training_df))
    checks["no_missing_values"] = bool(not df.isna().any().any() and not training_df.isna().any().any())
    deltas_minutes = df["timestamp"].diff().dropna().dt.total_seconds() / 60
    expected_interval = float(deltas_minutes.mode().iloc[0])
    checks["timestamp_interval_minutes"] = int(expected_interval)
    checks["timestamp_interval_consistency"] = bool((deltas_minutes == expected_interval).all())

    checks["sensor_ranges_ok"] = bool(
        df["ph_air"].between(5.8, 8.2).all()
        and df["suhu_air_c"].between(22.5, 34.0).all()
        and df["do_mg_l"].between(2.5, 8.6).all()
        and df["amonia_mg_l"].between(0, 1.5).all()
        and df["nitrit_mg_l"].between(0, 1.1).all()
        and df["nitrat_mg_l"].between(4, 90).all()
        and df["ec_ms_cm"].between(0.55, 2.35).all()
        and df["level_air_pct"].between(45, 100).all()
    )
    checks["water_temp_do_negative_corr"] = float(np.round(df["suhu_air_c"].corr(df["do_mg_l"]), 3))
    checks["light_air_temp_positive_corr"] = float(np.round(df["intensitas_cahaya_lux"].corr(df["suhu_udara_c"]), 3))
    checks["nitrate_smoother_than_ammonia"] = bool(df["nitrat_mg_l"].diff().abs().median() < df["amonia_mg_l"].diff().abs().median() * 80)
    checks["all_ai_classes_in_training"] = sorted(training_df["final_ai_class"].astype(str).unique().tolist())
    checks["all_final_classes_present_in_training"] = bool(
        set(LABEL_ORDER).issubset(set(checks["all_ai_classes_in_training"]))
    )
    checks["risk_levels_in_poster"] = sorted(df["risk_level"].astype(str).unique().tolist())
    checks["has_split_risk_columns"] = bool({"risk_score", "risk_level", "final_ai_class"}.issubset(df.columns))
    checks["macro_f1_minimum_met"] = bool(metrics["macro_f1"] >= 0.85)
    checks["accuracy_minimum_met"] = bool(metrics["accuracy"] >= 0.90)

    expected_files = [
        OUTPUT_DIR / "sensor_dataset.csv",
        OUTPUT_DIR / "training_dataset.csv",
        OUTPUT_DIR / "metrics.json",
        OUTPUT_DIR / "dashboard.html",
        OUTPUT_DIR / "rf_model.joblib",
        PLOTS_DIR / "tren_sensor_14_hari.png",
        PLOTS_DIR / "dashboard_tren_sensor.png",
        PLOTS_DIR / "status_ai_timeline.png",
        PLOTS_DIR / "feature_importance_rf.png",
        PLOTS_DIR / "distribusi_kelas.png",
        PLOTS_DIR / "confusion_matrix.png",
    ]
    checks["expected_files_exist"] = {str(path): path.exists() for path in expected_files}
    checks["all_expected_files_exist"] = bool(all(checks["expected_files_exist"].values()))
    checks["domain_thresholds"] = {
        "pH_operasional": f"{RANGES.ph_min_ok}-{RANGES.ph_max_ok}",
        "pH_warning": f"{RANGES.ph_low_warning}-{RANGES.ph_high_warning}",
        "pH_critical": f"<{RANGES.ph_low_critical} atau >{RANGES.ph_high_critical}",
        "do_warning_mg_l": f"{RANGES.do_critical_mg_l}-{RANGES.do_warning_mg_l}",
        "do_critical_mg_l": f"<{RANGES.do_critical_mg_l}",
        "suhu_air_warning_c": f"{RANGES.water_temp_warning_c}-{RANGES.water_temp_critical_c}",
        "suhu_air_critical_c": f">{RANGES.water_temp_critical_c}",
        "amonia_warning_mg_l": f"{RANGES.ammonia_warning_mg_l}-{RANGES.ammonia_critical_mg_l}",
        "nitrit_warning_mg_l": f"{RANGES.nitrite_warning_mg_l}-{RANGES.nitrite_critical_mg_l}",
        "level_warning_pct": f"{RANGES.level_critical_pct}-{RANGES.level_warning_pct}",
        "level_critical_pct": f"<{RANGES.level_critical_pct}",
    }
    checks["verification_passed"] = bool(
        checks["no_missing_values"]
        and checks["timestamp_interval_consistency"]
        and checks["sensor_ranges_ok"]
        and checks["water_temp_do_negative_corr"] < -0.25
        and checks["light_air_temp_positive_corr"] > 0.45
        and checks["has_split_risk_columns"]
        and checks["all_final_classes_present_in_training"]
        and checks["macro_f1_minimum_met"]
        and checks["all_expected_files_exist"]
    )
    return checks


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
