from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUT_DIR, PLOTS_DIR, RANGES


def verify_simulation(df: pd.DataFrame, training_df: pd.DataFrame, metrics: dict) -> dict:
    checks: dict[str, object] = {}
    checks["poster_rows"] = int(len(df))
    checks["training_rows"] = int(len(training_df))
    checks["no_missing_values"] = bool(not df.isna().any().any() and not training_df.isna().any().any())
    deltas = df["timestamp"].diff().dropna().dt.total_seconds() / 3600
    checks["hourly_timestamp_consistency"] = bool((deltas == 1).all())

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
    checks["all_ai_classes_in_training"] = sorted(training_df["ai_status"].astype(str).unique().tolist())
    checks["macro_f1_minimum_met"] = bool(metrics["macro_f1"] >= 0.85)
    checks["accuracy_minimum_met"] = bool(metrics["accuracy"] >= 0.90)

    expected_files = [
        OUTPUT_DIR / "sensor_dataset.csv",
        OUTPUT_DIR / "training_dataset.csv",
        OUTPUT_DIR / "metrics.json",
        OUTPUT_DIR / "dashboard.html",
        OUTPUT_DIR / "rf_model.joblib",
        PLOTS_DIR / "tren_sensor_14_hari.png",
        PLOTS_DIR / "status_ai_timeline.png",
        PLOTS_DIR / "feature_importance_rf.png",
        PLOTS_DIR / "distribusi_kelas.png",
        PLOTS_DIR / "confusion_matrix.png",
    ]
    checks["expected_files_exist"] = {str(path): path.exists() for path in expected_files}
    checks["all_expected_files_exist"] = bool(all(checks["expected_files_exist"].values()))
    checks["domain_thresholds"] = {
        "pH_operasional": f"{RANGES.ph_min_ok}-{RANGES.ph_max_ok}",
        "do_low_mg_l": RANGES.do_low_mg_l,
        "suhu_air_tinggi_c": RANGES.water_temp_high_c,
        "amonia_tinggi_mg_l": RANGES.ammonia_high_mg_l,
        "nitrit_tinggi_mg_l": RANGES.nitrite_high_mg_l,
        "ec_target_selada_ms_cm": f"{RANGES.ec_target_low_ms_cm}-{RANGES.ec_target_high_ms_cm}",
    }
    checks["verification_passed"] = bool(
        checks["no_missing_values"]
        and checks["hourly_timestamp_consistency"]
        and checks["sensor_ranges_ok"]
        and checks["water_temp_do_negative_corr"] < -0.25
        and checks["light_air_temp_positive_corr"] > 0.45
        and checks["macro_f1_minimum_met"]
        and checks["all_expected_files_exist"]
    )
    return checks


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
