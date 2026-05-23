from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_ORDER, RANGES


def _severity(value: float, warning: float, critical: float, direction: str) -> float:
    if direction == "low":
        if value >= warning:
            return 0.0
        return min(1.0, (warning - value) / max(warning - critical, 1e-9))
    if value <= warning:
        return 0.0
    return min(1.0, (value - warning) / max(critical - warning, 1e-9))


def _range_severity(
    value: float,
    low_warning: float,
    high_warning: float,
    low_critical: float,
    high_critical: float,
) -> float:
    return max(
        _severity(value, low_warning, low_critical, "low"),
        _severity(value, high_warning, high_critical, "high"),
    )


def _overall_severity(condition_scores: dict[str, float]) -> float:
    """Blend peak severity with cumulative sensor pressure.

    A single critical condition must stay visible, but the overall monitor should
    not behave like a pure max alarm. The weighted average lets several moderate
    warnings accumulate into a higher risk, similar to real control dashboards.
    """

    impact = {
        "DO rendah": 1.00,
        "Amonia tinggi": 1.00,
        "Nitrit tinggi": 1.00,
        "Nutrisi rendah": 0.90,
        "pH tidak stabil": 0.92,
        "Level air rendah": 0.84,
        "Suhu air tinggi": 0.90,
        "Risiko stres ikan": 1.00,
        "Greenhouse warning": 0.45,
    }
    priority_weights = {
        "DO rendah": 1.25,
        "Amonia tinggi": 1.20,
        "Nitrit tinggi": 1.15,
        "Nutrisi rendah": 0.90,
        "pH tidak stabil": 0.95,
        "Level air rendah": 0.85,
        "Suhu air tinggi": 1.05,
        "Risiko stres ikan": 1.30,
        "Greenhouse warning": 0.45,
    }
    adjusted = {name: condition_scores[name] * impact[name] for name in impact}
    weighted_sum = sum(adjusted[name] * weight for name, weight in priority_weights.items())
    weighted_average = weighted_sum / sum(priority_weights.values())
    max_severity = max(adjusted.values())
    return min(0.75 * max_severity + 0.25 * weighted_average, 1.0)


def calculate_condition_scores(row: pd.Series) -> dict[str, float]:
    do_score = _severity(row.do_mg_l, RANGES.do_warning_mg_l, RANGES.do_critical_mg_l, "low")
    ammonia_score = _severity(row.amonia_mg_l, RANGES.ammonia_warning_mg_l, RANGES.ammonia_critical_mg_l, "high")
    nitrite_score = _severity(row.nitrit_mg_l, RANGES.nitrite_warning_mg_l, RANGES.nitrite_critical_mg_l, "high")
    nutrient_score = max(
        _severity(row.nitrat_mg_l, RANGES.nitrate_warning_mg_l, RANGES.nitrate_critical_mg_l, "low"),
        _range_severity(
            row.ec_ms_cm,
            RANGES.ec_low_warning_ms_cm,
            RANGES.ec_high_warning_ms_cm,
            RANGES.ec_low_critical_ms_cm,
            RANGES.ec_high_critical_ms_cm,
        ),
        _range_severity(
            row.tds_ppm,
            RANGES.tds_low_warning_ppm,
            RANGES.tds_high_warning_ppm,
            RANGES.tds_low_critical_ppm,
            RANGES.tds_high_critical_ppm,
        ),
    )
    ph_score = _range_severity(
        row.ph_air,
        RANGES.ph_low_warning,
        RANGES.ph_high_warning,
        RANGES.ph_low_critical,
        RANGES.ph_high_critical,
    )
    level_score = _severity(row.level_air_pct, RANGES.level_warning_pct, RANGES.level_critical_pct, "low")
    temp_score = _severity(row.suhu_air_c, RANGES.water_temp_warning_c, RANGES.water_temp_critical_c, "high")
    greenhouse_score = max(
        _severity(row.suhu_udara_c, RANGES.air_temp_warning_c, RANGES.air_temp_critical_c, "high"),
        _range_severity(
            row.kelembapan_pct,
            RANGES.humidity_low_warning_pct,
            RANGES.humidity_high_warning_pct,
            RANGES.humidity_low_critical_pct,
            RANGES.humidity_high_critical_pct,
        ),
    )

    fish_stress_score = (
        0.30 * do_score
        + 0.22 * temp_score
        + 0.20 * ammonia_score
        + 0.17 * nitrite_score
        + 0.07 * ph_score
        + 0.04 * level_score
    )
    if row.do_mg_l < 4.6 and row.suhu_air_c > 30.0:
        fish_stress_score += 0.18
    if row.amonia_mg_l > 0.75 and row.ph_air > 7.5:
        fish_stress_score += 0.12

    condition_scores = {
        "Normal": 0.0,
        "DO rendah": do_score,
        "Amonia tinggi": ammonia_score,
        "Nitrit tinggi": nitrite_score,
        "Nutrisi rendah": nutrient_score,
        "pH tidak stabil": ph_score,
        "Level air rendah": level_score,
        "Suhu air tinggi": temp_score,
        "Risiko stres ikan": min(fish_stress_score, 1.0),
        "Greenhouse warning": greenhouse_score,
    }
    condition_scores["overall"] = _overall_severity({key: value for key, value in condition_scores.items() if key != "Normal"})
    return condition_scores


def assign_risk_level(risk_score: float) -> str:
    if risk_score >= 70:
        return "Critical"
    if risk_score >= 30:
        return "Warning"
    return "Normal"


def assign_ai_label(row: pd.Series) -> str:
    scores = calculate_condition_scores(row)

    # Combined fish stress is more useful than a single alarm when several risks co-occur.
    moderate_count = sum(scores[label] >= 0.45 for label in LABEL_ORDER if label not in {"Normal", "Risiko stres ikan"})
    if scores["Risiko stres ikan"] >= 0.55 and moderate_count >= 2:
        return "Risiko stres ikan"
    if scores["Risiko stres ikan"] >= 0.72:
        return "Risiko stres ikan"

    priority = [
        "Amonia tinggi",
        "Nitrit tinggi",
        "DO rendah",
        "Suhu air tinggi",
        "pH tidak stabil",
        "Level air rendah",
        "Nutrisi rendah",
    ]
    best = max(priority, key=lambda label: scores[label])
    if scores[best] >= 0.85:
        return best
    if scores["overall"] < 0.70:
        return "Normal"
    return best if scores[best] >= 0.70 else "Normal"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    score_rows = []
    labels = []
    risk_scores = []
    risk_levels = []
    for _, row in result.iterrows():
        scores = calculate_condition_scores(row)
        label = assign_ai_label(row)
        score_rows.append({f"score_{key}": value for key, value in scores.items() if key != "overall"})
        labels.append(label)
        risk_score = round(float(scores["overall"] * 100), 1)
        risk_scores.append(risk_score)
        risk_levels.append(assign_risk_level(risk_score))
    score_df = pd.DataFrame(score_rows, index=result.index)
    result = pd.concat([result, score_df], axis=1)
    result["risk_score"] = risk_scores
    result["risk_level"] = risk_levels
    result["final_ai_class"] = pd.Categorical(labels, categories=LABEL_ORDER, ordered=False)
    result["ai_status"] = result["final_ai_class"]
    return result


def recommendation_for_status(row: pd.Series) -> str:
    status = str(row.get("final_ai_class", row.get("ai_status", "Normal")))
    risk_level = str(row.get("risk_level", "Normal"))
    if status == "Normal":
        if risk_level == "Warning":
            return "Ada parameter mendekati batas; lanjutkan monitoring dan validasi sensor sebelum tindakan besar."
        return "Pertahankan pemantauan; sistem berada dalam rentang aman untuk ikan nila dan selada."
    if status == "DO rendah":
        return "Tingkatkan aerasi, cek diffuser, dan kurangi pemberian pakan sementara."
    if status == "Amonia tinggi":
        return "Hentikan pakan sementara, cek biofilter, dan lakukan pergantian air parsial terukur."
    if status == "Nitrit tinggi":
        return "Cek kematangan biofilter, tambahkan aerasi, dan validasi siklus nitrifikasi."
    if status == "Nutrisi rendah":
        return "Cek EC/TDS dan nitrat; tambahkan nutrisi ramah ikan secara bertahap bila diperlukan."
    if status == "pH tidak stabil":
        return "Kalibrasi sensor pH dan koreksi pH perlahan; hindari perubahan mendadak."
    if status == "Level air rendah":
        return "Tambahkan air deklorinasi dan cek evaporasi, kebocoran, atau sumbatan aliran."
    if status == "Suhu air tinggi":
        return "Aktifkan ventilasi/kipas, kurangi panas greenhouse, dan pantau DO lebih sering."
    if status == "Risiko stres ikan":
        return "Prioritaskan keselamatan ikan: aerasi maksimum, kurangi pakan, dan inspeksi kualitas air manual."
    return "Perlu inspeksi operator."


def action_for_status(row: pd.Series) -> str:
    status = str(row.get("final_ai_class", row.get("ai_status", "Normal")))
    risk_level = str(row.get("risk_level", "Normal"))
    actions = []
    if risk_level == "Warning":
        actions.append("monitoring intensif")
    if status in {"DO rendah", "Amonia tinggi", "Nitrit tinggi", "Suhu air tinggi", "Risiko stres ikan"}:
        actions.append("aerator ON")
    if status in {"Suhu air tinggi", "Risiko stres ikan"} or row.suhu_udara_c >= 31.5:
        actions.append("kipas ON")
    if status in {"Level air rendah", "Nutrisi rendah"}:
        actions.append("pompa ON terbatas")
    if status in {"Amonia tinggi", "Nitrit tinggi", "pH tidak stabil", "Risiko stres ikan"}:
        actions.append("rekomendasi pengguna untuk tindakan sensitif")
    if not actions:
        actions.append("monitoring normal")
    return " | ".join(actions)


def summarize_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    target_column = "final_ai_class" if "final_ai_class" in df.columns else "ai_status"
    counts = df[target_column].astype(str).value_counts().reindex(LABEL_ORDER, fill_value=0)
    out = counts.rename_axis("final_ai_class").reset_index(name="count")
    out["percent"] = np.round(out["count"] / max(len(df), 1) * 100, 2)
    return out
