from __future__ import annotations

import numpy as np
import pandas as pd

from config import LABEL_ORDER, RANGES


def _severity(value: float, soft: float, hard: float, direction: str) -> float:
    if direction == "low":
        if value >= soft:
            return 0.0
        return min(1.0, (soft - value) / max(soft - hard, 1e-9))
    if value <= soft:
        return 0.0
    return min(1.0, (value - soft) / max(hard - soft, 1e-9))


def calculate_condition_scores(row: pd.Series) -> dict[str, float]:
    do_score = _severity(row.do_mg_l, RANGES.do_low_mg_l, RANGES.do_critical_mg_l, "low")
    ammonia_score = _severity(row.amonia_mg_l, RANGES.ammonia_high_mg_l, RANGES.ammonia_critical_mg_l, "high")
    nitrite_score = _severity(row.nitrit_mg_l, RANGES.nitrite_high_mg_l, RANGES.nitrite_critical_mg_l, "high")
    nutrient_score = max(
        _severity(row.ec_ms_cm, RANGES.ec_low_ms_cm, 0.72, "low"),
        _severity(row.nitrat_mg_l, RANGES.nitrate_low_mg_l, 8.0, "low"),
    )
    ph_score = max(
        _severity(row.ph_air, RANGES.ph_low_action, 6.05, "low"),
        _severity(row.ph_air, RANGES.ph_high_action, 7.95, "high"),
    )
    level_score = _severity(row.level_air_pct, RANGES.level_low_pct, RANGES.level_critical_pct, "low")
    temp_score = _severity(row.suhu_air_c, RANGES.water_temp_high_c, RANGES.water_temp_critical_c, "high")

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
    if row.amonia_mg_l > 0.55 and row.ph_air > 7.35:
        fish_stress_score += 0.12

    overall = max(
        do_score,
        ammonia_score,
        nitrite_score,
        nutrient_score * 0.82,
        ph_score * 0.86,
        level_score * 0.78,
        temp_score * 0.88,
        fish_stress_score,
    )
    return {
        "Normal": 0.0,
        "DO rendah": do_score,
        "Amonia tinggi": ammonia_score,
        "Nitrit tinggi": nitrite_score,
        "Nutrisi rendah": nutrient_score,
        "pH tidak stabil": ph_score,
        "Level air rendah": level_score,
        "Suhu air tinggi": temp_score,
        "Risiko stres ikan": min(fish_stress_score, 1.0),
        "overall": min(overall, 1.0),
    }


def assign_ai_label(row: pd.Series) -> str:
    scores = calculate_condition_scores(row)
    if scores["overall"] < 0.18:
        return "Normal"

    # Combined fish stress is more useful than a single alarm when several risks co-occur.
    moderate_count = sum(scores[label] >= 0.35 for label in LABEL_ORDER if label not in {"Normal", "Risiko stres ikan"})
    if scores["Risiko stres ikan"] >= 0.45 and moderate_count >= 2:
        return "Risiko stres ikan"
    if scores["Risiko stres ikan"] >= 0.68:
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
    return best if scores[best] >= 0.24 else "Normal"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    score_rows = []
    labels = []
    risk_scores = []
    for _, row in result.iterrows():
        scores = calculate_condition_scores(row)
        label = assign_ai_label(row)
        score_rows.append({f"score_{key}": value for key, value in scores.items() if key != "overall"})
        labels.append(label)
        risk_scores.append(round(float(scores["overall"] * 100), 1))
    score_df = pd.DataFrame(score_rows, index=result.index)
    result = pd.concat([result, score_df], axis=1)
    result["ai_status"] = pd.Categorical(labels, categories=LABEL_ORDER, ordered=False)
    result["risk_score"] = risk_scores
    return result


def recommendation_for_status(row: pd.Series) -> str:
    status = str(row.ai_status)
    if status == "Normal":
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
    status = str(row.ai_status)
    actions = []
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
    counts = df["ai_status"].astype(str).value_counts().reindex(LABEL_ORDER, fill_value=0)
    out = counts.rename_axis("ai_status").reset_index(name="count")
    out["percent"] = np.round(out["count"] / max(len(df), 1) * 100, 2)
    return out
