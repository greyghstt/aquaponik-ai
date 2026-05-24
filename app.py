from __future__ import annotations

import html
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from labeling import action_for_status, recommendation_for_status


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"

SENSORS = [
    ("ph_air", "pH", ""),
    ("suhu_air_c", "Suhu Air", "C"),
    ("do_mg_l", "DO", "mg/L"),
    ("amonia_mg_l", "Amonia", "mg/L"),
    ("nitrit_mg_l", "Nitrit", "mg/L"),
    ("nitrat_mg_l", "Nitrat", "mg/L"),
    ("ec_ms_cm", "EC", "mS/cm"),
    ("tds_ppm", "TDS", "ppm"),
    ("level_air_pct", "Level Air", "%"),
    ("suhu_udara_c", "Suhu Udara", "C"),
    ("kelembapan_pct", "Kelembapan", "%"),
    ("intensitas_cahaya_lux", "Cahaya", "lux"),
]

TREND_SENSORS = {
    "do_mg_l": ("DO", 3.0, 8.0),
    "suhu_air_c": ("Suhu Air", 23.0, 34.0),
    "ph_air": ("pH", 6.0, 8.2),
    "amonia_mg_l": ("Amonia", 0.0, 1.4),
}


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 18% 12%, rgba(215, 232, 204, 0.62), transparent 28%),
                radial-gradient(circle at 82% 8%, rgba(238, 222, 181, 0.42), transparent 26%),
                #f3f6ef;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1580px;
            padding-top: 2rem;
            padding-bottom: 1.1rem;
        }
        [data-testid="stVerticalBlock"] {
            gap: 0.72rem;
        }
        [data-testid="stHorizontalBlock"] {
            gap: 0.85rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 253, 247, 0.95);
            border: 1px solid #d8dfd1;
            border-radius: 18px;
            box-shadow: 0 10px 28px rgba(34, 54, 42, 0.065);
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1rem 1.05rem;
        }
        h2, h3 {
            letter-spacing: -0.035em;
        }
        .app-kicker {
            color: #476057;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 0.12rem;
        }
        .app-title {
            color: #071815;
            font-size: 2.45rem;
            font-weight: 900;
            letter-spacing: -0.035em;
            line-height: 1.05;
            margin-bottom: 0.48rem;
        }
        .title-badge {
            display: inline-block;
            color: #157235;
        }
        .app-subtitle {
            color: #263f36;
            font-size: 0.95rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            margin-top: 0.1rem;
            margin-bottom: 0.25rem;
        }
        div[data-testid="stCaptionContainer"] p {
            color: #263f36;
            font-weight: 800;
            letter-spacing: 0.085em;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 253, 247, 0.92);
            border: 1px solid #d8dfd1;
            border-radius: 15px;
            padding: 0.65rem 0.8rem;
            box-shadow: 0 6px 16px rgba(34, 54, 42, 0.045);
        }
        [data-testid="stMetricLabel"] p {
            font-size: 0.78rem;
            color: #263f36;
            font-weight: 800;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.65rem;
            letter-spacing: -0.04em;
        }
        .status-title {
            margin: 0.05rem 0 0.1rem 0;
            color: #157235;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.04em;
        }
        .status-line {
            color: #10231f;
            font-size: 0.96rem;
            margin: 0.15rem 0;
        }
        .note-text {
            color: #10231f;
            line-height: 1.55;
            margin: 0;
        }
        .actuator-list {
            display: grid;
            gap: 0.35rem;
            margin-top: 0.15rem;
        }
        .actuator-item {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            border-bottom: 1px solid rgba(216, 223, 209, 0.7);
            padding: 0.28rem 0;
            font-size: 0.92rem;
        }
        .actuator-item:last-child {
            border-bottom: 0;
        }
        .actuator-state {
            font-weight: 800;
            color: #157235;
            white-space: nowrap;
        }
        .sensor-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.42rem;
            margin-top: 0.45rem;
            padding-bottom: 0.28rem;
        }
        .sensor-card {
            border: 1px solid #d8dfd1;
            border-radius: 13px;
            background: #fffdf7;
            padding: 0.45rem 0.64rem;
            min-height: 48px;
        }
        .sensor-label {
            color: #263f36;
            font-size: 0.74rem;
            font-weight: 700;
            margin-bottom: 0.12rem;
        }
        .sensor-value {
            color: #071815;
            font-size: 0.94rem;
            font-weight: 800;
            white-space: nowrap;
        }
        .flow-line {
            color: #60716a;
            font-size: 0.82rem;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    sensor_df = pd.read_csv(OUTPUTS / "sensor_dataset.csv", parse_dates=["timestamp"])
    class_df = pd.read_csv(OUTPUTS / "class_distribution.csv")
    feature_df = pd.read_csv(OUTPUTS / "feature_importance.csv")
    metrics = json.loads((OUTPUTS / "metrics.json").read_text(encoding="utf-8"))
    verification = json.loads((OUTPUTS / "verification.json").read_text(encoding="utf-8"))
    return sensor_df, class_df, feature_df, metrics, verification


def format_value(key: str, value: float, unit: str) -> str:
    if key == "intensitas_cahaya_lux":
        number = f"{value:,.0f}".replace(",", ".")
    elif key in {"amonia_mg_l", "nitrit_mg_l", "ec_ms_cm"}:
        number = f"{value:.3f}"
    else:
        number = f"{value:.2f}"
    return f"{number} {unit}".strip()


def build_sensor_table(latest: pd.Series) -> pd.DataFrame:
    rows = []
    for key, label, unit in SENSORS:
        rows.append({"Sensor": label, "Nilai": format_value(key, float(latest[key]), unit)})
    return pd.DataFrame(rows)


def actuator_states(status: str) -> dict[str, str]:
    aerator = "ON" if status in {"DO rendah", "Risiko stres ikan"} else "Standby"
    fan = "ON" if status in {"Suhu air tinggi", "Risiko stres ikan"} else "Standby"
    pump = "Terbatas" if status == "Level air rendah" else "Standby"
    manual = "Cek pengguna" if status in {"Amonia tinggi", "Nitrit tinggi", "pH tidak stabil", "Nutrisi rendah"} else "Aman"
    return {
        "Aerator": aerator,
        "Kipas/Ventilasi": fan,
        "Pompa": pump,
        "Manual": manual,
    }


def render_actuators(status: str) -> None:
    rows = []
    for label, state in actuator_states(status).items():
        rows.append(
            "<div class='actuator-item'>"
            f"<span>{html.escape(label)}</span>"
            f"<span class='actuator-state'>{html.escape(state)}</span>"
            "</div>"
        )
    st.markdown(f"<div class='actuator-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_sensor_cards(latest: pd.Series) -> None:
    cards = []
    for key, label, unit in SENSORS:
        cards.append(
            "<div class='sensor-card'>"
            f"<div class='sensor-label'>{html.escape(label)}</div>"
            f"<div class='sensor-value'>{html.escape(format_value(key, float(latest[key]), unit))}</div>"
            "</div>"
        )
    st.markdown(f"<div class='sensor-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def build_event_history(sensor_df: pd.DataFrame, interval_minutes: int) -> pd.DataFrame:
    event_df = sensor_df[
        (sensor_df["rf_prediction"] != "Normal")
        | (sensor_df["risk_score"] >= 40)
    ].copy()
    if event_df.empty:
        return pd.DataFrame(
            columns=["Mulai", "Selesai", "Durasi", "Event", "Level", "Risk Maks", "Tindakan", "Rekomendasi"]
        )

    event_df = event_df.sort_values("timestamp").reset_index(drop=True)
    time_gap = event_df["timestamp"].diff().dt.total_seconds().div(60).gt(interval_minutes * 1.5)
    status_changed = event_df["rf_prediction"].ne(event_df["rf_prediction"].shift())
    event_df["event_id"] = (time_gap | status_changed).cumsum()

    rows = []
    severity_rank = {"Normal": 0, "Warning": 1, "Critical": 2}
    for _, group in event_df.groupby("event_id", sort=True):
        start = group["timestamp"].iloc[0]
        end = group["timestamp"].iloc[-1]
        duration_minutes = int((end - start).total_seconds() / 60) + interval_minutes
        level = max(group["risk_level"], key=lambda value: severity_rank.get(str(value), 0))
        peak = group.loc[group["risk_score"].idxmax()]
        event_name = str(peak["rf_prediction"])
        if event_name == "Normal":
            event_name = "Peringatan agregat"
        action = str(group["semi_auto_action"].mode().iat[0])
        recommendation = str(group["recommendation"].mode().iat[0])
        rows.append(
            {
                "Mulai": start.strftime("%d %b %H:%M"),
                "Selesai": end.strftime("%d %b %H:%M"),
                "Durasi": f"{duration_minutes // 60}j {duration_minutes % 60}m",
                "Event": event_name,
                "Level": level,
                "Risk Maks": f"{float(group['risk_score'].max()):.1f}/100",
                "Tindakan": action,
                "Rekomendasi": recommendation,
            }
        )

    return pd.DataFrame(rows).sort_values(["Mulai", "Event"]).reset_index(drop=True)


def build_trend_chart(sensor_df: pd.DataFrame) -> alt.Chart:
    interval_minutes = max(int(sensor_df["timestamp"].diff().dropna().dt.total_seconds().median() / 60), 1)
    samples_per_hour = max(60 // interval_minutes, 1)
    recent = sensor_df.tail(72 * samples_per_hour)

    rows = []
    for key, (label, low, high) in TREND_SENSORS.items():
        scaled = ((recent[key] - low) / (high - low) * 100).clip(0, 100)
        rows.append(
            pd.DataFrame(
                {
                    "timestamp": recent["timestamp"],
                    "parameter": label,
                    "indeks": scaled,
                }
            )
        )
    chart_df = pd.concat(rows, ignore_index=True)

    return (
        alt.Chart(chart_df)
        .mark_line(strokeWidth=2.4)
        .encode(
            x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b", tickCount=5)),
            y=alt.Y("indeks:Q", title="Skala relatif", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "parameter:N",
                title=None,
                legend=alt.Legend(orient="right"),
                scale=alt.Scale(range=["#2f6f4e", "#c47a2c", "#4976a8", "#a64b3c"]),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Waktu", format="%d %b %H:%M"),
                alt.Tooltip("parameter:N", title="Sensor"),
                alt.Tooltip("indeks:Q", title="Indeks", format=".1f"),
            ],
        )
        .properties(height=342)
    )


def risk_level_from_score(score: float) -> str:
    if score >= 70:
        return "Critical"
    if score >= 40:
        return "Warning"
    return "Normal"


def build_risk_chart(sensor_df: pd.DataFrame) -> alt.LayerChart:
    chart_df = (
        sensor_df.set_index("timestamp")
        .resample("1D")
        .agg(
            risk_max=("risk_score", "max"),
            risk_mean=("risk_score", "mean"),
            status_ai=("rf_prediction", lambda values: values.mode().iat[0]),
        )
        .reset_index()
    )
    chart_df["level"] = chart_df["risk_max"].apply(risk_level_from_score)

    base = alt.Chart(chart_df).encode(
        x=alt.X("timestamp:T", title=None, axis=alt.Axis(format="%d %b", tickCount=8)),
        tooltip=[
            alt.Tooltip("timestamp:T", title="Tanggal", format="%d %b"),
            alt.Tooltip("risk_max:Q", title="Risk maksimum", format=".1f"),
            alt.Tooltip("risk_mean:Q", title="Risk rata-rata", format=".1f"),
            alt.Tooltip("level:N", title="Level"),
            alt.Tooltip("status_ai:N", title="Status dominan"),
        ],
    )

    bars = base.mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, opacity=0.82).encode(
        y=alt.Y("risk_max:Q", title="Risk score", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "level:N",
            title=None,
            legend=alt.Legend(orient="top", direction="horizontal"),
            scale=alt.Scale(
                domain=["Normal", "Warning", "Critical"],
                range=["#2f6f4e", "#c47a2c", "#a64b3c"],
            ),
        ),
    )

    mean_line = base.mark_line(strokeWidth=2.8, color="#17211f").encode(
        y="risk_mean:Q",
    )

    thresholds = pd.DataFrame({"risk_score": [40, 70], "label": ["warning", "critical"]})
    rules = (
        alt.Chart(thresholds)
        .mark_rule(strokeDash=[5, 5], color="#9aa69d")
        .encode(y="risk_score:Q")
    )

    return (rules + bars + mean_line).properties(height=420)


def main() -> None:
    st.set_page_config(page_title="Aquaponik AI", layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    try:
        sensor_df, class_df, feature_df, metrics, verification = load_outputs()
    except FileNotFoundError:
        st.error("Output belum tersedia. Jalankan `rtk python main.py` terlebih dahulu.")
        st.stop()

    latest = sensor_df.iloc[-1]
    status = str(latest["rf_prediction"])
    risk_level = str(latest["risk_level"])
    recommendation = recommendation_for_status(latest)
    action = action_for_status(latest)
    simulation_days = int(round((sensor_df["timestamp"].max() - sensor_df["timestamp"].min()).total_seconds() / 86400))

    interval_minutes = int(verification["timestamp_interval_minutes"])
    title_col, time_col = st.columns([1.25, 1.0])
    with title_col:
        st.markdown(
            "<div class='app-title'>Dashboard <span class='title-badge'>Aquaponik AI</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='app-subtitle'>Ikan nila + selada | {simulation_days} hari | "
            f"interval {interval_minutes} menit | 12 sensor | 9 kelas AI</div>",
            unsafe_allow_html=True,
        )
    with time_col:
        st.caption(f"Update terakhir: {latest['timestamp'].strftime('%d %b %Y %H:%M')}")

    left, center, right = st.columns([0.92, 1.55, 1.08], gap="small")

    with left:
        with st.container(border=True):
            st.caption("KEPUTUSAN MODEL")
            st.markdown(f"<div class='status-title'>{html.escape(status)}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<p class='status-line'>Confidence {latest['rf_confidence_pct']:.1f}%</p>",
                unsafe_allow_html=True,
            )
            st.progress(
                int(min(max(float(latest["risk_score"]), 0), 100)),
                text=f"Risk {latest['risk_score']:.1f}/100 | {risk_level}",
            )
        with st.container(border=True):
            st.caption("REKOMENDASI")
            st.markdown(
                f"<p class='note-text'><b>Rekomendasi:</b> {html.escape(recommendation)}</p>",
                unsafe_allow_html=True,
            )
        with st.container(border=True):
            st.caption("TINDAKAN SISTEM")
            st.markdown(f"<p class='note-text'><b>Tindakan:</b> {html.escape(action)}</p>", unsafe_allow_html=True)
        with st.container(border=True):
            st.caption("AKTUATOR")
            render_actuators(status)

    with center:
        with st.container(border=True):
            st.caption("RINGKASAN RISIKO AI 20 HARI")
            st.write("Bar menunjukkan risiko maksimum harian, garis menunjukkan rata-rata risiko harian.")
            st.altair_chart(build_risk_chart(sensor_df), use_container_width=True)
        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("Sensor", "12")
        metric_cols[1].metric("Kelas", "9")
        metric_cols[2].metric("Accuracy", f"{metrics['accuracy']:.4f}")
        metric_cols[3].metric("Macro F1", f"{metrics['macro_f1']:.4f}")

    with right:
        with st.container(border=True):
            st.subheader("Sensor Realtime")
            render_sensor_cards(latest)

    tab_model, tab_data = st.tabs(["Evaluasi Model", "Data & Verifikasi"])

    with tab_model:
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.subheader(f"Status AI {simulation_days} Hari")
            st.image(str(PLOTS / "status_ai_timeline.png"), use_container_width=True)
            st.subheader("Feature Importance")
            st.image(str(PLOTS / "feature_importance_rf.png"), use_container_width=True)
            st.dataframe(feature_df, use_container_width=True, hide_index=True)
        with col_b:
            st.subheader("Confusion Matrix")
            st.image(str(PLOTS / "confusion_matrix.png"), use_container_width=True)
            st.subheader("Distribusi Kelas")
            st.image(str(PLOTS / "distribusi_kelas.png"), use_container_width=True)

    with tab_data:
        event_history = build_event_history(sensor_df, interval_minutes)
        st.subheader("History Event")
        st.caption("Ringkasan kejadian ketika sistem masuk warning/critical atau kelas AI bukan Normal.")
        st.dataframe(event_history, use_container_width=True, hide_index=True, height=260)

        col_a, col_b = st.columns([0.8, 1.2], gap="medium")
        with col_a:
            st.subheader("Distribusi Kelas")
            st.dataframe(class_df, use_container_width=True, hide_index=True)
            st.subheader("Verifikasi")
            st.json(
                {
                    "verification_passed": verification["verification_passed"],
                    "timestamp_interval_minutes": verification["timestamp_interval_minutes"],
                    "risk_levels": verification["risk_levels_in_poster"],
                    "water_temp_do_corr": verification["water_temp_do_negative_corr"],
                    "light_air_temp_corr": verification["light_air_temp_positive_corr"],
                    "all_final_classes_present": verification["all_final_classes_present_in_training"],
                }
            )
        with col_b:
            st.subheader("Pembacaan Terakhir")
            st.dataframe(
                sensor_df.tail(10)[
                    ["timestamp", "do_mg_l", "suhu_air_c", "ph_air", "risk_score", "risk_level", "rf_prediction"]
                ],
                use_container_width=True,
                hide_index=True,
                height=250,
            )
            st.subheader("Dataset Simulasi")
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
