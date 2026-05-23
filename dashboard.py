from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, REFERENCE_LINKS, REFERENCE_NOTES
from labeling import action_for_status, recommendation_for_status
from visualization import CLASS_COLORS


SENSOR_LABELS = {
    "ph_air": "pH air",
    "suhu_air_c": "Suhu air",
    "do_mg_l": "DO",
    "amonia_mg_l": "Amonia",
    "nitrit_mg_l": "Nitrit",
    "nitrat_mg_l": "Nitrat",
    "ec_ms_cm": "EC",
    "tds_ppm": "TDS",
    "level_air_pct": "Level air",
    "suhu_udara_c": "Suhu udara",
    "kelembapan_pct": "Kelembapan",
    "intensitas_cahaya_lux": "Cahaya",
}

SENSOR_UNITS = {
    "ph_air": "",
    "suhu_air_c": "C",
    "do_mg_l": "mg/L",
    "amonia_mg_l": "mg/L",
    "nitrit_mg_l": "mg/L",
    "nitrat_mg_l": "mg/L",
    "ec_ms_cm": "mS/cm",
    "tds_ppm": "ppm",
    "level_air_pct": "%",
    "suhu_udara_c": "C",
    "kelembapan_pct": "%",
    "intensitas_cahaya_lux": "lux",
}


def _status_class(status: str) -> str:
    normalized = status.lower().replace(" ", "-").replace("ı", "i")
    return "".join(ch for ch in normalized if ch.isalnum() or ch == "-")


def _sensor_cards(latest: pd.Series) -> str:
    cards = []
    for key, label in SENSOR_LABELS.items():
        value = latest[key]
        if key == "intensitas_cahaya_lux":
            display = f"{value:,.0f}".replace(",", ".")
        elif key in {"amonia_mg_l", "nitrit_mg_l", "ec_ms_cm"}:
            display = f"{value:.3f}"
        else:
            display = f"{value:.2f}"
        cards.append(
            f"""
            <article class="sensor-card">
              <span>{escape(label)}</span>
              <strong>{display}</strong>
              <small>{escape(SENSOR_UNITS[key])}</small>
            </article>
            """
        )
    return "\n".join(cards)


def _recent_rows(df: pd.DataFrame) -> str:
    tail = df.tail(10).copy()
    rows = []
    for _, row in tail.iterrows():
        status = str(row["rf_prediction"])
        rows.append(
            f"""
            <tr>
              <td>{row['timestamp'].strftime('%d %b %H:%M')}</td>
              <td>{row['do_mg_l']:.2f}</td>
              <td>{row['suhu_air_c']:.2f}</td>
              <td>{row['ph_air']:.2f}</td>
              <td>{row['amonia_mg_l']:.3f}</td>
              <td>{row['nitrit_mg_l']:.3f}</td>
              <td><span class="pill status-{_status_class(status)}">{escape(status)}</span></td>
              <td>{row['risk_score']:.1f}</td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_dashboard(df: pd.DataFrame, metrics: dict, plot_paths: dict[str, str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = df.iloc[-1]
    status = str(latest["rf_prediction"])
    recommendation = recommendation_for_status(latest)
    action = action_for_status(latest)
    status_color = CLASS_COLORS.get(status, "#607D8B")

    class_css = "\n".join(
        f".status-{_status_class(label)} {{ background: {color}; color: #fff; }}"
        for label, color in CLASS_COLORS.items()
    )
    references = "\n".join(
        f"<li>{escape(note)} <a href=\"{escape(link)}\">sumber</a></li>"
        for note, link in zip(REFERENCE_NOTES, REFERENCE_LINKS, strict=False)
    )
    html = f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard Simulasi AI Smart Greenhouse Aquaponik</title>
  <style>
    :root {{
      --ink: #17211f;
      --muted: #5f6f6a;
      --paper: #f4f1e8;
      --panel: rgba(255,255,255,0.84);
      --line: rgba(23,33,31,0.12);
      --leaf: #2a6f62;
      --water: #176b87;
      --sun: #f0b429;
      --danger: #c2410c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Aptos", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 8% 12%, rgba(240,180,41,0.28), transparent 28rem),
        radial-gradient(circle at 90% 18%, rgba(23,107,135,0.22), transparent 30rem),
        linear-gradient(135deg, #eef5e8 0%, #f4f1e8 48%, #e7f3f0 100%);
    }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 30px; }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(310px, 0.6fr);
      gap: 22px;
      align-items: stretch;
      margin-bottom: 22px;
    }}
    h1 {{ font-size: clamp(2rem, 4vw, 4.4rem); line-height: 0.95; margin: 0 0 16px; letter-spacing: -0.055em; }}
    h2 {{ margin: 0 0 12px; font-size: 1.08rem; letter-spacing: 0.02em; text-transform: uppercase; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 24px;
      box-shadow: 0 20px 60px rgba(23,33,31,0.09);
      backdrop-filter: blur(10px);
    }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 18px;
    }}
    .flow span {{
      padding: 9px 12px;
      border-radius: 999px;
      background: rgba(42,111,98,0.11);
      border: 1px solid rgba(42,111,98,0.22);
      font-size: 0.9rem;
      font-weight: 700;
    }}
    .status-card {{
      display: grid;
      gap: 14px;
      min-height: 100%;
      border-top: 7px solid {status_color};
    }}
    .status-main {{ font-size: 2rem; font-weight: 900; color: {status_color}; }}
    .risk {{
      height: 18px;
      background: #d7ded9;
      border-radius: 999px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    .risk div {{ height: 100%; width: {latest['risk_score']:.1f}%; background: linear-gradient(90deg, var(--leaf), var(--sun), var(--danger)); }}
    .grid {{ display: grid; gap: 18px; }}
    .sensor-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .sensor-card {{
      padding: 18px;
      border-radius: 22px;
      background: linear-gradient(160deg, rgba(255,255,255,0.95), rgba(255,255,255,0.60));
      border: 1px solid var(--line);
    }}
    .sensor-card span {{ display:block; color: var(--muted); font-size: 0.85rem; }}
    .sensor-card strong {{ display:block; margin-top: 7px; font-size: 1.65rem; letter-spacing: -0.03em; }}
    .sensor-card small {{ color: var(--muted); }}
    .two-col {{ grid-template-columns: 1.05fr 0.95fr; margin-top: 18px; }}
    img {{ width: 100%; border-radius: 18px; border: 1px solid var(--line); background: #fff; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.86rem; }}
    th, td {{ padding: 9px 7px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .pill {{ display: inline-block; padding: 5px 8px; border-radius: 999px; font-weight: 800; font-size: 0.75rem; }}
    {class_css}
    .metric-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .metric-row div {{ background: rgba(23,107,135,0.10); padding: 12px 14px; border-radius: 16px; font-weight: 800; }}
    ul {{ margin: 0; padding-left: 20px; color: var(--muted); }}
    a {{ color: var(--water); }}
    @media (max-width: 980px) {{
      main {{ padding: 18px; }}
      header, .two-col {{ grid-template-columns: 1fr; }}
      .sensor-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 560px) {{
      .sensor-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <section class="hero">
      <h1>Grand Design AI Smart Greenhouse Aquaponik</h1>
      <p>Simulasi end-to-end monitoring ikan nila dan selada: data sensor numerik realistis, preprocessing, Random Forest Classifier, rekomendasi, tindakan semi-otomatis, dan dashboard akhir.</p>
      <div class="flow">
        <span>sensor</span><span>preprocessing</span><span>AI</span><span>klasifikasi</span><span>rekomendasi</span><span>tindakan semi-otomatis</span><span>dashboard</span>
      </div>
    </section>
    <aside class="panel status-card">
      <h2>Status AI realtime</h2>
      <div class="status-main">{escape(status)}</div>
      <div><strong>Confidence RF:</strong> {latest['rf_confidence_pct']:.1f}%</div>
      <div><strong>Skor risiko:</strong> {latest['risk_score']:.1f}/100</div>
      <div class="risk"><div></div></div>
      <p><strong>Rekomendasi:</strong> {escape(recommendation)}</p>
      <p><strong>Tindakan:</strong> {escape(action)}</p>
      <small>Timestamp: {latest['timestamp'].strftime('%d %B %Y %H:%M')}</small>
    </aside>
  </header>

  <section class="grid sensor-grid">
    {_sensor_cards(latest)}
  </section>

  <section class="grid two-col">
    <article class="panel">
      <h2>Grafik utama sensor</h2>
      <img src="plots/tren_sensor_14_hari.png" alt="Tren sensor aquaponik 14 hari">
    </article>
    <article class="panel">
      <h2>Status AI terhadap waktu</h2>
      <img src="plots/status_ai_timeline.png" alt="Timeline status AI">
    </article>
  </section>

  <section class="grid two-col">
    <article class="panel">
      <h2>Feature importance</h2>
      <img src="plots/feature_importance_rf.png" alt="Feature importance Random Forest">
    </article>
    <article class="panel">
      <h2>Evaluasi model</h2>
      <div class="metric-row">
        <div>Accuracy {metrics['accuracy']:.3f}</div>
        <div>Macro F1 {metrics['macro_f1']:.3f}</div>
        <div>Weighted F1 {metrics['weighted_f1']:.3f}</div>
      </div>
      <img src="plots/confusion_matrix.png" alt="Confusion matrix Random Forest" style="margin-top:16px">
    </article>
  </section>

  <section class="grid two-col">
    <article class="panel">
      <h2>Pembacaan terakhir</h2>
      <table>
        <thead><tr><th>Waktu</th><th>DO</th><th>Suhu air</th><th>pH</th><th>NH3</th><th>NO2</th><th>Status</th><th>Risiko</th></tr></thead>
        <tbody>{_recent_rows(df)}</tbody>
      </table>
    </article>
    <article class="panel">
      <h2>Catatan akademik</h2>
      <p>Dataset dummy dibangun dengan dinamika greenhouse: suhu dan cahaya berpola harian, DO menurun saat suhu naik, amonia meningkat setelah pakan, nitrit mengikuti respons biologis, nitrat/EC berubah lambat, dan level air turun bertahap.</p>
      <ul>{references}</ul>
    </article>
  </section>
</main>
</body>
</html>
"""
    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out
