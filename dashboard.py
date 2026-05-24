from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, REFERENCE_LINKS, REFERENCE_NOTES
from labeling import action_for_status, recommendation_for_status
from visualization import CLASS_COLORS


SENSOR_LABELS = {
    "ph_air": "pH",
    "suhu_air_c": "Suhu Air",
    "do_mg_l": "DO",
    "amonia_mg_l": "Amonia",
    "nitrit_mg_l": "Nitrit",
    "nitrat_mg_l": "Nitrat",
    "ec_ms_cm": "EC",
    "tds_ppm": "TDS",
    "level_air_pct": "Level",
    "suhu_udara_c": "Suhu Udara",
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


def _format_sensor_value(key: str, value: float) -> str:
    if key == "intensitas_cahaya_lux":
        return f"{value:,.0f}".replace(",", ".")
    if key in {"amonia_mg_l", "nitrit_mg_l", "ec_ms_cm"}:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _sensor_tiles(latest: pd.Series) -> str:
    tiles = []
    for key, label in SENSOR_LABELS.items():
        tiles.append(
            f"""
            <article class="sensor-tile">
              <span>{escape(label)}</span>
              <strong>{_format_sensor_value(key, latest[key])}</strong>
              <small>{escape(SENSOR_UNITS[key]) or '&nbsp;'}</small>
            </article>
            """
        )
    return "\n".join(tiles)


def _recent_rows(df: pd.DataFrame) -> str:
    rows = []
    for _, row in df.tail(8).iterrows():
        status = str(row["rf_prediction"])
        rows.append(
            f"""
            <tr>
              <td>{row['timestamp'].strftime('%d %b %H:%M')}</td>
              <td>{row['do_mg_l']:.2f}</td>
              <td>{row['suhu_air_c']:.2f}</td>
              <td>{row['ph_air']:.2f}</td>
              <td>{row['risk_score']:.1f}</td>
              <td><span class="pill status-{_status_class(status)}">{escape(status)}</span></td>
            </tr>
            """
        )
    return "\n".join(rows)


def build_dashboard(df: pd.DataFrame, metrics: dict, plot_paths: dict[str, str]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    latest = df.iloc[-1]
    status = str(latest["rf_prediction"])
    risk_level = str(latest.get("risk_level", "Normal"))
    status_color = CLASS_COLORS.get(status, "#2E7D32")
    recommendation = recommendation_for_status(latest)
    action = action_for_status(latest)

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
  <title>Dashboard Smart Greenhouse Aquaponik</title>
  <style>
    :root {{
      --bg: #eef3ea;
      --panel: #fbfaf4;
      --panel-2: #f4f1e7;
      --line: #d8ddd2;
      --ink: #15201c;
      --muted: #64736c;
      --green: #2f7d41;
      --blue: #176b87;
      --amber: #d79a20;
      --red: #b84535;
      --shadow: 0 10px 24px rgba(21,32,28,.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Segoe UI", "Trebuchet MS", sans-serif;
      background:
        linear-gradient(90deg, rgba(21,32,28,.035) 1px, transparent 1px),
        linear-gradient(rgba(21,32,28,.035) 1px, transparent 1px),
        var(--bg);
      background-size: 40px 40px;
    }}
    main {{
      width: min(98vw, 1760px);
      margin: 0 auto;
      padding: 12px 0 20px;
    }}
    .screen {{
      min-height: calc(100vh - 24px);
      display: grid;
      grid-template-rows: auto auto minmax(380px, 1fr);
      gap: 10px;
    }}
    .topbar {{
      display: grid;
      grid-template-columns: 1.1fr .7fr .9fr;
      gap: 10px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
    }}
    .topbar .panel {{ min-height: 0; }}
    .title-panel {{ padding: 16px 18px; }}
    .eyebrow {{
      color: var(--green);
      font-size: .72rem;
      font-weight: 800;
      letter-spacing: .11em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(1.85rem, 3vw, 3.35rem);
      line-height: .98;
      letter-spacing: -.045em;
    }}
    .subtitle {{
      margin: 8px 0 0;
      color: var(--muted);
      font-size: .92rem;
      line-height: 1.45;
      max-width: 850px;
    }}
    .stat-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 12px;
    }}
    .stat {{
      border: 1px solid var(--line);
      background: #fffdf8;
      border-radius: 10px;
      padding: 9px 10px;
    }}
    .stat strong {{ display: block; font-size: 1.25rem; }}
    .stat span {{ color: var(--muted); font-size: .74rem; font-weight: 700; }}
    .status-panel {{
      padding: 12px 14px;
      border-top: 6px solid {status_color};
    }}
    h2 {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: .76rem;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    .status-name {{
      color: {status_color};
      font-size: 2rem;
      line-height: 1;
      font-weight: 800;
      letter-spacing: -.045em;
      margin-bottom: 8px;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    .mini-box {{
      background: #eef5f6;
      border: 1px solid #cbdde0;
      border-radius: 10px;
      padding: 8px 9px;
    }}
    .mini-box span {{ display: block; color: var(--muted); font-size: .72rem; font-weight: 700; }}
    .mini-box strong {{ display: block; margin-top: 2px; font-size: 1.05rem; }}
    .risk-track {{
      height: 8px;
      margin: 8px 0;
      border-radius: 999px;
      background: #dfe5dc;
      overflow: hidden;
    }}
    .risk-track div {{
      width: {latest['risk_score']:.1f}%;
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--amber), var(--red));
    }}
    .action-panel {{
      padding: 12px 14px;
      display: grid;
      align-content: start;
      gap: 8px;
    }}
    .action-box {{
      background: var(--panel-2);
      border: 1px solid #ddd6c5;
      border-radius: 10px;
      padding: 10px 11px;
      color: #4f5f58;
      line-height: 1.45;
      font-size: .86rem;
    }}
    .action-box p {{ margin: 0 0 8px; }}
    .action-box p:last-child {{ margin-bottom: 0; }}
    .updated {{ color: var(--muted); font-size: .78rem; }}
    .sensor-strip {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 8px;
    }}
    .sensor-tile {{
      min-height: 76px;
      padding: 8px 9px;
      background: #fffdf8;
      border: 1px solid var(--line);
      border-radius: 11px;
      box-shadow: 0 5px 14px rgba(21,32,28,.05);
    }}
    .sensor-tile span {{ display: block; color: var(--muted); font-size: .72rem; font-weight: 750; }}
    .sensor-tile strong {{
      display: block;
      margin-top: 6px;
      font-size: clamp(1.05rem, 1.35vw, 1.42rem);
      letter-spacing: -.04em;
    }}
    .sensor-tile small {{ color: var(--muted); font-size: .7rem; }}
    .main-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(430px, .9fr);
      gap: 10px;
      min-height: 0;
    }}
    .chart-panel {{
      padding: 12px;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 8px;
    }}
    .panel-head p {{
      margin: 0;
      color: var(--muted);
      font-size: .82rem;
    }}
    .chart-frame {{
      min-height: 0;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
    }}
    .chart-frame img {{
      display: block;
      width: 100%;
      height: auto;
      min-height: 0;
      object-fit: contain;
    }}
    .side-grid {{
      display: grid;
      grid-template-rows: minmax(160px, .9fr) minmax(220px, 1.1fr);
      gap: 10px;
      min-height: 0;
    }}
    .side-panel {{ padding: 12px; overflow: auto; }}
    .side-panel img {{
      width: 100%;
      max-height: 260px;
      object-fit: contain;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: white;
    }}
    .metric-row {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 7px;
      margin-bottom: 9px;
    }}
    .metric-row div {{
      background: #edf5f3;
      border: 1px solid #d1e0dc;
      border-radius: 9px;
      padding: 8px;
      font-weight: 750;
      font-size: .82rem;
    }}
    .details {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: .8rem; }}
    th, td {{ padding: 7px 6px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-size: .68rem; text-transform: uppercase; }}
    .pill {{ display: inline-block; padding: 4px 7px; border-radius: 999px; font-size: .68rem; font-weight: 800; }}
    {class_css}
    ul {{ margin: 0; padding-left: 18px; color: var(--muted); font-size: .84rem; line-height: 1.45; }}
    a {{ color: var(--blue); }}
    @media (max-width: 1200px) {{
      .topbar, .main-grid, .details {{ grid-template-columns: 1fr; }}
      .sensor-strip {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
      .chart-frame img {{ max-height: none; }}
    }}
    @media (max-width: 640px) {{
      main {{ width: auto; padding: 10px; }}
      .sensor-strip {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .stat-row, .status-grid, .metric-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="screen">
    <div class="topbar">
      <section class="panel title-panel">
        <div class="eyebrow">Dashboard Monitoring Aquaponik</div>
        <h1>Smart Greenhouse Aquaponik</h1>
        <p class="subtitle">Simulasi ikan nila dan selada berbasis sensor numerik, Random Forest Classifier, risk score, rekomendasi, dan tindakan semi-otomatis.</p>
        <div class="stat-row">
          <div class="stat"><strong>12</strong><span>parameter sensor</span></div>
          <div class="stat"><strong>9</strong><span>kelas kondisi</span></div>
          <div class="stat"><strong>{metrics['macro_f1']:.3f}</strong><span>macro F1</span></div>
        </div>
      </section>

      <aside class="panel status-panel">
        <h2>Status Sistem</h2>
        <div class="status-name">{escape(status)}</div>
        <div class="status-grid">
          <div class="mini-box"><span>Confidence RF</span><strong>{latest['rf_confidence_pct']:.1f}%</strong></div>
          <div class="mini-box"><span>Risiko</span><strong>{latest['risk_score']:.1f}/100 · {escape(risk_level)}</strong></div>
        </div>
        <div class="risk-track"><div></div></div>
        <div class="updated">Update: {latest['timestamp'].strftime('%d %B %Y %H:%M')}</div>
      </aside>

      <aside class="panel action-panel">
        <h2>Rekomendasi & Tindakan</h2>
        <div class="action-box">
          <p><strong>Rekomendasi:</strong> {escape(recommendation)}</p>
          <p><strong>Tindakan:</strong> {escape(action)}</p>
        </div>
      </aside>
    </div>

    <section class="sensor-strip">
      {_sensor_tiles(latest)}
    </section>

    <section class="main-grid">
      <article class="panel chart-panel">
        <div class="panel-head">
          <h2>Tren Sensor 20 Hari</h2>
          <p>Grafik utama kualitas air, nutrisi, level air, dan kondisi greenhouse.</p>
        </div>
        <div class="chart-frame">
          <img src="plots/dashboard_tren_sensor.png" alt="Ringkasan tren sensor aquaponik 20 hari">
        </div>
      </article>

      <aside class="side-grid">
        <article class="panel side-panel">
          <h2>Status AI terhadap waktu</h2>
          <img src="plots/status_ai_timeline.png" alt="Timeline status AI">
        </article>
        <article class="panel side-panel">
          <h2>Evaluasi Model</h2>
          <div class="metric-row">
            <div>Accuracy {metrics['accuracy']:.3f}</div>
            <div>Macro F1 {metrics['macro_f1']:.3f}</div>
            <div>Weighted F1 {metrics['weighted_f1']:.3f}</div>
          </div>
          <img src="plots/feature_importance_rf.png" alt="Feature importance Random Forest">
        </article>
      </aside>
    </section>
  </section>

  <section class="details">
    <article class="panel side-panel">
      <h2>Pembacaan Terakhir</h2>
      <table>
        <thead><tr><th>Waktu</th><th>DO</th><th>Suhu</th><th>pH</th><th>Risk</th><th>Status</th></tr></thead>
        <tbody>{_recent_rows(df)}</tbody>
      </table>
    </article>
    <article class="panel side-panel">
      <h2>Catatan Akademik</h2>
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
