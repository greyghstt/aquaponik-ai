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
    accent_map = {
        "ph_air": "chem",
        "suhu_air_c": "temp",
        "do_mg_l": "water",
        "amonia_mg_l": "alert",
        "nitrit_mg_l": "alert",
        "nitrat_mg_l": "plant",
        "ec_ms_cm": "plant",
        "tds_ppm": "plant",
        "level_air_pct": "water",
        "suhu_udara_c": "temp",
        "kelembapan_pct": "water",
        "intensitas_cahaya_lux": "sun",
    }
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
            <article class="sensor-card sensor-{accent_map[key]}">
              <div class="sensor-top">
                <span>{escape(label)}</span>
                <i></i>
              </div>
              <strong>{display}</strong>
              <small>{escape(SENSOR_UNITS[key]) or '&nbsp;'}</small>
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
              <td>{row['risk_score']:.1f} · {escape(str(row.get('risk_level', 'Normal')))}</td>
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
    risk_label = str(latest.get("risk_level", "Normal"))

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
      --ink: #14201d;
      --muted: #65746f;
      --paper: #f5f2e8;
      --panel: rgba(255,255,255,0.86);
      --panel-solid: #fffdf7;
      --line: rgba(20,32,29,0.12);
      --leaf: #2f7d57;
      --water: #146c94;
      --sun: #e7a928;
      --danger: #bf3b2b;
      --shadow: 0 24px 70px rgba(20,32,29,0.10);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 7% 12%, rgba(231,169,40,0.24), transparent 23rem),
        radial-gradient(circle at 92% 8%, rgba(20,108,148,0.20), transparent 28rem),
        linear-gradient(135deg, #edf5e9 0%, #f5f2e8 46%, #e4f1ee 100%);
      min-height: 100vh;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(20,32,29,0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(20,32,29,0.035) 1px, transparent 1px);
      background-size: 42px 42px;
      mask-image: linear-gradient(to bottom, #000 0%, transparent 82%);
    }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 30px; position: relative; }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(330px, 0.75fr);
      gap: 18px;
      align-items: stretch;
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: clamp(2.2rem, 4.6vw, 4.9rem);
      line-height: 0.92;
      margin: 0 0 16px;
      letter-spacing: -0.07em;
      max-width: 830px;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 0.86rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    p {{ color: var(--muted); line-height: 1.58; margin: 0; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 30px;
      padding: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }}
    .hero {{
      overflow: hidden;
      position: relative;
      min-height: 340px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .hero::after {{
      content: "";
      position: absolute;
      right: -80px;
      bottom: -110px;
      width: 320px;
      height: 320px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(47,125,87,0.20), rgba(20,108,148,0.10) 48%, transparent 70%);
    }}
    .eyebrow {{
      display: inline-flex;
      width: fit-content;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      margin-bottom: 20px;
      border-radius: 999px;
      background: rgba(47,125,87,0.12);
      color: #245d43;
      font-size: 0.78rem;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .hero-copy {{ max-width: 720px; font-size: 1.02rem; }}
    .hero-content, .flow {{ position: relative; z-index: 1; }}
    .quick-stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 22px;
      max-width: 650px;
    }}
    .quick-stats div {{
      padding: 13px 14px;
      border-radius: 18px;
      background: rgba(255,255,255,0.58);
      border: 1px solid var(--line);
    }}
    .quick-stats strong {{ display: block; font-size: 1.34rem; letter-spacing: -0.04em; }}
    .quick-stats span {{ color: var(--muted); font-size: 0.78rem; font-weight: 800; }}
    .flow {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 22px;
    }}
    .flow span {{
      padding: 8px 11px;
      border-radius: 999px;
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      font-size: 0.78rem;
      font-weight: 900;
      color: #314640;
    }}
    .status-card {{
      display: flex;
      flex-direction: column;
      gap: 15px;
      min-height: 100%;
      border-top: 8px solid {status_color};
      background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.74));
    }}
    .status-main {{
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1;
      font-weight: 1000;
      letter-spacing: -0.06em;
      color: {status_color};
    }}
    .status-meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }}
    .status-meta div {{
      padding: 13px;
      border-radius: 18px;
      background: rgba(20,108,148,0.08);
      border: 1px solid rgba(20,108,148,0.12);
    }}
    .status-meta span {{ display: block; color: var(--muted); font-size: 0.76rem; font-weight: 900; }}
    .status-meta strong {{ display: block; margin-top: 4px; font-size: 1.16rem; }}
    .risk {{
      height: 12px;
      background: #dce4df;
      border-radius: 999px;
      overflow: hidden;
      box-shadow: inset 0 0 0 1px var(--line);
    }}
    .risk div {{ height: 100%; width: {latest['risk_score']:.1f}%; background: linear-gradient(90deg, var(--leaf), var(--sun), var(--danger)); }}
    .rec-box {{
      display: grid;
      gap: 10px;
      padding: 16px;
      border-radius: 20px;
      background: rgba(245,242,232,0.78);
      border: 1px solid var(--line);
    }}
    .rec-box p {{ font-size: 0.92rem; }}
    .timestamp {{ color: var(--muted); font-size: 0.84rem; margin-top: auto; }}
    .grid {{ display: grid; gap: 16px; }}
    .sensor-grid {{ grid-template-columns: repeat(6, minmax(0, 1fr)); }}
    .sensor-card {{
      padding: 15px;
      border-radius: 22px;
      background: var(--panel-solid);
      border: 1px solid var(--line);
      box-shadow: 0 10px 26px rgba(20,32,29,0.06);
      min-height: 132px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }}
    .sensor-top {{ display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
    .sensor-top span {{ color: var(--muted); font-size: 0.78rem; font-weight: 900; }}
    .sensor-top i {{ width: 10px; height: 10px; border-radius: 50%; background: var(--leaf); box-shadow: 0 0 0 5px rgba(47,125,87,0.12); }}
    .sensor-card strong {{ display:block; margin-top: 12px; font-size: clamp(1.4rem, 2.2vw, 2rem); letter-spacing: -0.06em; }}
    .sensor-card small {{ color: var(--muted); }}
    .sensor-water .sensor-top i {{ background: var(--water); box-shadow: 0 0 0 5px rgba(20,108,148,0.12); }}
    .sensor-temp .sensor-top i {{ background: var(--danger); box-shadow: 0 0 0 5px rgba(191,59,43,0.12); }}
    .sensor-alert .sensor-top i {{ background: #9f2d20; box-shadow: 0 0 0 5px rgba(159,45,32,0.12); }}
    .sensor-plant .sensor-top i {{ background: var(--leaf); box-shadow: 0 0 0 5px rgba(47,125,87,0.12); }}
    .sensor-sun .sensor-top i {{ background: var(--sun); box-shadow: 0 0 0 5px rgba(231,169,40,0.16); }}
    .sensor-chem .sensor-top i {{ background: #7257a8; box-shadow: 0 0 0 5px rgba(114,87,168,0.12); }}
    .two-col {{ grid-template-columns: 1.1fr 0.9fr; margin-top: 16px; }}
    .wide {{ grid-column: 1 / -1; }}
    .panel-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 12px;
    }}
    .panel-head p {{ max-width: 620px; font-size: 0.9rem; }}
    img {{ width: 100%; border-radius: 20px; border: 1px solid var(--line); background: #fff; display: block; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.86rem; min-width: 720px; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    tbody tr:hover {{ background: rgba(47,125,87,0.06); }}
    th {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .pill {{ display: inline-block; padding: 5px 8px; border-radius: 999px; font-weight: 800; font-size: 0.75rem; }}
    {class_css}
    .metric-row {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .metric-row div {{
      flex: 1 1 120px;
      background: rgba(20,108,148,0.09);
      padding: 14px;
      border-radius: 18px;
      font-weight: 900;
      border: 1px solid rgba(20,108,148,0.11);
    }}
    ul {{ margin: 0; padding-left: 20px; color: var(--muted); line-height: 1.55; }}
    a {{ color: var(--water); }}
    .footer-note {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 0.86rem;
      text-align: center;
    }}
    @media (max-width: 980px) {{
      main {{ padding: 18px; }}
      header, .two-col {{ grid-template-columns: 1fr; }}
      .sensor-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .quick-stats {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 560px) {{
      .sensor-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .status-meta {{ grid-template-columns: 1fr; }}
      .sensor-card {{ min-height: 120px; }}
    }}
  </style>
</head>
<body>
<main>
  <header>
    <section class="hero">
      <div class="hero-content">
        <div class="eyebrow">AI Aquaponik Monitoring</div>
        <h1>Smart Greenhouse Aquaponik</h1>
        <p class="hero-copy">Monitoring ikan nila dan selada dengan sensor numerik, Random Forest Classifier, rekomendasi, dan tindakan semi-otomatis.</p>
        <div class="quick-stats">
          <div><strong>12</strong><span>parameter sensor</span></div>
          <div><strong>9</strong><span>kelas kondisi AI</span></div>
          <div><strong>{metrics['macro_f1']:.3f}</strong><span>macro F1 model</span></div>
        </div>
      </div>
      <div class="flow">
        <span>sensor</span><span>preprocessing</span><span>AI</span><span>klasifikasi</span><span>rekomendasi</span><span>tindakan semi-otomatis</span><span>dashboard</span>
      </div>
    </section>
    <aside class="panel status-card">
      <h2>Status AI realtime</h2>
      <div class="status-main">{escape(status)}</div>
      <div class="status-meta">
        <div><span>Confidence RF</span><strong>{latest['rf_confidence_pct']:.1f}%</strong></div>
        <div><span>Risiko</span><strong>{latest['risk_score']:.1f}/100 · {risk_label}</strong></div>
      </div>
      <div class="risk"><div></div></div>
      <div class="rec-box">
        <p><strong>Rekomendasi:</strong> {escape(recommendation)}</p>
        <p><strong>Tindakan:</strong> {escape(action)}</p>
      </div>
      <div class="timestamp">Update terakhir: {latest['timestamp'].strftime('%d %B %Y %H:%M')}</div>
    </aside>
  </header>

  <section class="grid sensor-grid">
    {_sensor_cards(latest)}
  </section>

  <section class="grid two-col">
    <article class="panel wide">
      <div class="panel-head">
        <h2>Tren Sensor 14 Hari</h2>
        <p>Pola sensor dibuat mengikuti dinamika greenhouse: cahaya dan suhu berpola harian, DO turun saat suhu air meningkat, serta nutrisi berubah bertahap.</p>
      </div>
      <img src="plots/tren_sensor_14_hari.png" alt="Tren sensor aquaponik 14 hari">
    </article>
  </section>

  <section class="grid two-col">
    <article class="panel">
      <div class="panel-head">
        <h2>Status AI</h2>
      </div>
      <img src="plots/status_ai_timeline.png" alt="Timeline status AI">
    </article>
    <article class="panel">
      <div class="panel-head">
        <h2>Feature Importance</h2>
      </div>
      <img src="plots/feature_importance_rf.png" alt="Feature importance Random Forest">
    </article>
  </section>

  <section class="grid two-col">
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
      <div class="table-wrap">
        <table>
          <thead><tr><th>Waktu</th><th>DO</th><th>Suhu air</th><th>pH</th><th>NH3</th><th>NO2</th><th>Status</th><th>Risiko</th></tr></thead>
          <tbody>{_recent_rows(df)}</tbody>
        </table>
      </div>
    </article>
    <article class="panel">
      <h2>Catatan akademik</h2>
      <p>Dataset dummy dibangun dengan dinamika greenhouse: suhu dan cahaya berpola harian, DO menurun saat suhu naik, amonia meningkat setelah pakan, nitrit mengikuti respons biologis, nitrat/EC berubah lambat, dan level air turun bertahap.</p>
      <ul>{references}</ul>
    </article>
  </section>
  <p class="footer-note">Smart greenhouse aquaponik ikan nila + selada · Random Forest Classifier · simulasi sensor numerik realistis</p>
</main>
</body>
</html>
"""
    out = OUTPUT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out
