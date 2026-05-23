from __future__ import annotations

import json

from config import OUTPUT_DIR, PLOTS_DIR, POSTER_DAYS, REFERENCE_LINKS, REFERENCE_NOTES, TRAINING_DAYS
from dashboard import build_dashboard
from labeling import add_labels, action_for_status, recommendation_for_status, summarize_class_balance
from model_training import predict_system_status, save_model, train_random_forest
from simulator import generate_sensor_data
from verification import save_json, verify_simulation
from visualization import create_all_plots


def _write_poster_summary(latest_row, metrics: dict, verification: dict) -> None:
    lines = [
        "# Ringkasan Poster Grand Design AI Smart Greenhouse Aquaponik",
        "",
        "## Konsep Sistem",
        "sensor -> preprocessing -> AI -> klasifikasi -> rekomendasi -> tindakan semi-otomatis -> dashboard",
        "",
        "## Output AI Terakhir",
        f"- Status: {latest_row['rf_prediction']}",
        f"- Confidence RF: {latest_row['rf_confidence_pct']:.1f}%",
        f"- Skor risiko: {latest_row['risk_score']:.1f}/100",
        f"- Rekomendasi: {recommendation_for_status(latest_row)}",
        f"- Tindakan semi-otomatis: {action_for_status(latest_row)}",
        "",
        "## Evaluasi Model",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Macro F1: {metrics['macro_f1']:.4f}",
        f"- Weighted F1: {metrics['weighted_f1']:.4f}",
        f"- Data training: {metrics['n_train']} baris",
        f"- Data uji: {metrics['n_test']} baris",
        "",
        "## Verifikasi Simulasi",
        f"- Verifikasi lulus: {verification['verification_passed']}",
        f"- Korelasi suhu air vs DO: {verification['water_temp_do_negative_corr']}",
        f"- Korelasi cahaya vs suhu udara: {verification['light_air_temp_positive_corr']}",
        f"- Timestamp per jam konsisten: {verification['hourly_timestamp_consistency']}",
        f"- Rentang sensor realistis: {verification['sensor_ranges_ok']}",
        "",
        "## Catatan Referensi Domain",
    ]
    lines.extend(f"- {note} ({link})" for note, link in zip(REFERENCE_NOTES, REFERENCE_LINKS, strict=False))
    (OUTPUT_DIR / "poster_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    training_raw = generate_sensor_data(TRAINING_DAYS, training=True)
    poster_raw = generate_sensor_data(POSTER_DAYS, training=False)

    training_df = add_labels(training_raw)
    poster_labeled = add_labels(poster_raw)

    result = train_random_forest(training_df)
    save_model(result.model)

    poster_predicted = predict_system_status(result.model, poster_labeled)
    poster_predicted["recommendation"] = poster_predicted.apply(recommendation_for_status, axis=1)
    poster_predicted["semi_auto_action"] = poster_predicted.apply(action_for_status, axis=1)

    plot_paths = create_all_plots(
        poster_df=poster_predicted,
        training_df=training_df,
        feature_importance=result.feature_importance,
        confusion=result.confusion,
    )
    dashboard_path = build_dashboard(poster_predicted, result.metrics, plot_paths)

    training_df.to_csv(OUTPUT_DIR / "training_dataset.csv", index=False)
    poster_predicted.to_csv(OUTPUT_DIR / "sensor_dataset.csv", index=False)
    result.feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    result.confusion.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    summarize_class_balance(training_df).to_csv(OUTPUT_DIR / "class_distribution.csv", index=False)

    metrics = {
        **result.metrics,
        "plot_paths": plot_paths,
        "dashboard_path": str(dashboard_path),
    }
    save_json(metrics, OUTPUT_DIR / "metrics.json")
    verification = verify_simulation(poster_predicted, training_df, result.metrics)
    save_json(verification, OUTPUT_DIR / "verification.json")
    _write_poster_summary(poster_predicted.iloc[-1], result.metrics, verification)

    print("Grand Design AI Smart Greenhouse Aquaponik simulation complete.")
    print(f"Dashboard: {dashboard_path}")
    print(f"Accuracy: {result.metrics['accuracy']:.4f}")
    print(f"Macro F1: {result.metrics['macro_f1']:.4f}")
    print(f"Verification passed: {verification['verification_passed']}")
    print("Latest AI status:", poster_predicted.iloc[-1]["rf_prediction"])


if __name__ == "__main__":
    main()
