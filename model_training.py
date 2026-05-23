from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config import LABEL_ORDER, MODEL_FEATURES, MODEL_PATH


@dataclass(frozen=True)
class TrainingResult:
    model: RandomForestClassifier
    metrics: dict
    confusion: pd.DataFrame
    feature_importance: pd.DataFrame
    test_predictions: pd.DataFrame


def train_random_forest(df: pd.DataFrame) -> TrainingResult:
    X = df[MODEL_FEATURES]
    y = df["ai_status"].astype(str)
    counts = y.value_counts()
    stratify = y if counts.min() >= 2 and len(counts) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    model = RandomForestClassifier(
        n_estimators=420,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    labels_present = [label for label in LABEL_ORDER if label in set(y_test) or label in set(y_pred)]
    report = classification_report(y_test, y_pred, labels=labels_present, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, labels=labels_present, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "labels_present": labels_present,
        "classification_report": report,
    }
    confusion = pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels_present),
        index=labels_present,
        columns=labels_present,
    )
    feature_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "importance": np.round(model.feature_importances_, 5),
        }
    ).sort_values("importance", ascending=False)
    test_predictions = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": y_pred})
    return TrainingResult(model, metrics, confusion, feature_importance, test_predictions)


def save_model(model: RandomForestClassifier) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def predict_system_status(model: RandomForestClassifier, df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    raw_predictions = model.predict(result[MODEL_FEATURES])
    result["rf_raw_prediction"] = raw_predictions
    result["rf_prediction"] = raw_predictions
    probabilities = model.predict_proba(result[MODEL_FEATURES])
    class_names = list(model.classes_)
    result["rf_confidence_pct"] = np.round(probabilities.max(axis=1) * 100, 1)
    for class_name in class_names:
        result[f"prob_{class_name}"] = np.round(probabilities[:, class_names.index(class_name)] * 100, 1)

    # Safety layer: keep RF as the classifier, but prevent obvious domain-risk states
    # from being hidden as "Normal" in the operational dashboard.
    if "ai_status" in result.columns and "risk_score" in result.columns:
        hybrid_label = result["ai_status"].astype(str)
        override_mask = (result["rf_prediction"] == "Normal") & (hybrid_label != "Normal") & (result["risk_score"] >= 20)
        result.loc[override_mask, "rf_prediction"] = hybrid_label[override_mask]
        result.loc[override_mask, "rf_confidence_pct"] = np.maximum(
            result.loc[override_mask, "rf_confidence_pct"],
            np.round(68 + result.loc[override_mask, "risk_score"] * 0.25, 1),
        )
    return result
