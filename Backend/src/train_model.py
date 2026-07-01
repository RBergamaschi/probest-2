from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import CLASS_LABELS, load_dataset, validate_dataset


MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "naive_bayes_wdbc.joblib"
FEATURES_PATH = MODEL_DIR / "features.json"
METRICS_PATH = REPORT_DIR / "metrics.json"


def train_model() -> dict:
    bundle = load_dataset(cache=True)
    validate_dataset(bundle.features, bundle.target)

    x_train, x_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=0.2,
        random_state=42,
        stratify=bundle.target,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB()),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    labels = ["B", "M"]
    metrics = {
        "dataset_source": bundle.source,
        "model": "GaussianNB",
        "test_size": 0.2,
        "random_state": 42,
        "features_count": len(bundle.features.columns),
        "features": bundle.features.columns.tolist(),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, labels=labels, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, labels=labels, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, labels=labels, average="macro"),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=[CLASS_LABELS[label] for label in labels],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"real_{CLASS_LABELS[label]}" for label in labels],
            columns=[f"previsto_{CLASS_LABELS[label]}" for label in labels],
        ).to_dict(),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    FEATURES_PATH.write_text(
        json.dumps(
            {
                "features": bundle.features.columns.tolist(),
                "class_labels": CLASS_LABELS,
                "target_column": "Diagnosis",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    result = train_model()
    print("Modelo treinado e salvo em models/naive_bayes_wdbc.joblib")
    print(f"Acuracia: {result['accuracy']:.4f}")
    print(f"Precisao macro: {result['precision_macro']:.4f}")
    print(f"Recall macro: {result['recall_macro']:.4f}")
    print(f"F1 macro: {result['f1_macro']:.4f}")
