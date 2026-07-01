from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from src.data import CLASS_LABELS, TARGET_COLUMN, load_dataset, validate_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "naive_bayes_wdbc.joblib"
FEATURES_PATH = BASE_DIR / "models" / "features.json"
REGISTRY_PATH = BASE_DIR / "models" / "model_registry.json"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"
EXPERIMENTS_PATH = BASE_DIR / "reports" / "model_experiments.json"
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not FEATURES_PATH.exists():
        raise FileNotFoundError("Metadados do modelo nao encontrados. Execute: python -m src.train_model")
    metadata = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    return {
        "default_approach_id": metadata.get("default_approach_id", "default"),
        "target_column": metadata["target_column"],
        "class_labels": metadata["class_labels"],
        "all_features": metadata.get("all_features", metadata["features"]),
        "approaches": [
            {
                "id": metadata.get("default_approach_id", "default"),
                "name": "Modelo Naive Bayes",
                "description": "Modelo salvo antes do pipeline experimental.",
                "model_path": MODEL_PATH.name,
                "features": metadata["features"],
                "features_count": len(metadata["features"]),
                "metrics": {},
            }
        ],
    }


def load_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    registry = load_registry()
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo nao encontrado. Execute: python -m src.train_model")
    models = {}
    for approach in registry["approaches"]:
        path = BASE_DIR / "models" / approach["model_path"]
        if not path.exists():
            raise FileNotFoundError(f"Artefato de modelo nao encontrado: {path}")
        models[approach["id"]] = joblib.load(path)
    return models, registry


def get_approach(registry: dict[str, Any], approach_id: str | None) -> dict[str, Any]:
    selected_id = approach_id or registry["default_approach_id"]
    for approach in registry["approaches"]:
        if approach["id"] == selected_id:
            return approach
    raise KeyError(selected_id)


def probability_map(model: Any, probabilities: list[float]) -> dict[str, float]:
    return {label: float(prob) for label, prob in zip(model.classes_, probabilities)}


def display_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    return {CLASS_LABELS[label]: probability for label, probability in probabilities.items()}


def confidence_level(confidence: float) -> str:
    if confidence >= 0.9:
        return "alta"
    if confidence >= 0.7:
        return "moderada"
    return "baixa"


def parse_threshold(payload: dict[str, Any] | None = None) -> float:
    raw_threshold = request.args.get("threshold")
    if payload and "threshold" in payload:
        raw_threshold = payload["threshold"]
    if raw_threshold is None:
        return 0.5
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold must be numeric.") from exc
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return threshold


def parse_approach_id(payload: dict[str, Any] | None = None) -> str | None:
    if payload and "approach_id" in payload:
        return str(payload["approach_id"])
    value = request.args.get("approach_id")
    return str(value) if value else None


def classify_with_threshold(probabilities: dict[str, float], threshold: float) -> str:
    return "M" if probabilities.get("M", 0.0) >= threshold else "B"


def feature_statistics(features_df: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    df = features_df.copy()
    df[TARGET_COLUMN] = target
    stats: dict[str, Any] = {}
    for feature in features_df.columns:
        stats[feature] = {
            "overall": {
                "mean": float(features_df[feature].mean()),
                "std": float(features_df[feature].std()),
                "min": float(features_df[feature].min()),
                "max": float(features_df[feature].max()),
            },
            "classes": {
                label: {
                    "label": CLASS_LABELS[label],
                    "mean": float(df.loc[df[TARGET_COLUMN] == label, feature].mean()),
                    "std": float(df.loc[df[TARGET_COLUMN] == label, feature].std()),
                }
                for label in ["B", "M"]
            },
        }
    return stats


def explain_features(row: dict[str, float], stats: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    explanations = []
    for feature, value in row.items():
        feature_stats = stats[feature]
        overall = feature_stats["overall"]
        std = overall["std"] or 1.0
        z_score = (value - overall["mean"]) / std
        distance_b = abs(value - feature_stats["classes"]["B"]["mean"])
        distance_m = abs(value - feature_stats["classes"]["M"]["mean"])
        closer_to = "M" if distance_m < distance_b else "B"
        explanations.append(
            {
                "feature": feature,
                "value": value,
                "z_score": float(z_score),
                "abs_z_score": float(abs(z_score)),
                "closer_to": closer_to,
                "closer_to_label": CLASS_LABELS[closer_to],
                "distance_to_benign_mean": float(distance_b),
                "distance_to_malignant_mean": float(distance_m),
                "benign_mean": float(feature_stats["classes"]["B"]["mean"]),
                "malignant_mean": float(feature_stats["classes"]["M"]["mean"]),
                "overall_mean": float(overall["mean"]),
            }
        )
    explanations.sort(key=lambda item: item["abs_z_score"], reverse=True)
    return explanations[:limit]


def build_prediction_response(
    model: Any,
    class_labels: dict[str, str],
    row: dict[str, float],
    features: list[str],
    threshold: float,
    stats: dict[str, Any],
    approach: dict[str, Any],
) -> dict[str, Any]:
    data = pd.DataFrame([row], columns=features)
    probabilities = probability_map(model, model.predict_proba(data)[0])
    prediction = classify_with_threshold(probabilities, threshold)
    confidence = probabilities[prediction]
    top_features = explain_features(row, stats)
    top_names = ", ".join(item["feature"] for item in top_features[:3])
    uncertainty_note = (
        "O modelo esta pouco confiante; trate este caso como incerto."
        if confidence < 0.7
        else "A probabilidade da classe prevista indica uma decisao mais estavel."
    )
    narrative = (
        f"O modelo classificou a amostra como {class_labels[prediction]} "
        f"com {confidence * 100:.1f}% de confianca. "
        f"As variaveis que mais se destacaram foram: {top_names}. {uncertainty_note}"
    )

    return {
        "class": prediction,
        "prediction": class_labels[prediction],
        "threshold": threshold,
        "approach_id": approach["id"],
        "approach_name": approach["name"],
        "features_used": features,
        "features_count": len(features),
        "confidence": float(confidence),
        "confidence_level": confidence_level(confidence),
        "probabilities": display_probabilities(probabilities),
        "raw_probabilities": probabilities,
        "top_features": top_features,
        "explanation": narrative,
    }


def validate_prediction_values(values: Any, features: list[str]) -> tuple[dict[str, float] | None, tuple[dict[str, Any], int] | None]:
    if not isinstance(values, dict):
        return None, ({"error": "Field 'features' must be an object."}, 400)

    missing = [feature for feature in features if feature not in values]
    if missing:
        return None, ({"error": "Missing required features.", "missing": missing}, 400)

    row = {}
    invalid = {}
    for feature in features:
        try:
            row[feature] = float(values[feature])
        except (TypeError, ValueError):
            invalid[feature] = values[feature]

    if invalid:
        return None, ({"error": "All features must be numeric.", "invalid": invalid}, 400)

    return row, None


def select_examples(model: Any, approach: dict[str, Any], features_df: pd.DataFrame, target: pd.Series) -> list[dict[str, Any]]:
    approach_features = approach["features"]
    probabilities = pd.DataFrame(model.predict_proba(features_df[approach_features]), columns=model.classes_)
    predictions = ["M" if value >= 0.5 else "B" for value in probabilities["M"]]
    working = features_df.copy()
    working[TARGET_COLUMN] = target.values
    working["predicted"] = predictions
    working["probability_malignant"] = probabilities["M"].values
    working["confidence"] = probabilities.max(axis=1).values
    working["uncertainty"] = (probabilities["M"] - 0.5).abs().values

    candidates = [
        ("benign_typical", "Benigno tipico", working[working[TARGET_COLUMN] == "B"].sort_values("confidence", ascending=False).head(1)),
        ("malignant_typical", "Maligno tipico", working[working[TARGET_COLUMN] == "M"].sort_values("confidence", ascending=False).head(1)),
        ("borderline", "Caso incerto", working.sort_values("uncertainty").head(1)),
        ("false_positive", "Falso positivo do teste visual", working[(working[TARGET_COLUMN] == "B") & (working["predicted"] == "M")].sort_values("confidence", ascending=False).head(1)),
        ("false_negative", "Falso negativo do teste visual", working[(working[TARGET_COLUMN] == "M") & (working["predicted"] == "B")].sort_values("confidence", ascending=False).head(1)),
        ("random", "Amostra aleatoria", working.sample(1, random_state=42)),
    ]

    examples = []
    for example_id, name, frame in candidates:
        if frame.empty:
            continue
        row = frame.iloc[0]
        feature_values = {feature: float(row[feature]) for feature in features_df.columns}
        examples.append(
            {
                "id": example_id,
                "name": name,
                "approach_id": approach["id"],
                "actual_class": row[TARGET_COLUMN],
                "actual_label": CLASS_LABELS[row[TARGET_COLUMN]],
                "model_prediction": row["predicted"],
                "probability_malignant": float(row["probability_malignant"]),
                "confidence": float(row["confidence"]),
                "features": feature_values,
            }
        )
    return examples


def create_app() -> Flask:
    app = Flask(__name__)
    models, registry = load_artifacts()
    class_labels = registry["class_labels"]
    all_features = registry["all_features"]
    bundle = load_dataset(cache=True)
    validate_dataset(bundle.features, bundle.target)
    stats = feature_statistics(bundle.features, bundle.target)

    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            return ("", 204)
        return None

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin in ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model_loaded": True})

    @app.get("/model-info")
    def model_info():
        return jsonify(
            {
                "model": "GaussianNB",
                "target": registry["target_column"],
                "features": get_approach(registry, None)["features"],
                "all_features": all_features,
                "classes": class_labels,
                "default_approach_id": registry["default_approach_id"],
                "approaches": registry["approaches"],
                "input_format": {"features": {feature: "number" for feature in all_features}},
            }
        )

    @app.get("/feature-stats")
    def feature_stats():
        return jsonify({"features": stats})

    @app.get("/examples")
    def examples():
        try:
            approach = get_approach(registry, parse_approach_id())
        except KeyError as exc:
            return jsonify({"error": f"Unknown approach_id: {exc.args[0]}"}), 400
        return jsonify({"examples": select_examples(models[approach["id"]], approach, bundle.features, bundle.target)})

    @app.get("/model-summary")
    def model_summary():
        metrics = {}
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        experiments = []
        if EXPERIMENTS_PATH.exists():
            experiments = json.loads(EXPERIMENTS_PATH.read_text(encoding="utf-8"))
        class_counts = bundle.target.value_counts().reindex(["B", "M"])
        majority_baseline = float(class_counts.max() / class_counts.sum())
        return jsonify(
            {
                "dataset": {
                    "name": "Breast Cancer Wisconsin Diagnostic Dataset",
                    "observations": int(len(bundle.features)),
                    "features": int(len(bundle.features.columns)),
                    "class_distribution": {
                        label: {
                            "label": CLASS_LABELS[label],
                            "count": int(class_counts[label]),
                            "percentage": float(class_counts[label] / class_counts.sum()),
                        }
                        for label in ["B", "M"]
                    },
                },
                "metrics": metrics,
                "experiments": experiments,
                "approaches": registry["approaches"],
                "default_approach_id": registry["default_approach_id"],
                "majority_baseline_accuracy": majority_baseline,
            }
        )

    @app.post("/predict")
    def predict():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "JSON body is required."}), 400

        try:
            approach = get_approach(registry, parse_approach_id(payload))
        except KeyError as exc:
            return jsonify({"error": f"Unknown approach_id: {exc.args[0]}"}), 400

        values = payload.get("features", payload)
        row, error = validate_prediction_values(values, approach["features"])
        if error:
            body, status = error
            return jsonify(body), status

        try:
            threshold = parse_threshold(payload)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        return jsonify(
            build_prediction_response(
                models[approach["id"]],
                class_labels,
                row,
                approach["features"],
                threshold,
                stats,
                approach,
            )
        )

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    create_app().run(host="127.0.0.1", port=port, debug=False, use_reloader=False)
