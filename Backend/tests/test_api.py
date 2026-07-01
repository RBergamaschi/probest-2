from __future__ import annotations

import pytest

from app import create_app
from src.data import load_dataset
from src.train_model import MODEL_PATH, train_model


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_model()


@pytest.fixture()
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_model_info(client):
    response = client.get("/model-info")
    payload = response.get_json()
    assert response.status_code == 200
    assert len(payload["all_features"]) == 30
    assert len(payload["features"]) >= 1
    assert payload["default_approach_id"]
    assert len(payload["approaches"]) >= 4
    assert payload["classes"]["B"] == "Tumor Benigno"
    assert payload["classes"]["M"] == "Tumor Maligno"


def test_predict_valid_payload(client):
    bundle = load_dataset(cache=True)
    sample = bundle.features.iloc[0].to_dict()
    response = client.post("/predict", json={"features": sample})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["class"] in {"B", "M"}
    assert payload["prediction"] in {"Tumor Benigno", "Tumor Maligno"}
    assert payload["approach_id"]
    assert payload["features_count"] >= 1
    assert "probabilities" in payload
    assert "confidence" in payload
    assert "confidence_level" in payload
    assert "top_features" in payload
    assert "explanation" in payload


def test_predict_missing_feature(client):
    response = client.post("/predict", json={"features": {}})
    payload = response.get_json()
    assert response.status_code == 400
    assert payload["error"] == "Missing required features."


def test_predict_threshold(client):
    bundle = load_dataset(cache=True)
    sample = bundle.features.iloc[0].to_dict()
    response = client.post("/predict", json={"features": sample, "threshold": 0.8})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["threshold"] == 0.8


def test_predict_specific_approach(client):
    bundle = load_dataset(cache=True)
    sample = bundle.features.iloc[0].to_dict()
    response = client.post(
        "/predict",
        json={"features": sample, "threshold": 0.5, "approach_id": "baseline_30_features"},
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["approach_id"] == "baseline_30_features"
    assert payload["features_count"] == 30


def test_examples(client):
    response = client.get("/examples")
    payload = response.get_json()
    assert response.status_code == 200
    assert len(payload["examples"]) >= 3
    assert len(payload["examples"][0]["features"]) == 30
    assert payload["examples"][0]["actual_class"] in {"B", "M"}


def test_feature_stats(client):
    response = client.get("/feature-stats")
    payload = response.get_json()
    assert response.status_code == 200
    assert "radius1" in payload["features"]
    assert "overall" in payload["features"]["radius1"]
    assert "B" in payload["features"]["radius1"]["classes"]
    assert "M" in payload["features"]["radius1"]["classes"]


def test_model_summary(client):
    response = client.get("/model-summary")
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["dataset"]["observations"] == 569
    assert payload["dataset"]["features"] == 30
    assert payload["majority_baseline_accuracy"] > 0
    assert "metrics" in payload
    assert len(payload["experiments"]) >= 4
    assert payload["default_approach_id"]
