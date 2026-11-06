"""Tests for the FastAPI serving layer."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.data.load_data import split_data
from src.features.build_features import build_feature_pipeline
from src.models.train_model import evaluate_model, save_model, train_model
from tests.conftest import make_frame


def _build_artifacts(model_dir: Path) -> list[str]:
    """Train a small model and write model.joblib + metadata + preprocessor."""
    frame = make_frame(n_samples=300, seed=11)
    X_train, X_test, y_train, y_test = split_data(
        frame, "churn", test_size=0.2, random_state=42
    )
    bundle = build_feature_pipeline(X_train, y_train, X_test, scaling="standard", k=10)

    model = train_model(
        bundle["X_train"],
        y_train,
        "random_forest",
        {"n_estimators": 40, "random_state": 0},
    )
    metrics = evaluate_model(model, bundle["X_test"], y_test)

    save_model(
        model,
        metrics,
        data_version="v1",
        model_version="serving-test-1",
        output_dir=model_dir,
        preprocessor={
            "scaler": bundle["scaler"],
            "selector": bundle["selector"],
            "input_features": bundle["input_features"],
            "selected_features": bundle["selected_features"],
            "scaling_method": bundle["scaling_method"],
        },
    )
    return bundle["input_features"]


@pytest.fixture
def serving(tmp_path: Path, monkeypatch):
    model_dir = tmp_path / "models"
    log_dir = tmp_path / "logs"
    features = _build_artifacts(model_dir)

    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(log_dir))

    import src.serving.app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as client:
        yield client, features, log_dir


class TestHealth:
    def test_health_reports_model_version(self, serving):
        client, _, _ = serving

        response = client.get("/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"] == "serving-test-1"
        assert body["data_version"] == "v1"

    def test_metadata_endpoint(self, serving):
        client, _, _ = serving

        response = client.get("/metadata")

        assert response.status_code == 200
        assert response.json()["model_version"] == "serving-test-1"


class TestPredict:
    def test_predict_returns_contract(self, serving):
        client, features, _ = serving
        row = make_frame(n_samples=1, seed=5).drop(columns=["churn"]).iloc[0].tolist()
        assert len(row) == len(features)

        response = client.post("/predict", json={"features": [float(v) for v in row]})

        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["confidence"] <= 1.0
        assert body["latency"] >= 0.0

    def test_predict_logs_without_raw_features(self, serving):
        client, _features, log_dir = serving
        row = [
            float(v)
            for v in make_frame(n_samples=1, seed=6).drop(columns=["churn"]).iloc[0]
        ]

        client.post("/predict", json={"features": row})

        logs = list(log_dir.glob("predictions_*.json"))
        assert len(logs) == 1
        payload = logs[0].read_text(encoding="utf-8")
        assert '"prediction"' in payload
        assert '"confidence"' in payload
        assert '"latency"' in payload
        assert "monthly_charges" not in payload

    def test_wrong_feature_count_is_rejected(self, serving):
        client, features, _ = serving

        response = client.post("/predict", json={"features": [1.0, 2.0]})

        assert response.status_code == 400
        assert response.json()["detail"]["expected"] == len(features)

    def test_non_numeric_payload_is_rejected(self, serving):
        client, _, _ = serving

        response = client.post("/predict", json={"features": ["a", "b"]})

        assert response.status_code == 422
