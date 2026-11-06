"""Tests for model creation, training, evaluation and artifact metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.data.load_data import split_data
from src.features.build_features import build_feature_pipeline
from src.models.train_model import (
    MODEL_TYPES,
    evaluate_model,
    get_model,
    load_model,
    load_preprocessor,
    save_model,
    train_model,
)


@pytest.fixture
def prepared(sample_df):
    """Split + build features once and hand back everything a test needs."""
    X_train, X_test, y_train, y_test = split_data(
        sample_df, "churn", test_size=0.2, random_state=42
    )
    bundle = build_feature_pipeline(X_train, y_train, X_test, scaling="standard", k=10)
    bundle["y_train"] = y_train
    bundle["y_test"] = y_test
    return bundle


class TestGetModel:
    def test_random_forest(self):
        model = get_model("random_forest", {"n_estimators": 10, "random_state": 0})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 10

    def test_logistic_regression(self):
        model = get_model("logistic_regression", {"C": 0.5})
        assert isinstance(model, LogisticRegression)
        assert model.C == 0.5

    def test_svm_has_probability_enabled(self):
        model = get_model("svm", {"C": 1.0})
        assert isinstance(model, SVC)
        assert model.probability is True

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            get_model("invalid", {})

    def test_all_declared_types_are_constructible(self):
        for model_type in MODEL_TYPES:
            assert get_model(model_type, {}) is not None


class TestTrainModel:
    def test_train_model_fits_and_predicts(self, prepared):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "random_forest",
            {"n_estimators": 20, "random_state": 0},
        )

        assert hasattr(model, "predict")
        predictions = model.predict(prepared["X_test"])
        assert len(predictions) == len(prepared["X_test"])

    def test_train_logistic_regression(self, prepared):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "logistic_regression",
            {"max_iter": 500},
        )
        assert hasattr(model, "predict_proba")

    def test_train_svm(self, prepared):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "svm",
            {"C": 1.0, "probability": True},
        )
        assert hasattr(model, "predict_proba")


class TestEvaluateModel:
    def test_evaluate_returns_metrics(self, prepared):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "random_forest",
            {"n_estimators": 30, "random_state": 0},
        )

        metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"])

        assert set(metrics) >= {"accuracy", "f1_score"}
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0
        # the synthetic signal is learnable, so the model must beat chance
        assert metrics["accuracy"] > 0.6


class TestSaveModel:
    def test_save_creates_model_and_metadata(self, prepared, tmp_path: Path):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "random_forest",
            {"n_estimators": 30, "random_state": 0},
        )
        metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"])

        model_path = save_model(
            model,
            metrics,
            data_version="v1",
            model_version="rf-v1-test",
            output_dir=tmp_path,
        )

        assert Path(model_path).exists()
        metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
        for key in ("model_version", "created_at", "data_version", "metric"):
            assert key in metadata
        assert metadata["model_version"] == "rf-v1-test"
        assert metadata["data_version"] == "v1"
        assert metadata["metric"]["name"] == "f1_score"

    def test_save_with_preprocessor(self, prepared, tmp_path: Path):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "svm",
            {"C": 1.0, "probability": True},
        )
        metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"])

        save_model(
            model,
            metrics,
            data_version="v1",
            model_version="svm-v1-test",
            output_dir=tmp_path,
            preprocessor={
                "scaler": prepared["scaler"],
                "selector": prepared["selector"],
                "input_features": prepared["input_features"],
                "selected_features": prepared["selected_features"],
                "scaling_method": prepared["scaling_method"],
            },
        )

        assert (tmp_path / "preprocessor.joblib").exists()
        preprocessor = load_preprocessor(tmp_path)
        assert preprocessor is not None
        assert len(preprocessor["input_features"]) == 12

    def test_load_model_roundtrip(self, prepared, tmp_path: Path):
        model = train_model(
            prepared["X_train"],
            prepared["y_train"],
            "random_forest",
            {"n_estimators": 20, "random_state": 0},
        )
        metrics = evaluate_model(model, prepared["X_test"], prepared["y_test"])
        save_model(model, metrics, "v1", "rf-roundtrip", output_dir=tmp_path)

        loaded, metadata = load_model(tmp_path)

        assert metadata["model_version"] == "rf-roundtrip"
        assert hasattr(loaded, "predict")

    def test_load_missing_model_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path)
