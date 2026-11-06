"""Tests for prediction logging and drift detection."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from src.monitoring.drift_detection import (
    detect_drift_ks,
    detect_drift_mean_shift,
    detect_feature_drift,
    generate_drift_report,
    log_prediction,
    read_prediction_logs,
)


class TestDriftDetection:
    def test_no_drift_same_distribution(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(50, 5, 500)

        result = detect_drift_ks(reference, reference.copy(), threshold=0.05)

        assert result["drift_detected"] is False
        assert result["p_value"] == pytest.approx(1.0)

    def test_drift_different_distributions(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(50, 5, 500)
        shifted = rng.normal(70, 5, 500)

        result = detect_drift_ks(reference, shifted, threshold=0.05)

        assert result["drift_detected"] is True
        assert result["statistic"] > 0.5
        assert result["p_value"] < 0.05

    def test_ks_result_structure(self):
        rng = np.random.default_rng(2)
        result = detect_drift_ks(rng.normal(0, 1, 100), rng.normal(0, 1, 100))

        for key in ("method", "statistic", "p_value", "drift_detected"):
            assert key in result
        assert result["method"] == "ks_test"

    def test_mean_shift_detected(self):
        rng = np.random.default_rng(3)
        result = detect_drift_mean_shift(
            rng.normal(10, 1, 200), rng.normal(13, 1, 200), threshold=0.1
        )

        assert result["drift_detected"] is True
        assert result["relative_shift"] > 0.1
        assert result["method"] == "mean_shift"

    def test_mean_shift_not_detected(self):
        rng = np.random.default_rng(4)
        result = detect_drift_mean_shift(
            rng.normal(10, 1, 200), rng.normal(10.01, 1, 200), threshold=0.1
        )

        assert result["drift_detected"] is False

    def test_too_few_samples_raise(self):
        with pytest.raises(ValueError):
            detect_drift_ks([1.0], [2.0, 3.0])

    def test_real_versions_drift(
        self, real_dataset_path: Path, drift_dataset_path: Path
    ):
        import pandas as pd

        v1 = pd.read_csv(real_dataset_path)
        v2 = pd.read_csv(drift_dataset_path)
        features = [column for column in v1.columns if column != "churn"]

        detail = detect_feature_drift(
            v1, v2, ks_threshold=0.05, mean_shift_threshold=0.1, features=features
        )

        assert detail["n_features"] == 12
        assert "monthly_charges" in detail["drifted_features"]
        assert "support_tickets" in detail["drifted_features"]
        assert "credit_score" in detail["drifted_features"]
        assert "age" not in detail["drifted_features"]
        assert detail["drift_score"] > 0.3


class TestPredictionLogging:
    def test_log_prediction_writes_structured_entry(self, tmp_path: Path):
        entry = log_prediction(
            prediction=1,
            confidence=0.87,
            latency=0.023,
            log_path=tmp_path,
            model_version="rf-v1",
        )

        day = datetime.now(UTC).strftime("%Y-%m-%d")
        log_file = tmp_path / f"predictions_{day}.json"
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        payload = json.loads(lines[0])
        assert set(payload) == {
            "timestamp",
            "latency",
            "prediction",
            "confidence",
            "model_version",
        }
        assert payload["prediction"] == 1
        assert payload["confidence"] == 0.87
        assert payload["latency"] == 0.023
        datetime.fromisoformat(payload["timestamp"])
        assert entry == payload

    def test_log_prediction_drops_sensitive_fields(self, tmp_path: Path):
        log_prediction(
            prediction=0,
            confidence=0.5,
            latency=0.01,
            log_path=tmp_path,
            user_id="cust-42",
            email="someone@example.com",
            features=[1.0, 2.0, 3.0],
            raw_payload={"credit_score": 700},
            model_version="rf-v1",
        )

        logged = read_prediction_logs(tmp_path)
        serialised = json.dumps(logged)

        assert "user_id" not in logged[0]
        assert "email" not in logged[0]
        assert "features" not in logged[0]
        assert "cust-42" not in serialised
        assert "someone@example.com" not in serialised
        assert "700" not in serialised

    def test_appends_multiple_predictions(self, tmp_path: Path):
        for index in range(3):
            log_prediction(index % 2, 0.6, 0.01, log_path=tmp_path)

        assert len(read_prediction_logs(tmp_path)) == 3


class TestDriftReport:
    def test_report_contains_required_sections(self, tmp_path: Path):
        import pandas as pd

        v1 = pd.read_csv("data/raw/dataset_v1.csv")
        v2 = pd.read_csv("data/raw/dataset_v2.csv")
        features = [column for column in v1.columns if column != "churn"]

        report = generate_drift_report(
            v1,
            v2,
            reference_version="v1",
            current_version="v2",
            threshold=0.3,
            features=features,
            output_dir=tmp_path,
        )

        for key in (
            "drift_detected",
            "drift_score",
            "severity",
            "affected_features",
            "recommended_action",
            "summary",
            "generated_at",
        ):
            assert key in report

        assert report["drift_detected"] is True
        assert report["severity"] in {"low", "medium", "high"}
        assert report["affected_features"]
        assert report["recommended_action"]
        assert (tmp_path / "drift_report.json").exists()
        assert (tmp_path / "drift_report.md").exists()

    def test_report_without_drift(self, tmp_path: Path):
        import pandas as pd

        v1 = pd.read_csv("data/raw/dataset_v1.csv")
        features = [column for column in v1.columns if column != "churn"]

        report = generate_drift_report(
            v1, v1.copy(), features=features, output_dir=tmp_path, threshold=0.3
        )

        assert report["drift_detected"] is False
        assert report["drift_score"] == 0.0
        assert report["severity"] == "none"
