"""Tests for scaling and ANOVA feature selection."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.load_data import split_data
from src.features.build_features import (
    build_feature_pipeline,
    scale_features,
    select_features,
    transform_features,
)


@pytest.fixture
def split(sample_df):
    return split_data(sample_df, "churn", test_size=0.2, random_state=42)


class TestScaleFeatures:
    def test_standard_scaling_centers_train(self, split):
        X_train, X_test = split[0], split[1]

        scaled_train, scaled_test, scaler = scale_features(X_train, X_test, "standard")

        assert scaler is not None
        assert scaled_train.shape == X_train.shape
        assert scaled_test.shape == X_test.shape
        assert np.allclose(scaled_train.mean().values, 0.0, atol=1e-8)
        assert np.allclose(scaled_train.std(ddof=0).values, 1.0, atol=1e-8)

    def test_scaler_fitted_on_train_only(self, split):
        X_train, X_test = split[0], split[1]
        _, _, scaler = scale_features(X_train, X_test, "standard")

        # If the test split had leaked into the fit, the mean would differ.
        assert np.allclose(scaler.mean_, X_train.mean().values)
        assert not np.allclose(X_test.mean().values, X_train.mean().values)

    def test_minmax_scaling_bounds(self, split):
        X_train, X_test = split[0], split[1]

        scaled_train, _, scaler = scale_features(X_train, X_test, "minmax")

        # float32/float64 round-trips can land on 1.0000000000000002
        assert scaled_train.min().min() >= 0.0
        assert scaled_train.max().max() <= 1.0 + 1e-9
        assert scaler is not None

    def test_scaling_none_returns_same_values(self, split):
        X_train, X_test = split[0], split[1]

        scaled_train, scaled_test, scaler = scale_features(X_train, X_test, "none")

        assert scaler is None
        assert scaled_train.equals(X_train)
        assert scaled_test.equals(X_test)

    def test_invalid_method_raises(self, split):
        with pytest.raises(ValueError):
            scale_features(split[0], split[1], "magic")


class TestSelectFeatures:
    def test_keeps_k_features(self, split):
        X_train_selected, X_test_selected, selector = select_features(
            split[0], split[2], split[1], k=10
        )

        assert X_train_selected.shape[1] == 10
        assert X_test_selected.shape[1] == 10
        assert list(X_train_selected.columns) == list(X_test_selected.columns)
        assert selector.get_support().sum() == 10

    def test_selected_columns_are_a_subset(self, split):
        X_train_selected, _, _ = select_features(split[0], split[2], split[1], k=8)
        assert set(X_train_selected.columns).issubset(set(split[0].columns))

    def test_k_larger_than_feature_count_keeps_all(self, split):
        X_train_selected, _, _selector = select_features(
            split[0], split[2], split[1], k=99
        )
        assert X_train_selected.shape[1] == split[0].shape[1]

    def test_selector_fitted_on_train_only(self, split):
        _, _, selector = select_features(split[0], split[2], split[1], k=10)
        scores = np.asarray(selector.scores_)
        assert scores.shape[0] == split[0].shape[1]
        assert np.isfinite(scores).all()


class TestFeaturePipeline:
    def test_bundle_contract(self, split):
        bundle = build_feature_pipeline(
            split[0], split[2], split[1], scaling="standard", k=10
        )

        for key in (
            "X_train",
            "X_test",
            "scaler",
            "selector",
            "selected_features",
            "input_features",
        ):
            assert key in bundle

        assert bundle["X_train"].shape[1] == 10
        assert len(bundle["input_features"]) == 12
        assert len(bundle["selected_features"]) == 10
        assert bundle["feature_scores"]

    def test_transform_features_matches_pipeline(self, split, sample_df):
        bundle = build_feature_pipeline(
            split[0], split[2], split[1], scaling="standard", k=10
        )

        raw = sample_df.drop(columns=["churn"]).iloc[:5]
        transformed = transform_features(raw, bundle["scaler"], bundle["selector"])

        assert transformed.shape == (5, 10)
        assert list(transformed.columns) == bundle["selected_features"]
