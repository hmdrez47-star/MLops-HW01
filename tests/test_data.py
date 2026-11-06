"""Tests for data loading, splitting, hashing and data versioning."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import (
    FEATURE_COLUMNS,
    compute_hash,
    create_data_version,
    get_data_version,
    list_data_versions,
    load_config,
    load_csv,
    split_data,
    validate_schema,
    verify_data_version,
)


class TestLoadData:
    """Loading behaviour on existing, missing and empty files."""

    def test_load_csv_existing_file(self, real_dataset_path: Path):
        df = load_csv(real_dataset_path)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (400, 13)
        assert "churn" in df.columns
        assert all(column in df.columns for column in FEATURE_COLUMNS)

    def test_load_csv_from_tmp_file(self, tmp_path: Path):
        path = tmp_path / "tiny.csv"
        pd.DataFrame({"a": [1, 2, 3], "churn": [0, 1, 0]}).to_csv(path, index=False)

        df = load_csv(path)

        assert len(df) == 3
        assert list(df.columns) == ["a", "churn"]

    def test_load_csv_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_csv(tmp_path / "does_not_exist.csv")

    def test_load_csv_empty_file_raises(self, tmp_path: Path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            load_csv(path)

    def test_relative_path_is_resolved_from_project_root(self):
        assert load_csv("data/raw/dataset_v1.csv").shape == (400, 13)

    def test_validate_schema_rejects_missing_columns(self, sample_df):
        broken = sample_df.drop(columns=["credit_score"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(broken)


class TestHash:
    """Hash computation must be deterministic and data sensitive."""

    def test_compute_hash_is_deterministic(self, sample_df):
        first = compute_hash(sample_df)
        second = compute_hash(sample_df.copy())

        assert first == second
        assert len(first) == 64
        assert all(character in "0123456789abcdef" for character in first)

    def test_compute_hash_changes_with_data(self, sample_df):
        modified = sample_df.copy()
        modified.loc[0, "credit_score"] = modified.loc[0, "credit_score"] + 1

        assert compute_hash(modified) != compute_hash(sample_df)

    def test_hash_matches_recorded_version(self, real_dataset_path: Path):
        df = load_csv(real_dataset_path)
        versions = list_data_versions("data_versions")
        if not any(version["version_id"] == "v1" for version in versions):
            pytest.skip("data version metadata has not been generated yet")

        assert verify_data_version(df, "v1", "data_versions") is True


class TestSplitData:
    """Shape and determinism of the train/test split."""

    def test_split_returns_four_elements(self, sample_df):
        result = split_data(sample_df, "churn", test_size=0.2, random_state=42)
        assert len(result) == 4

    def test_split_sizes(self, real_dataset_path: Path):
        df = load_csv(real_dataset_path)
        X_train, X_test, y_train, y_test = split_data(
            df, "churn", test_size=0.2, random_state=42
        )

        assert len(X_train) == 320
        assert len(X_test) == 80
        assert len(y_train) == 320
        assert len(y_test) == 80
        assert X_train.shape[1] == len(FEATURE_COLUMNS)

    def test_split_reproducible(self, sample_df):
        first = split_data(sample_df, "churn", test_size=0.2, random_state=42)
        second = split_data(sample_df, "churn", test_size=0.2, random_state=42)

        assert first[0].equals(second[0])
        assert first[2].equals(second[2])

    def test_split_is_stratified_when_requested(self, sample_df):
        _, _, y_train, y_test = split_data(
            sample_df, "churn", test_size=0.2, random_state=42
        )
        assert abs(y_train.mean() - y_test.mean()) < 0.05

    def test_invalid_test_size_raises(self, sample_df):
        with pytest.raises(ValueError):
            split_data(sample_df, "churn", test_size=1.5)

    def test_missing_target_raises(self, sample_df):
        with pytest.raises(ValueError):
            split_data(sample_df, target_column="not_a_column")


class TestDataVersioning:
    """data_versions/metadata.json contract."""

    def test_create_data_version_writes_metadata(self, sample_df, tmp_path: Path):
        entry = create_data_version(
            sample_df, "v1", "reference dataset", output_dir=tmp_path / "versions"
        )

        metadata_path = tmp_path / "versions" / "metadata.json"
        assert metadata_path.exists()

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert isinstance(payload, list) and len(payload) == 1

        for key in ("version_id", "hash", "description", "n_rows", "created_at"):
            assert key in payload[0]

        assert payload[0]["version_id"] == "v1"
        assert payload[0]["n_rows"] == len(sample_df)
        assert payload[0]["hash"] == compute_hash(sample_df)
        datetime.fromisoformat(payload[0]["created_at"])  # must be valid ISO-8601
        assert entry == payload[0]
        assert (tmp_path / "versions" / "v1.csv").exists()

    def test_two_versions_are_kept(self, sample_df, tmp_path: Path):
        directory = tmp_path / "versions"
        create_data_version(sample_df, "v1", "first", output_dir=directory)
        changed = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        create_data_version(changed, "v2", "second, more rows", output_dir=directory)

        versions = list_data_versions(directory)
        assert [version["version_id"] for version in versions] == ["v1", "v2"]
        assert get_data_version("v2", directory)["n_rows"] == len(sample_df) + 5

    def test_unknown_version_raises(self, tmp_path: Path):
        with pytest.raises(KeyError):
            get_data_version("nope", tmp_path)


class TestConfig:
    def test_load_config_returns_mapping(self):
        config = load_config("config/config.yaml")
        assert config["data"]["target_column"] == "churn"
        assert config["features"]["feature_selection_k"] == 10
        assert config["model"]["type"] in {
            "random_forest",
            "logistic_regression",
            "svm",
        }

    def test_missing_config_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "missing.yaml")
