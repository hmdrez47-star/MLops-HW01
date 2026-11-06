"""Data loading, schema validation, splitting and data versioning.

Every function here is pure (it takes what it needs as arguments) so the whole
pipeline can be driven from ``config/config.yaml`` and unit-tested in isolation.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

#: Columns that must exist in every dataset version.
FEATURE_COLUMNS: list[str] = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_services",
    "support_tickets",
    "avg_session_minutes",
    "login_frequency",
    "contract_length",
    "payment_delay_days",
    "credit_score",
    "age",
    "num_dependents",
]
TARGET_COLUMN = "churn"


def project_root() -> Path:
    """Absolute path of the repository root (``.../ANSWER``)."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | os.PathLike) -> Path:
    """Resolve ``path_like`` against the project root unless it is absolute."""
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root() / path


def utc_now() -> str:
    """Timezone aware ISO-8601 timestamp (seconds precision)."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def load_config(
    config_path: str | os.PathLike = "config/config.yaml",
) -> dict[str, Any]:
    """Load the YAML configuration file.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    TypeError
        If the file does not contain a YAML mapping.
    """
    path = resolve_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Config file must contain a YAML mapping: {path}")
    return config


def load_csv(file_path: str | os.PathLike) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist - the pipeline must fail loudly instead
        of silently training on an empty frame.
    ValueError
        If the file is empty.
    """
    path = resolve_path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Data file is empty: {path}")
    return df


def validate_schema(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    required_columns: list[str] | None = None,
) -> None:
    """Validate that all expected feature/target columns are present."""
    required = list(required_columns or FEATURE_COLUMNS) + [target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def compute_hash(df: pd.DataFrame) -> str:
    """Deterministic SHA-256 fingerprint of a DataFrame.

    The frame is serialised to a canonical CSV (no index) before hashing, so the
    same data always produces the same hash regardless of dtypes or platform.
    """
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
):
    """Split ``df`` into ``(X_train, X_test, y_train, y_test)``.

    ``stratify`` keeps the (slightly imbalanced) churn ratio identical in both
    splits; it degrades gracefully if a class has too few samples.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in DataFrame")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float strictly between 0 and 1")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    kwargs: dict[str, Any] = {
        "test_size": test_size,
        "random_state": random_state,
    }
    if stratify:
        counts = y.value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            kwargs["stratify"] = y

    return train_test_split(X, y, **kwargs)


def _read_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    if not metadata_path.exists():
        return []
    with metadata_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):  # tolerate a single-entry file
        return [data]
    if not isinstance(data, list):
        raise TypeError(f"Invalid data version metadata: {metadata_path}")
    return data


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def create_data_version(
    df: pd.DataFrame,
    version_id: str,
    description: str,
    output_dir: str | os.PathLike = "data_versions",
    snapshot: bool = True,
) -> dict[str, Any]:
    """Register ``df`` as data version ``version_id``.

    Writes (and updates) ``<output_dir>/metadata.json`` with one entry per
    version: ``version_id``, ``hash``, ``description``, ``n_rows``,
    ``created_at``. An untouched CSV snapshot is stored next to the metadata so
    the hash can always be re-verified later.
    """
    if not version_id:
        raise ValueError("version_id must be a non-empty string")

    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"

    entry: dict[str, Any] = {
        "version_id": version_id,
        "hash": compute_hash(df),
        "description": description,
        "n_rows": len(df),
        "n_columns": int(df.shape[1]),
        "created_at": utc_now(),
    }

    versions = [
        v for v in _read_metadata(metadata_path) if v.get("version_id") != version_id
    ]
    versions.append(entry)
    versions.sort(key=lambda item: str(item.get("version_id")))
    _write_json_atomic(metadata_path, versions)

    if snapshot:
        df.to_csv(out_dir / f"{version_id}.csv", index=False)

    return entry


def list_data_versions(
    output_dir: str | os.PathLike = "data_versions",
) -> list[dict[str, Any]]:
    """Return every registered data version (empty list if none)."""
    return _read_metadata(resolve_path(output_dir) / "metadata.json")


def get_data_version(
    version_id: str, output_dir: str | os.PathLike = "data_versions"
) -> dict[str, Any]:
    """Return the metadata entry of ``version_id`` or raise ``KeyError``."""
    for entry in list_data_versions(output_dir):
        if entry.get("version_id") == version_id:
            return entry
    raise KeyError(f"Unknown data version: {version_id}")


def verify_data_version(
    df: pd.DataFrame,
    version_id: str,
    output_dir: str | os.PathLike = "data_versions",
) -> bool:
    """Check that ``df`` still hashes to the recorded fingerprint."""
    stored = get_data_version(version_id, output_dir)
    return stored.get("hash") == compute_hash(df)
