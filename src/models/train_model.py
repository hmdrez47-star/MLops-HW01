"""Model creation, training, evaluation and artifact persistence."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

#: Supported model families.
MODEL_TYPES = ("random_forest", "logistic_regression", "svm")

#: The metric used to pick / compare runs (classes are not perfectly balanced).
PRIMARY_METRIC = "f1_score"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_like: str | os.PathLike) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else project_root() / path


def get_model(model_type: str, params: dict[str, Any] | None = None):
    """Instantiate an (untrained) estimator for ``model_type``.

    Raises
    ------
    ValueError
        If ``model_type`` is not one of :data:`MODEL_TYPES`.
    """
    hyperparameters = dict(params or {})

    if model_type == "random_forest":
        return RandomForestClassifier(**hyperparameters)

    if model_type == "logistic_regression":
        # A higher iteration budget keeps lbfgs convergent on scaled data.
        hyperparameters.setdefault("max_iter", 1000)
        return LogisticRegression(**hyperparameters)

    if model_type == "svm":
        # ``predict_proba`` needs probability=True; the API returns confidence.
        hyperparameters.setdefault("probability", True)
        return SVC(**hyperparameters)

    raise ValueError(
        f"Unsupported model_type {model_type!r}; expected one of {MODEL_TYPES}"
    )


def train_model(
    X_train, y_train, model_type: str, params: dict[str, Any] | None = None
):
    """Fit a model on the training split and return it."""
    model = get_model(model_type, params)
    model.fit(X_train, np.asarray(y_train).ravel())
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, float]:
    """Evaluate ``model``; returns accuracy, F1 and supporting metrics."""
    y_true = np.asarray(y_test).ravel()
    y_pred = np.asarray(model.predict(X_test)).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "n_samples": int(y_true.shape[0]),
    }


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return commit.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def save_model(
    model,
    metrics: dict[str, Any],
    data_version: str,
    model_version: str,
    output_dir: str | os.PathLike = "models",
    preprocessor: dict[str, Any] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> Path:
    """Persist ``model.joblib`` (plus preprocessor and ``metadata.json``).

    The metadata always contains ``model_version``, ``created_at``,
    ``data_version`` and ``metric`` so a served artifact is self-describing.
    """
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    if preprocessor is not None:
        joblib.dump(preprocessor, out_dir / "preprocessor.joblib")

    metadata: dict[str, Any] = {
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data_version": data_version,
        "metric": {
            "name": PRIMARY_METRIC,
            "value": metrics.get(PRIMARY_METRIC),
        },
        "metrics": dict(metrics),
        "model_type": type(model).__name__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "artifact": model_path.name,
    }
    if preprocessor is not None:
        metadata["preprocessor"] = "preprocessor.joblib"
        metadata["selected_features"] = preprocessor.get("selected_features", [])
        metadata["input_features"] = preprocessor.get("input_features", [])
        metadata["scaling_method"] = preprocessor.get("scaling_method")

    if metadata_extra:
        metadata.update(metadata_extra)

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return model_path


def load_model(output_dir: str | os.PathLike = "models"):
    """Load ``model.joblib`` and ``metadata.json``; returns ``(model, metadata)``."""
    out_dir = resolve_path(output_dir)
    model_path = out_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)
    metadata_path = out_dir / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    return model, metadata


def load_preprocessor(
    output_dir: str | os.PathLike = "models",
) -> dict[str, Any] | None:
    """Load the fitted scaler/selector bundle, or ``None`` if it was not saved."""
    path = resolve_path(output_dir) / "preprocessor.joblib"
    if not path.exists():
        return None
    return joblib.load(path)
