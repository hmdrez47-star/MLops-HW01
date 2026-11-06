"""Feature engineering: scaling and ANOVA F-value feature selection.

Both transformers are fitted on the training split **only** and then applied to
the test split, which keeps the evaluation honest (no leakage).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#: Supported scaling strategies.
SCALING_METHODS = ("standard", "minmax", "none")


def _as_frame(data) -> pd.DataFrame:
    """Return ``data`` as a DataFrame, naming bare arrays generically."""
    if isinstance(data, pd.DataFrame):
        return data
    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("Expected a 2-D array or DataFrame")
    return pd.DataFrame(array, columns=[f"feature_{i}" for i in range(array.shape[1])])


def _apply(transformer, frame: pd.DataFrame) -> pd.DataFrame:
    """Transform ``frame`` and keep the column names/index intact."""
    transformed = transformer.transform(frame)
    return pd.DataFrame(transformed, columns=frame.columns, index=frame.index)


def scale_features(
    X_train,
    X_test,
    method: str = "standard",
):
    """Scale features with the requested method.

    Parameters
    ----------
    method : str
        ``"standard"`` (zero mean / unit variance), ``"minmax"`` (0-1) or
        ``"none"`` (no scaling - suitable for tree models).

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
        ``scaler`` is ``None`` when ``method="none"``.
        The scaler is fitted on ``X_train`` only.
    """
    method = (method or "none").lower()
    if method not in SCALING_METHODS:
        raise ValueError(
            f"Unsupported scaling method {method!r}; expected one of {SCALING_METHODS}"
        )

    X_train_frame = _as_frame(X_train).copy()
    X_test_frame = _as_frame(X_test).copy()
    if list(X_train_frame.columns) != list(X_test_frame.columns):
        raise ValueError("Train and test frames must share the same columns")

    if method == "none":
        return X_train_frame, X_test_frame, None

    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    scaler.fit(X_train_frame)  # fit on train only

    return _apply(scaler, X_train_frame), _apply(scaler, X_test_frame), scaler


def select_features(
    X_train,
    y_train,
    X_test,
    k: int = 10,
):
    """Keep the ``k`` most informative features (ANOVA F-value).

    ``SelectKBest(f_classif)`` is fitted on the training split only; if ``k`` is
    greater than (or equal to) the number of columns, every column is kept.

    Returns
    -------
    (X_train_selected, X_test_selected, selector)
    """
    X_train_frame = _as_frame(X_train)
    X_test_frame = _as_frame(X_test)
    if list(X_train_frame.columns) != list(X_test_frame.columns):
        raise ValueError("Train and test frames must share the same columns")

    n_features = X_train_frame.shape[1]
    k_effective = n_features if k is None or k <= 0 else min(int(k), n_features)

    selector = SelectKBest(score_func=f_classif, k=k_effective)
    selector.fit(X_train_frame, np.asarray(y_train).ravel())

    mask = selector.get_support()
    columns = list(X_train_frame.columns[mask])

    return (
        X_train_frame.loc[:, columns].copy(),
        X_test_frame.loc[:, columns].copy(),
        selector,
    )


def get_feature_scores(selector: SelectKBest, columns: list[str]) -> dict[str, float]:
    """Return ``{feature: ANOVA F-value}`` sorted from most to least useful."""
    scores = getattr(selector, "scores_", None)
    if scores is None:
        return {}
    table = {name: float(score) for name, score in zip(columns, scores)}
    return dict(sorted(table.items(), key=lambda item: item[1], reverse=True))


def build_feature_pipeline(
    X_train,
    y_train,
    X_test,
    scaling: str = "standard",
    k: int = 10,
) -> dict[str, Any]:
    """Run scaling + selection and return the transformed splits and artifacts."""
    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train, X_test, method=scaling
    )
    X_train_selected, X_test_selected, selector = select_features(
        X_train_scaled, y_train, X_test_scaled, k=k
    )

    input_features = list(_as_frame(X_train).columns)
    return {
        "X_train": X_train_selected,
        "X_test": X_test_selected,
        "scaler": scaler,
        "selector": selector,
        "scaling_method": scaling,
        "input_features": input_features,
        "selected_features": list(X_train_selected.columns),
        "feature_scores": get_feature_scores(selector, list(X_train_scaled.columns)),
    }


def transform_features(X, scaler, selector) -> pd.DataFrame:
    """Apply an already fitted preprocessor to raw features (used by serving)."""
    frame = _as_frame(X).copy()
    if scaler is not None:
        frame = _apply(scaler, frame)
    if selector is not None:
        columns = list(frame.columns[selector.get_support()])
        frame = frame.loc[:, columns]
    return frame
