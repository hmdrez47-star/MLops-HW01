"""Pytest fixtures shared by the test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_data import FEATURE_COLUMNS

REAL_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_v1.csv"
DRIFT_DATASET = PROJECT_ROOT / "data" / "raw" / "dataset_v2.csv"


def make_frame(n_samples: int = 300, seed: int = 7) -> pd.DataFrame:
    """Synthetic frame with the real schema and a learnable churn signal."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "tenure_months": rng.uniform(1, 72, n_samples).round(2),
            "monthly_charges": rng.uniform(20, 130, n_samples).round(2),
            "num_services": rng.integers(1, 7, n_samples),
            "support_tickets": rng.poisson(1.5, n_samples),
            "avg_session_minutes": rng.uniform(5, 60, n_samples).round(2),
            "login_frequency": rng.uniform(1, 30, n_samples).round(2),
            "contract_length": rng.choice([1, 12, 24], n_samples),
            "payment_delay_days": rng.uniform(0, 30, n_samples).round(2),
            "credit_score": rng.integers(500, 850, n_samples),
            "age": rng.integers(18, 80, n_samples),
            "num_dependents": rng.integers(0, 5, n_samples),
        }
    )
    frame["total_charges"] = (frame["tenure_months"] * frame["monthly_charges"]).round(
        2
    )

    logits = (
        -0.04 * frame["tenure_months"]
        + 0.7 * frame["support_tickets"]
        + 0.02 * frame["monthly_charges"]
        - 0.01 * frame["credit_score"]
        + 0.05 * frame["payment_delay_days"]
    )
    probability = 1.0 / (1.0 + np.exp(-(logits - logits.mean())))
    frame["churn"] = (rng.random(n_samples) < probability).astype(int)

    return frame[FEATURE_COLUMNS + ["churn"]]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def real_dataset_path() -> Path:
    return REAL_DATASET


@pytest.fixture
def drift_dataset_path() -> Path:
    return DRIFT_DATASET
