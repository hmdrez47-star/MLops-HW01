"""FastAPI model serving: ``/health``, ``/predict`` and ``/metadata``.

The service loads ``models/model.joblib``, the fitted preprocessor (scaler +
selector) and ``models/metadata.json`` at startup, so a request can send the 12
raw features and get back a prediction plus a confidence.
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.monitoring.drift_detection import log_prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))
PREDICTION_LOG_PATH = os.getenv(
    "PREDICTION_LOG_PATH", str(PROJECT_ROOT / "experiments" / "logs")
)

MODEL = None
METADATA: dict = {}
PREPROCESSOR: dict | None = None
STARTED_AT = time.time()


def _load_artifacts() -> None:
    """Load model + metadata + preprocessor from ``MODEL_DIR`` (if present)."""
    global MODEL, METADATA, PREPROCESSOR

    model_path = MODEL_DIR / "model.joblib"
    if not model_path.exists():
        MODEL, METADATA, PREPROCESSOR = None, {}, None
        return

    import joblib

    MODEL = joblib.load(model_path)

    metadata_path = MODEL_DIR / "metadata.json"
    if metadata_path.exists():
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            METADATA = json.load(handle)

    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    PREPROCESSOR = (
        joblib.load(preprocessor_path) if preprocessor_path.exists() else None
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="ML Model Serving API",
    version="1.0.0",
    description="Churn prediction service (MLOps Homework 01)",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        description="Raw feature vector, ordered as metadata.input_features",
    )


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    latency: float


def _expected_features() -> list[str]:
    """Feature order the model expects (falls back to metadata, then config)."""
    if PREPROCESSOR and PREPROCESSOR.get("input_features"):
        return list(PREPROCESSOR["input_features"])
    if METADATA.get("input_features"):
        return list(METADATA["input_features"])
    return []


def _prepare(features: list[float]) -> pd.DataFrame:
    """Build the model input matrix (scale + select) for one raw sample."""
    expected = _expected_features()
    if expected and len(features) != len(expected):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_feature_count",
                "expected": len(expected),
                "received": len(features),
            },
        )

    columns = expected or [f"feature_{i}" for i in range(len(features))]
    frame = pd.DataFrame([features], columns=columns)

    if PREPROCESSOR is None:
        return frame

    scaler = PREPROCESSOR.get("scaler")
    selector = PREPROCESSOR.get("selector")
    if scaler is not None:
        frame = pd.DataFrame(
            scaler.transform(frame), columns=frame.columns, index=frame.index
        )
    if selector is not None:
        frame = frame.loc[:, list(frame.columns[selector.get_support()])]
    return frame


def _confidence(model, X: pd.DataFrame, prediction: int) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X))[0]
        return float(probabilities[int(prediction)])
    return 1.0


@app.get("/health")
def health_check():
    """Liveness/readiness probe with the model version currently served."""
    loaded = MODEL is not None
    return {
        "status": "ok" if loaded else "degraded",
        "model_version": METADATA.get("model_version", "unknown"),
        "data_version": METADATA.get("data_version", "unknown"),
        "model_loaded": loaded,
        "uptime_seconds": round(time.time() - STARTED_AT, 3),
    }


@app.get("/metadata")
def model_metadata():
    """Metadata of the artifact currently being served."""
    if not METADATA:
        raise HTTPException(status_code=503, detail="model_not_loaded")
    return METADATA


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict churn for one sample and log the prediction (no raw features)."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model_not_loaded")

    started = time.perf_counter()
    X = _prepare(request.features)

    try:
        prediction = int(np.asarray(MODEL.predict(X)).ravel()[0])
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"prediction_failed: {type(exc).__name__}"
        ) from exc

    confidence = _confidence(MODEL, X, prediction)
    latency = time.perf_counter() - started

    log_prediction(
        prediction=prediction,
        confidence=confidence,
        latency=latency,
        log_path=PREDICTION_LOG_PATH,
        model_version=METADATA.get("model_version", "unknown"),
    )

    return PredictionResponse(
        prediction=prediction,
        confidence=round(confidence, 4),
        latency=round(latency, 6),
    )
