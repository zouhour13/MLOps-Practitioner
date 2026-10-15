"""FastAPI application for trip-duration predictions."""

import hashlib
import logging
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import configure_logging, correlation_id_var
from prodml.predict import DurationPredictor

configure_logging()
logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


def artifact_sha256(path: Path) -> str:
    """Calculate the SHA-256 hash of a model artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the prediction model once when the application starts."""
    predictor = DurationPredictor()
    predictor.load()
    app.state.predictor = predictor
    app.state.model_loaded = predictor.model is not None and predictor.dv is not None
    logger.info("Prediction model loaded at startup")
    try:
        yield
    finally:
        app.state.model_loaded = False
        logger.info("Prediction API shutting down")


app = FastAPI(
    title="ProdML Trip Duration API",
    description="Predict taxi trip duration using a trained regression model.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Any):
    """Attach a correlation ID to each request and its logs."""
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid4())
    token = correlation_id_var.set(correlation_id)
    request.state.correlation_id = correlation_id

    try:
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
    finally:
        correlation_id_var.reset(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Return readable validation errors without a traceback."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid4()))
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": errors,
            "correlation_id": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Log unexpected errors while keeping internal details out of responses."""
    correlation_id = getattr(request.state, "correlation_id", str(uuid4()))
    logger.exception("Unexpected API error")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": correlation_id,
        },
    )


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Report healthy only when the model and vectorizer are loaded."""
    predictor = getattr(request.app.state, "predictor", None)
    if (
        predictor is None
        or predictor.model is None
        or predictor.dv is None
        or not getattr(request.app.state, "model_loaded", False)
    ):
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metadata")
def metadata() -> dict[str, Any]:
    """Return model version, artifact details, and feature names."""
    artifact_path = settings.model_path
    if not artifact_path.is_file():
        raise HTTPException(status_code=503, detail="Model artifact not found")

    try:
        import pickle

        with artifact_path.open("rb") as model_file:
            artifact = pickle.load(model_file)
        feature_names = list(artifact["dv"].feature_names_)
    except Exception:
        logger.exception("Failed to read model metadata")
        raise HTTPException(
            status_code=500,
            detail="Unable to read model metadata",
        ) from None

    modified_at = datetime.fromtimestamp(
        artifact_path.stat().st_mtime,
        tz=UTC,
    )
    return {
        "model_version": MODEL_VERSION,
        "training_date": modified_at.isoformat(),
        "feature_names": feature_names,
        "framework": "scikit-learn",
        "artifact": artifact_path.name,
        "artifact_sha256": artifact_sha256(artifact_path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Predict the duration of one trip."""
    predictor: DurationPredictor = request.app.state.predictor
    correlation_id = request.state.correlation_id
    start = time.perf_counter()

    prediction = predictor.predict_one(
        {
            "PU_DO": payload.pu_do,
            "trip_distance": payload.trip_distance,
        }
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_version=MODEL_VERSION,
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    """Predict durations for a batch of trips."""
    predictor: DurationPredictor = request.app.state.predictor
    correlation_id = request.state.correlation_id
    start = time.perf_counter()

    features = [
        {
            "PU_DO": item.pu_do,
            "trip_distance": item.trip_distance,
        }
        for item in payload.predictions
    ]
    predictions = predictor.predict_batch(features)
    latency_ms = (time.perf_counter() - start) * 1000

    per_item_latency = latency_ms / len(predictions)
    results = [
        PredictionResponse(
            prediction=value,
            model_version=MODEL_VERSION,
            correlation_id=correlation_id,
            latency_ms=per_item_latency,
        )
        for value in predictions
    ]
    return BatchPredictionResponse(predictions=results)
