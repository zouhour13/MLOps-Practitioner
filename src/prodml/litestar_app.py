"""Optional Litestar implementation for comparing API frameworks."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from litestar import Litestar, Request, post
from litestar.exceptions import HTTPException

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.predict import DurationPredictor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncIterator[None]:
    """Load the model once during application startup."""
    predictor = DurationPredictor()
    predictor.load()
    app.state.predictor = predictor
    logger.info("Litestar prediction model loaded at startup")
    try:
        yield
    finally:
        logger.info("Litestar prediction API shutting down")


@post("/predict", status_code=200, sync_to_thread=True)
def predict(
    data: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    """Predict the duration of one trip."""
    predictor: DurationPredictor = request.app.state.predictor

    correlation_id = request.headers.get("x-correlation-id") or str(uuid4())
    start_time = perf_counter()

    try:
        prediction = predictor.predict_one(
            {
                "PU_DO": data.pu_do,
                "trip_distance": data.trip_distance,
            }
        )
    except Exception:
        logger.exception(
            "Unexpected Litestar prediction error",
            extra={"correlation_id": correlation_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from None

    latency_ms = (perf_counter() - start_time) * 1000

    return PredictionResponse(
        prediction=prediction,
        model_version="1.0.0",
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )


app = Litestar(
    route_handlers=[predict],
    lifespan=[lifespan],
)
