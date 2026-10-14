import logging
import pickle
from collections.abc import Callable
from functools import wraps
from time import perf_counter
from typing import Any, TypeVar, cast

from prodml.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def timed[T](func: Callable[..., T]) -> Callable[..., T]:
    """Log prediction results and execution latency."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (perf_counter() - start) * 1000

        if func.__name__ == "predict_one":
            logger.info(
                "Prediction served: prediction=%.3f minutes latency_ms=%.2f",
                result,
                elapsed_ms,
            )
        else:
            logger.info(
                "%s completed in %.2f ms",
                func.__name__,
                elapsed_ms,
            )

        return result

    return cast(Callable[..., T], wrapper)


class DurationPredictor:
    """Load a trained duration model and predict trip durations."""

    def __init__(self) -> None:
        self.dv: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the fitted vectorizer and model from disk."""
        try:
            with settings.model_path.open("rb") as model_file:
                artifact = pickle.load(model_file)

            self.dv = artifact["dv"]
            self.model = artifact["model"]
        except Exception:
            logger.exception(
                "Failed to load model artifact from %s",
                settings.model_path,
            )
            raise

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Predict the duration of one trip in minutes."""
        if self.model is None or self.dv is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        logger.debug("Prediction input features: %s", features)

        distance = features.get("trip_distance")
        if isinstance(distance, (int, float)) and distance > 100:
            logger.warning("Unusually large trip distance: %.2f", distance)

        vectorized = self.dv.transform([features])
        logger.debug("Transformed feature vector shape: %s", vectorized.shape)

        return float(self.model.predict(vectorized)[0])

    def predict_batch(self, features: list[dict[str, Any]]) -> list[float]:
        """Predict durations for multiple trips in minutes."""
        if self.model is None or self.dv is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        if not features:
            return []

        for trip_features in features:
            distance = trip_features.get("trip_distance")
            if isinstance(distance, (int, float)) and distance > 100:
                logger.warning(
                    "Unusually large trip distance: %.2f",
                    distance,
                )

        vectorized = self.dv.transform(features)
        logger.debug("Transformed feature vector shape: %s", vectorized.shape)
        predictions = [float(value) for value in self.model.predict(vectorized)]

        logger.info("Batch predictions served: %d trips", len(predictions))
        return predictions
