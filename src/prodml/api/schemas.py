"""Pydantic request and response schemas for the prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input features for one trip-duration prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "pu_do": "100_180",
                    "trip_distance": 2.5,
                }
            ]
        }
    )

    pu_do: str = Field(
        min_length=3,
        max_length=15,
        pattern=r"^\d+_\d+$",
        description="Pickup/dropoff location pair, for example '100_180'.",
    )
    trip_distance: float = Field(
        gt=0,
        lt=200,
        description="Trip distance; must be greater than 0 and less than 200.",
        examples=[2.5],
    )


class PredictionResponse(BaseModel):
    """Prediction result and request metadata."""

    prediction: float = Field(description="Predicted trip duration in minutes.")
    model_version: str
    correlation_id: str
    latency_ms: float = Field(ge=0)


class BatchPredictionRequest(BaseModel):
    """Input for predicting multiple trips in one request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "predictions": [
                        {"pu_do": "100_180", "trip_distance": 2.5},
                        {"pu_do": "100_200", "trip_distance": 4.0},
                    ]
                }
            ]
        }
    )

    predictions: list[PredictionRequest] = Field(
        min_length=1,
        max_length=100,
        description="Between 1 and 100 trips to predict.",
    )


class BatchPredictionResponse(BaseModel):
    """Prediction results for a batch of trips."""

    predictions: list[PredictionResponse]
