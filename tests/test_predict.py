import pytest

from prodml.predict import DurationPredictor


def test_predict_one_returns_float(trained_model):
    prediction = trained_model.predict_one({"PU_DO": "100_180", "trip_distance": 2.5})

    assert isinstance(prediction, float)


def test_predict_one_returns_sane_duration(trained_model):
    prediction = trained_model.predict_one({"PU_DO": "100_180", "trip_distance": 2.5})

    assert 0 < prediction < 180


def test_predict_one_is_deterministic(trained_model):
    features = {"PU_DO": "100_180", "trip_distance": 2.5}

    first = trained_model.predict_one(features)
    second = trained_model.predict_one(features)

    assert first == second


def test_predict_without_loaded_model_raises():
    predictor = DurationPredictor()

    with pytest.raises(RuntimeError, match="Model is not loaded"):
        predictor.predict_one({"PU_DO": "100_180", "trip_distance": 2.5})


def test_predict_batch_returns_one_prediction_per_trip(trained_model):
    predictions = trained_model.predict_batch(
        [
            {"PU_DO": "100_180", "trip_distance": 2.5},
            {"PU_DO": "100_200", "trip_distance": 4.0},
        ]
    )

    assert predictions == [12.5, 12.5]


def test_predict_batch_empty_input_returns_empty_list(trained_model):
    assert trained_model.predict_batch([]) == []
