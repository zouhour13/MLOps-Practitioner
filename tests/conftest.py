import numpy as np
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.predict import DurationPredictor


class FakeVectorizer:
    def transform(self, features):
        return np.zeros((len(features), 2))


class FakeModel:
    def predict(self, features):
        return np.array([12.5 for _ in features])


@pytest.fixture
def sample_features():
    return {"PU_DO": "100_180", "trip_distance": 2.5}


@pytest.fixture(scope="session")
def trained_model():
    predictor = DurationPredictor()
    predictor.dv = FakeVectorizer()
    predictor.model = FakeModel()
    return predictor


@pytest.fixture
def client(monkeypatch):
    def fake_load(self):
        self.dv = FakeVectorizer()
        self.model = FakeModel()

    monkeypatch.setattr(DurationPredictor, "load", fake_load)

    with TestClient(app) as test_client:
        yield test_client
