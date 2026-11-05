def test_health_returns_200(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "1.0.0",
    }


def test_predict_happy_path(client, sample_features):
    payload = {
        "pu_do": sample_features["PU_DO"],
        "trip_distance": sample_features["trip_distance"],
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 12.5
    assert body["model_version"] == "1.0.0"


def test_predict_invalid_payload_returns_422(client):
    response = client.post(
        "/predict",
        json={"pu_do": "100_180", "trip_distance": 0},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Request validation failed"
    assert body["errors"]


def test_predict_response_schema(client, sample_features):
    response = client.post(
        "/predict",
        json={
            "pu_do": sample_features["PU_DO"],
            "trip_distance": sample_features["trip_distance"],
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "prediction",
        "model_version",
        "correlation_id",
        "latency_ms",
    }
    assert body["prediction"] == 12.5
    assert body["model_version"] == "1.0.0"
    assert body["correlation_id"]
    assert body["latency_ms"] >= 0


def test_correlation_id_is_preserved(client, sample_features):
    correlation_id = "test-request-123"

    response = client.post(
        "/predict",
        headers={"X-Correlation-ID": correlation_id},
        json={
            "pu_do": sample_features["PU_DO"],
            "trip_distance": sample_features["trip_distance"],
        },
    )

    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == correlation_id
    assert response.json()["correlation_id"] == correlation_id


def test_batch_prediction_returns_one_result_per_trip(client):
    response = client.post(
        "/predict/batch",
        json={
            "predictions": [
                {"pu_do": "100_180", "trip_distance": 2.5},
                {"pu_do": "100_190", "trip_distance": 4.0},
            ]
        },
    )

    assert response.status_code == 200
    results = response.json()["predictions"]
    assert len(results) == 2
    assert all(item["prediction"] == 12.5 for item in results)


def test_batch_prediction_rejects_empty_list(client):
    response = client.post(
        "/predict/batch",
        json={"predictions": []},
    )

    assert response.status_code == 422
