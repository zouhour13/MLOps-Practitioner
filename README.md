# MLOps Practitioner — Module 1

This project transforms a baseline machine learning model for NYC Green Taxi trip-duration prediction into a containerized prediction API. It demonstrates an end-to-end introductory MLOps workflow, including data preparation, feature engineering, model training, Pickle and ONNX serialization, inference benchmarking, automated testing, structured logging, and Docker packaging. The public Docker image makes the prediction service available without requiring users to clone the repository or install the Python dependencies.

## Quick start

**Prerequisite:** Docker must be installed and running.

**1. Pull the image**

```bash
docker pull zouhour13/prodml-api:0.1.0
```

**2. Start the API**

```bash
docker run -d --name prodml-api -p 8000:8000 zouhour13/prodml-api:0.1.0
```

**3. Request a prediction**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pu_do":"100_180","trip_distance":2.5}'
```

Example JSON input:

```json
{
  "pu_do": "100_180",
  "trip_distance": 2.5
}
```

The response contains the predicted trip duration in minutes.

To verify the service is healthy:

```bash
curl http://localhost:8000/health
```

To stop and remove the container:

```bash
docker stop prodml-api && docker rm prodml-api
```

**Docker Hub:** https://hub.docker.com/r/zouhour13/prodml-api

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| GET | `/metadata` | Model and service metadata |
| POST | `/predict` | Predict one trip duration |
| POST | `/predict/batch` | Predict multiple trip durations |

### Prediction request

```json
{
  "pu_do": "100_180",
  "trip_distance": 2.5
}
```

- `pu_do`: pickup and drop-off location IDs combined into one categorical feature.
- `trip_distance`: trip distance.

## Model

The model is trained using the January 2021 NYC Green Taxi dataset.

| Property | Value |
|---|---|
| Prediction target | Trip duration (minutes) |
| Features | `PU_DO`, `trip_distance` |
| Vectorization | `DictVectorizer` |
| Estimator | Linear Regression |
| Validation MAE | 4.939 minutes |
| Validation RMSE | 7.101 minutes |

## Serialization and benchmarking

The project evaluates Pickle and ONNX serialization.

- Pickle stores the Python model and preprocessing workflow.
- ONNX provides a portable inference representation executed with ONNX Runtime.
- Numerical parity was validated with an absolute tolerance of `1e-4`.
- Inference latency was measured after warmup over repeated batches.
- Preprocessing and ONNX input conversion were excluded from the inference timing.

## Testing and quality

- 25 automated tests passing.
- 72.46% code coverage, exceeding the enforced 70% minimum.
- Tests cover feature engineering, prediction, serialization, benchmarking, and API behavior.
- A mutation experiment changed the `PU_DO` separator from `_` to `-`; regression tests detected the change.
- Ruff, Black, and pre-commit quality checks passed.

## Docker image

The application uses a multi-stage build based on `python:3.12-slim`.

- Runs as a non-root user (`appuser`).
- Loads the model at startup.
- Exposes port `8000`.
- Includes a Docker health check.

Published image tags:

- `zouhour13/prodml-api:0.1.0`
- `zouhour13/prodml-api:latest`

The public image was pulled and tested independently of the source repository. Health, metadata, single-prediction, and batch-prediction checks succeeded.

## Repository structure

```text
MLOps-Practitioner/
├── data/
│   └── green_tripdata_2021-01.parquet
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   ├── baseline.pkl
│   ├── model.pkl
│   └── model.onnx
├── notebooks/
│   └── 00-baseline.ipynb
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       ├── benchmark_serialization.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_benchmark_serialization.py
│   ├── test_features.py
│   ├── test_predict.py
│   └── test_serialization.py
├── .dockerignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
└── uv.lock
```

## Scope

Module 1 focuses on turning a baseline machine learning model into a tested, containerized prediction service. Automated pipelines, continuous deployment, and production monitoring are outside the scope of this module.
