# Module 1 — From Notebook to Production-Ready Code

**Validation RMSE:** 7.101 minutes  
**Validation MAE:** 4.939 minutes

## Mini Project 1: Baseline Model

### 1. Objective

Build a baseline regression model to predict NYC Green Taxi trip duration in minutes. This experiment establishes a reference point for evaluating improvements during the productionization process.

### 2. Dataset

- **Source:** NYC TLC Green Taxi Trip Records
- **Period:** January 2021
- **Format:** Parquet
- **Original observations:** 76,518
- **Observations after cleaning:** 72,282
- **Observations removed:** 4,236

### 3. Data Preparation

The target variable, `duration`, was calculated as the difference between the drop-off and pickup timestamps, expressed in minutes.

Cleaning criteria:
- Trip duration between 1 and 60 minutes, inclusive.
- Positive `trip_distance`.
- Non-missing pickup and drop-off location IDs.

Features:
- `PU_DO`: concatenation of pickup and drop-off location IDs.
- `trip_distance`: recorded trip distance.

### 4. Model and Methodology

- **Feature vectorization:** `DictVectorizer`
- **Regression algorithm:** `LinearRegression`
- **Training/validation split:** 80% / 20%
- **Random state:** 42
- **Target:** Trip duration in minutes

The vectorizer was fitted exclusively on the training data and then used to transform the validation data.

### 5. Baseline Results

| Metric | Validation result |
|---|---:|
| Mean Absolute Error (MAE) | 4.939 minutes |
| Root Mean Squared Error (RMSE) | 7.101 minutes |

The model's predictions have a mean absolute error of approximately 4.94 minutes on the validation set. The RMSE of 7.10 minutes indicates that larger prediction errors have a meaningful impact on overall performance.

These metrics establish the initial baseline for subsequent experiments.

### 6. Artifacts

- Notebook: `notebooks/00-baseline.ipynb`
- Serialized model and vectorizer: `models/baseline.pkl`

### 7. Next Steps

1. Refactor notebook code into a reusable, installable Python package.
2. Add structured logging and improve error handling.
3. Compare model serialization formats, including Pickle and ONNX.
4. Expose predictions through a FastAPI service.
5. Add automated tests and containerize the application with Docker.

### 8. Conclusion

The baseline notebook loads and cleans the dataset, engineers the required features, trains a linear regression model, and evaluates its performance. The resulting MAE and RMSE provide reference metrics for the next stages of the MLOps workflow.

---

## Module 04 — Model Serialization

### Pickle vs ONNX: Inference Benchmark

Both formats were evaluated on the same 500 validation examples. The benchmark performed 100 timed batch-inference repetitions after a warm-up. Feature vectorization and ONNX dense-array conversion were performed before timing, so preprocessing is excluded.

| Metric | Pickle / scikit-learn | ONNX Runtime |
|---|---:|---:|
| Validation batch size | 500 rows | 500 rows |
| Mean latency (ms/batch) | 0.1325 | 1.3905 |
| p95 latency (ms/batch) | 0.1719 | 2.6175 |

The parity test passed with an absolute tolerance of `1e-4`. In this local inference-only benchmark, Pickle/scikit-learn was faster than ONNX Runtime. These measurements describe this environment and workload; they do not establish that one format is universally faster. In particular, preprocessing costs are excluded, and ONNX receives dense input while scikit-learn uses sparse input.

### Serialization Format Comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No built-in schema; JSON Schema can provide validation | Generally safe to parse, but validate the data |
| Protobuf | No, normally binary (a text format also exists) | Yes | Yes, using a `.proto` schema | Generally safer than executable object formats; validate input |
| Pickle | No | No, Python-specific | No | **No** — loading can execute arbitrary code |
| ONNX | No, normally a binary model graph | Yes, through compatible runtimes | Partially — defines typed tensors and graph structure, not a complete application-level schema | Safer than Pickle against Python-code execution, but only use trusted, validated model artifacts |

### Service Format Decision

The prediction service will use **ONNX for regression inference** because it supports interoperable inference across languages and runtimes; the trusted Pickle artifact is retained for the fitted `DictVectorizer` needed to prepare input features.

### Security Note

Never load a Pickle artifact from an untrusted source: deserialization can execute arbitrary code. Model artifacts should come from a trusted build or training process.

---

## Module 1 — Testing and Regression Validation

### 1. Objective

The objective of this stage was to establish an automated testing strategy for the productionized ML application.

Beyond checking whether the application executes successfully, the test suite must validate expected behavior, detect incorrect outputs, and protect important components against regressions.

The testing strategy covers:

- Feature engineering and data transformations.
- Prediction logic and batch inference.
- API endpoints and request validation.
- Serialization parity between Pickle and ONNX.
- Serialization benchmark behavior.
- Error handling and invalid inputs.

The project uses `pytest` for automated testing, fixtures and mocks for dependency isolation, and `pytest-cov` to measure code coverage.

### 2. Test Suite Structure

The automated tests are organized as follows:

| Test file | Main responsibility |
|---|---|
| `tests/conftest.py` | Shared fixtures and test configuration |
| `tests/test_api.py` | API endpoints, request validation and response schemas |
| `tests/test_features.py` | Feature engineering and vectorization |
| `tests/test_predict.py` | Single and batch prediction behavior |
| `tests/test_serialization.py` | Pickle and ONNX prediction parity |
| `tests/test_benchmark_serialization.py` | Latency measurement and benchmark execution |

This structure separates testing responsibilities and makes individual components easier to validate and maintain.

### 3. Coverage Configuration

A minimum coverage threshold of 70% is enforced directly in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=src/prodml --cov-fail-under=70"
```

The following command executes the complete test suite and displays the coverage report:

```bash
pytest -v --cov=src/prodml --cov-report=term-missing
```

The `--cov-fail-under=70` option ensures that the test command fails if total statement coverage falls below the required threshold.

### 4. Final Test Results

The final test execution produced the following result:

```text
25 passed, 1 warning in 4.03s

Required test coverage of 70% reached.
Total coverage: 72.12%
```

| Metric | Result |
|---|---:|
| Total tests | 25 |
| Passed tests | 25 |
| Failed tests | 0 |
| Coverage threshold | 70% |
| Achieved coverage | 72.12% |

The project successfully exceeded the required coverage threshold.


### 5. Module-Level Coverage Analysis

The final coverage report provides the following distribution:

| Module | Statements | Missed | Coverage |

|---|---:|---:|---:|

| `api/main.py` | 96 | 22 | 77% |

| `api/schemas.py` | 15 | 0 | 100% |

| `benchmark_serialization.py` | 42 | 3 | 93% |

| `config.py` | 16 | 0 | 100% |

| `data.py` | 15 | 0 | 100% |

| `export.py` | 24 | 24 | 0% |

| `features.py` | 15 | 0 | 100% |

| `logging_conf.py` | 19 | 1 | 95% |

| `predict.py` | 58 | 12 | 79% |

| `train.py` | 30 | 30 | 0% |

| **Total** | **330** | **92** | **72.12%** |

The feature-engineering, configuration, data-loading and API schema modules achieved 100% statement coverage.

The `train.py` and `export.py` modules remain untested and represent opportunities for further improvement. Their uncovered statements do not prevent the project from meeting the enforced 70% threshold.

### 6. Mutation Testing: Verifying Regression Detection

#### 6.1 Why Mutation Testing?

Code coverage measures how much code is executed during testing, but it does not necessarily indicate whether assertions are strong enough to detect incorrect behavior.

For example, a test may execute a feature transformation without verifying that the resulting feature value is correct.

To evaluate the effectiveness of the existing feature tests, a deliberate mutation experiment was performed on the `create_features()` function.

#### 6.2 Mutation Scenario

The experiment targeted the construction of the `PU_DO` feature.

The original implementation concatenates pickup and drop-off location IDs using an underscore:

```python

features["PU_DO"] = (

    features["PULocationID"].astype(int).astype(str)

    + "_"

    + features["DOLocationID"].astype(int).astype(str)

)

```

A deliberate incorrect implementation replaced the underscore (`_`) with a hyphen (`-`).

For example:

| Pickup ID | Drop-off ID | Expected output | Mutated output |

|---:|---:|---|---|

| 100 | 180 | `100_180` | `100-180` |

| 1 | 2 | `1_2` | `1-2` |

| 265 | 265 | `265_265` | `265-265` |

Although this modification is small, it changes the categorical feature representation. Consequently, it can affect compatibility with the feature vectorizer fitted during training.

#### 6.3 Regression Detection Results

The targeted feature tests were executed against the incorrect implementation:

```bash

pytest -q tests/test_features.py --no-cov

```

Observed results:

| Result | Number |

|---|---:|

| Failed tests | 3 |

| Passed tests | 5 |

| Total tests | 8 |

The three parametrized cases failed because the actual generated feature did not match the expected value.

The assertions detected all three incorrect outputs:

```text

Expected: 100_180

Actual:   100-180

Expected: 1_2

Actual:   1-2

Expected: 265_265

Actual:   265-265

```

This demonstrates that the existing tests validate the actual feature representation rather than simply checking that the function executes.

#### 6.4 Restoration and Regression Verification

After the experiment, the original implementation using the underscore separator was restored.

The complete test suite was executed again:

```bash

pytest -v --cov=src/prodml --cov-report=term-missing

```

The final execution confirmed:

```text

25 passed, 1 warning in 4.03s

Required test coverage of 70% reached.

Total coverage: 72.12%

```

All 25 tests passed after restoration, and the original feature-engineering behavior was preserved.

No deliberately incorrect implementation was retained in the production code.

#### 6.5 Interpretation

The experiment demonstrates the distinction between two important testing concepts:

- **Code coverage:** Measures the proportion of source statements executed during testing.

- **Regression detection:** Determines whether tests identify incorrect behavior when the implementation changes.

Mutation testing provides an additional way to assess test effectiveness by introducing controlled defects and observing whether existing assertions detect them.

In this project, the incorrect `PU_DO` separator was detected by all three parametrized test cases. This provides concrete evidence that the feature-engineering tests protect an important part of the model's input representation.

### 7. Testing Acceptance Summary

| Acceptance criterion | Result |

|---|---|

| Automated test suite executes successfully | Passed |

| Minimum coverage of 70% | Passed — 72.12% |

| Feature engineering coverage | Passed — 100% |

| Deliberate feature regression detected | Passed — 3 failures |

| Original implementation restored | Passed |

| Full regression suite passes after restoration | Passed — 25 tests |

| Coverage threshold enforced in configuration | Passed |

### 8. Conclusion

The testing stage established an automated validation strategy for the productionized ML application.

The project now includes tests for feature engineering, prediction logic, API behavior, serialization parity and benchmark execution. The coverage threshold is enforced through the pytest configuration, and the final suite achieves 72.12% statement coverage.

The mutation experiment provides additional evidence that the tests can detect incorrect feature transformations rather than merely executing the code.

Together, these practices improve the reliability and maintainability of the application and establish a foundation for further productionization.

---

**## Module 1 — Docker Packaging and Image-Size Comparison
1. Objective
The objective of this stage was to containerize the FastAPI prediction service, produce a reproducible Docker image, and evaluate the effect of Docker build configuration on the resulting image.
2. Production Docker Image
The application was packaged using a multi-stage Dockerfile based on python:3.12-slim. Python 3.12 was selected to satisfy the project's declared Python requirement (>=3.12,<3.14).

The runtime image includes the API dependencies and serialized model artifact. The container is configured to run the FastAPI application with Uvicorn on port 8000, use the non-root user appuser, load the model from /app/models/model.pkl, and expose a Docker health check that calls /health.

The versioned image and latest tag were published to Docker Hub:
- Docker Hub repository: https://hub.docker.com/r/zouhour13/prodml-api
- Version tag: zouhour13/prodml-api:0.1.0
- Latest tag: zouhour13/prodml-api:latest
The published version was also pulled successfully from Docker Hub.
3. .dockerignore Image-Size Experiment
Two builds were performed using the same production Dockerfile: one with .dockerignore enabled and one without it.

The image sizes were inspected using:
docker image inspect prodml-api:with-ignore --format='{{.Size}}'
docker image inspect prodml-api:without-ignore --format='{{.Size}}'
Results:
Build configuration	Image size (bytes)	Approximate size
With .dockerignore	130,871,277	130.87 MB
Without .dockerignore	130,870,576	130.87 MB
Difference	701	Approximately 0.0007 MB


The two final images are practically identical in size. The image built without .dockerignore was 701 bytes smaller, a negligible difference that should not be interpreted as an improvement.

This result is explained by the production Dockerfile's selective COPY instructions: it copies only the required project files and model artifact rather than the entire repository. Consequently, files excluded by .dockerignore were not included in either final image.

This experiment illustrates an important distinction:
- .dockerignore controls which files are included in the Docker build context.
- COPY instructions determine which files are added to the image.
- Excluding unnecessary files from the build context remains useful for reducing unnecessary context transfer and avoiding accidental inclusion of irrelevant or sensitive files, even when the final image size does not change.
The experiment measured final image sizes; it did not measure build-context size.
4. Docker Validation
The container was started and validated locally.
Validation	Result
Docker image build	Passed
Container runs as appuser	Passed
/health endpoint	HTTP 200
/predict endpoint	Passed
Model loaded at application startup	Confirmed
Docker health check	Healthy
Docker Compose service	Started and healthy
Docker Hub image pull	Passed


A prediction request using pu_do = "100_180" and trip_distance = 2.5 returned a prediction of approximately 34.98 minutes.
5. Interpretation
The final production image size was approximately 130.87 MB in both .dockerignore configurations. The negligible 701-byte difference is consistent with the Dockerfile's targeted file-copy strategy.

The .dockerignore file is retained as a build-hygiene and security measure. It helps prevent unnecessary repository content from entering the build context, while the Dockerfile independently limits what is copied into the runtime image.

The single-stage versus multi-stage image-size comparison is a separate experiment and should be documented once both corresponding image sizes have been measured.
Module 1 — Overall Progress and Remaining Deliverables**
| Deliverable | Status |

|---|---|

| Baseline regression model and evaluation | Completed |

| Reusable Python package structure | Completed |

| Structured logging | Implemented |

| Pickle and ONNX serialization | Completed |

| Serialization parity validation | Passed |

| Inference latency benchmark | Completed |

| FastAPI prediction service | Implemented |

| Automated test suite | 25 tests passing |

| Coverage enforcement | 72.12% — passed |

| Mutation experiment | Regression detected |

| Pre-commit quality checks | Passed |

| Git commit | `020d401` |

| Multi-stage Docker build | Completed — image built and validated |

| Docker Hub publication | Completed — versioned image pushed and pulled successfully |
| .dockerignore image-size comparison | Completed — 130,871,277 bytes with ignore; 130,870,576 bytes without ignore |

| README rewrite | Pending |

| Maturity self-assessment | Pending |

| Pull request and `v0.1.0` release | Pending |

**Next milestone:** Containerize the FastAPI prediction service using a multi-stage Docker build, run the application as a non-root user, validate the API endpoints, compare image sizes, and publish the resulting image to Docker Hub.
