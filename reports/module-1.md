# Module 1 — From Notebook to Production-Ready Code

## Project Overview

This module transforms a baseline NYC Green Taxi trip-duration regression notebook into a reusable Python package and a containerized prediction service. The objective is to establish a production-oriented foundation through modular code, structured logging, model serialization, API development, automated testing, and Docker deployment.

**Baseline validation results**

| Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 4.939 minutes |
| Root Mean Squared Error (RMSE) | 7.101 minutes |

---

## 1. Baseline Model

### 1.1 Objective

Build a baseline regression model to predict NYC Green Taxi trip duration in minutes. This experiment establishes a reference point for evaluating improvements during the productionization process.

### 1.2 Dataset

| Property | Description |
|---|---|
| Source | NYC TLC Green Taxi Trip Records |
| Period | January 2021 |
| Format | Parquet |
| Original observations | 76,518 |
| Observations after cleaning | 72,282 |
| Observations removed | 4,236 |

### 1.3 Data Preparation

The target variable, `duration`, was calculated as the difference between the drop-off and pickup timestamps, expressed in minutes.

The following cleaning criteria were applied:

- Trip duration between 1 and 60 minutes, inclusive.
- Positive `trip_distance`.
- Non-missing pickup and drop-off location IDs.

Two features were selected:

- `PU_DO`: concatenation of pickup and drop-off location IDs.
- `trip_distance`: recorded trip distance.

### 1.4 Model and Methodology

| Component | Configuration |
|---|---|
| Feature vectorization | `DictVectorizer` |
| Regression algorithm | `LinearRegression` |
| Training/validation split | 80% / 20% |
| Random state | 42 |
| Target | Trip duration in minutes |

The vectorizer was fitted exclusively on the training data and then used to transform the validation data, avoiding data leakage during feature preparation.

### 1.5 Baseline Results

| Metric | Validation result |
|---|---:|
| Mean Absolute Error (MAE) | 4.939 minutes |
| Root Mean Squared Error (RMSE) | 7.101 minutes |

The model's predictions have a mean absolute error of approximately 4.94 minutes on the validation set. The RMSE of 7.10 minutes indicates that larger prediction errors have a meaningful impact on overall performance.

These metrics establish the initial baseline for subsequent experiments.

### 1.6 Artifacts

- Notebook: `notebooks/00-baseline.ipynb`
- Serialized baseline model and vectorizer: `models/baseline.pkl`

### 1.7 Conclusion

The baseline notebook loads and cleans the dataset, engineers the required features, trains a linear regression model, and evaluates its performance. The resulting MAE and RMSE provide reference metrics for the next stages of the MLOps workflow.

---

## 2. Python Packaging and Project Structure

The notebook logic was reorganized into a reusable Python package under `src/prodml/`, separating the main responsibilities of the application.

| Module | Responsibility |
|---|---|
| `data.py` | Dataset loading and preparation |
| `features.py` | Feature engineering and vectorization |
| `train.py` | Model training and persistence |
| `predict.py` | Model loading and prediction |
| `config.py` | Application configuration |
| `logging_conf.py` | Logging configuration |
| `export.py` | Model export and serialization |
| `api/main.py` | FastAPI application and endpoints |
| `api/schemas.py` | Pydantic request and response schemas |
| `benchmark_serialization.py` | Serialization performance comparison |

Project dependencies and packaging metadata are defined in `pyproject.toml`, with a `src/`-based package layout.

This separation makes the code easier to test, maintain, and extend. It also establishes a clear boundary between model logic and the API that serves predictions.

### Engineering practices

The project incorporates:

- A reusable package structure.
- Environment-based application configuration.
- Type hints and modular interfaces.
- Pre-commit quality checks using Ruff, Black, and end-of-file-fixer.
- A dedicated prediction layer used by the API.

---

## 3. Structured Logging

Structured logging was implemented in `src/prodml/logging_conf.py` to produce JSON-formatted application logs.

Each log record includes the following fields:

| Field | Description |
|---|---|
| `timestamp` | Time at which the log record was created |
| `level` | Severity level of the event |
| `logger` | Name of the logger that generated the record |
| `message` | Log message |
| `correlation_id` | Identifier used to associate log events with an API request |

The logging configuration also supports exception details when an exception is logged.

### Request Correlation

The API middleware handles the `X-Correlation-ID` request header. If a client supplies an identifier, the application preserves it; otherwise, it generates one. The identifier is also returned in the response header.

This mechanism helps trace requests across application logs and API responses.

**Outcome:** JSON-formatted logging and request correlation ID handling are implemented. The logging configuration is included in the automated test suite.

---

## 4. Model Serialization — Pickle vs ONNX

### 4.1 Inference Benchmark

Pickle/scikit-learn and ONNX Runtime were evaluated using the same 500 validation examples.

The benchmark performed 100 timed batch-inference repetitions after a warm-up. Feature vectorization and ONNX dense-array conversion were performed before timing, so preprocessing costs are excluded.

| Metric | Pickle / scikit-learn | ONNX Runtime |
|---|---:|---:|
| Validation batch size | 500 rows | 500 rows |
| Mean latency (ms/batch) | 0.1325 | 1.3905 |
| p95 latency (ms/batch) | 0.1719 | 2.6175 |

The parity test passed with an absolute tolerance of `1e-4`.

In this local inference-only benchmark, Pickle/scikit-learn was faster than ONNX Runtime. These measurements describe this particular environment and workload; they do not establish that one format is universally faster. Preprocessing costs are excluded, and ONNX receives dense input while scikit-learn uses sparse input.

### 4.2 Serialization Format Comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No built-in schema; JSON Schema can provide validation | Generally safe to parse, but validate the data |
| Protobuf | Normally binary; a text format also exists | Yes | Yes, using a `.proto` schema | Safer than executable object formats, but input still requires validation |
| Pickle | No | No, Python-specific | No | **No — loading can execute arbitrary code** |
| ONNX | Normally a binary model graph | Yes, through compatible runtimes | Defines typed tensors and graph structure, but not a complete application-level schema | Avoids Pickle-style Python object execution, but model artifacts must still be trusted and validated |

### 4.3 Service Format Decision

The prediction service uses **ONNX for regression inference** because it supports interoperability across languages and compatible inference runtimes. The trusted Pickle artifact is retained for the fitted `DictVectorizer` required to prepare input features.

### 4.4 Security Note

Never load a Pickle artifact from an untrusted source: deserialization can execute arbitrary code. Model artifacts should come from a trusted training or build process.

---

## 5. FastAPI Prediction Service

The model is exposed through a FastAPI application with Pydantic-based request and response validation.

The API provides four endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Health check for the loaded model |
| `/metadata` | GET | Model and application metadata |
| `/predict` | POST | Single trip-duration prediction |
| `/predict/batch` | POST | Batch trip-duration predictions |

The application loads the model during startup rather than loading it separately for each request. This avoids repeated model-loading overhead during inference.

Pydantic schemas define the structure and validation of API inputs and outputs.

**Outcome:** The prediction API is implemented, and the public Docker image was subsequently tested against all four endpoints.

---

## 6. Automated Testing and Regression Validation

### 6.1 Objective

The objective of this stage was to establish an automated testing strategy for the productionized ML application.

Beyond checking whether the application executes successfully, the test suite validates expected behavior, detects incorrect outputs, and protects important components against regressions.

The testing strategy covers:

- Feature engineering and data transformations.
- Prediction logic and batch inference.
- API endpoints and request validation.
- Serialization parity between Pickle and ONNX.
- Serialization benchmark behavior.

The project uses `pytest` for automated testing, fixtures and mocks for dependency isolation, and `pytest-cov` to measure code coverage.

### 6.2 Test Suite Structure

| Test file | Main responsibility |
|---|---|
| `tests/conftest.py` | Shared fixtures and test configuration |
| `tests/test_api.py` | API endpoints, request validation, and response schemas |
| `tests/test_features.py` | Feature engineering and vectorization |
| `tests/test_predict.py` | Single and batch prediction behavior |
| `tests/test_serialization.py` | Pickle and ONNX prediction parity |
| `tests/test_benchmark_serialization.py` | Latency measurement and benchmark execution |

### 6.3 Coverage Configuration

A minimum coverage threshold of 70% is enforced directly in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov=src/prodml --cov-fail-under=70"
```

The following command executes the complete test suite and displays the coverage report:

```bash
uv run pytest -v --cov=src/prodml --cov-report=term-missing
```

The `--cov-fail-under=70` option ensures that the test command fails if total statement coverage falls below the required threshold.

### 6.4 Final Test Results

The final test execution produced the following result:

```text
25 passed, 1 warning
Required test coverage of 70% reached.
Total coverage: 72.46%
```

| Metric | Result |
|---|---:|
| Total tests collected | 25 |
| Passed tests | 25 |
| Failed tests | 0 |
| Coverage threshold | 70% |
| Achieved coverage | 72.46% |

The project successfully exceeded the required coverage threshold. One deprecation warning was reported by the Starlette TestClient integration; it did not affect test execution.

### 6.5 Module-Level Coverage Analysis

| Module | Statements | Missed | Coverage |
|---|---:|---:|---:|
| `__init__.py` | 0 | 0 | 100% |
| `api/__init__.py` | 0 | 0 | 100% |
| `api/main.py` | 96 | 22 | 77% |
| `api/schemas.py` | 15 | 0 | 100% |
| `benchmark_serialization.py` | 46 | 3 | 93% |
| `config.py` | 16 | 0 | 100% |
| `data.py` | 15 | 0 | 100% |
| `export.py` | 24 | 24 | 0% |
| `features.py` | 15 | 0 | 100% |
| `logging_conf.py` | 19 | 1 | 95% |
| `predict.py` | 58 | 12 | 79% |
| `train.py` | 30 | 30 | 0% |
| **Total** | **334** | **92** | **72.46%** |

The feature-engineering, configuration, data-loading, and API schema modules achieved 100% statement coverage.

The `train.py` and `export.py` modules remain uncovered and represent opportunities for further improvement. Their uncovered statements do not prevent the project from meeting the enforced 70% threshold.

### 6.6 Mutation Testing: Verifying Regression Detection

#### Why Mutation Testing?

Code coverage measures how much code is executed during testing, but it does not necessarily indicate whether assertions are strong enough to detect incorrect behavior.

For example, a test may execute a feature transformation without verifying that the resulting feature value is correct.

Mutation testing provides an additional way to evaluate test effectiveness by deliberately introducing a defect and checking whether the existing tests detect it.

#### Mutation Scenario

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

#### Regression Detection Results

The targeted feature tests were executed against the incorrect implementation:

```bash
uv run pytest -q tests/test_features.py --no-cov
```

Observed results:

| Result | Number |
|---|---:|
| Failed tests | 3 |
| Passed tests | 5 |
| Total tests | 8 |

The three parametrized cases failed because the actual generated feature did not match the expected value.

```text
Expected: 100_180
Actual:   100-180

Expected: 1_2
Actual:   1-2

Expected: 265_265
Actual:   265-265
```

This demonstrates that the existing tests validate the expected feature representation rather than simply checking that the function executes.

#### Restoration and Regression Verification

After the experiment, the original implementation using the underscore separator was restored.

The complete test suite was executed again:

```bash
uv run pytest -v --cov=src/prodml --cov-report=term-missing
```

The final execution confirmed:

```text
25 passed, 1 warning in 4.03s

Required test coverage of 70% reached.
Total coverage: 72.46%
```

All 25 tests passed after restoration, and the original feature-engineering behavior was preserved.

No deliberately incorrect implementation was retained in the production code.

#### Interpretation

This manual mutation-testing experiment demonstrates the distinction between code coverage and test effectiveness:

- **Code coverage:** Measures the proportion of source statements executed during testing.
- **Regression detection:** Determines whether tests identify incorrect behavior when the implementation changes.

Mutation testing provides an additional way to assess test effectiveness by introducing controlled defects and observing whether existing assertions detect them.

In this project, the incorrect `PU_DO` separator was detected by all three affected parametrized test cases. This provides concrete evidence that the feature-engineering tests protect an important part of the model's input representation.

### 6.7 Testing Acceptance Summary

| Acceptance criterion | Result |
|---|---|
| Automated test suite executes successfully | Passed |
| Minimum coverage of 70% | Passed — 72.46% |
| Feature engineering module statement coverage | Passed — 100% |
| Deliberate feature regression detected | Passed — 3 failures |
| Original implementation restored | Passed |
| Full regression suite passes after restoration | Passed — 25 tests |
| Coverage threshold enforced in configuration | Passed |

### 6.8 Conclusion

The testing stage established an automated validation strategy for the productionized ML application.

The project now includes tests for feature engineering, prediction logic, API behavior, serialization parity, and benchmark execution. The coverage threshold is enforced through the pytest configuration, and the final suite achieves 72.46% statement coverage.

The mutation experiment provides additional evidence that the tests can detect incorrect feature transformations rather than merely executing the code.

Together, these practices improve the reliability and maintainability of the application and establish a foundation for further productionization.

---

## 7. Docker Packaging and Image-Size Comparison

### 7.1 Multi-stage Docker Build

The API is packaged using a multi-stage Docker build based on `python:3.12-slim`.

Python 3.12 is intentional because the project requires Python `>=3.12,<3.14`.

The production image:

- Separates dependency installation from the runtime stage.
- Runs the application as the non-root user `appuser`.
- Loads the model at application startup.
- Includes a Docker health check against `/health`.
- Exposes port `8000`.

### 7.2 Single-stage vs Multi-stage Image Size

Both images were built using the same application, model, dependencies, and Python base image.

| Build | Image size |
|---|---:|
| Single-stage | 131,906,200 bytes (131.91 MB) |
| Multi-stage | 130,871,277 bytes (130.87 MB) |
| Difference | 1,034,923 bytes (approximately 1.03 MB) |

The multi-stage image is approximately **0.78% smaller** than the single-stage image.

The size reduction is modest for this project, but the multi-stage structure keeps the production runtime separate from the build environment.

### 7.3 `.dockerignore` Experiment

| Configuration | Image size |
|---|---:|
| With `.dockerignore` | 130,871,277 bytes |
| Without `.dockerignore` | 130,870,576 bytes |
| Difference | 701 bytes |

The final image sizes are nearly identical. This is expected because the Dockerfile uses targeted `COPY` instructions, so development files excluded by `.dockerignore` are not copied into the final image in either case.

The `.dockerignore` file still reduces the build context and helps prevent unnecessary files from being sent to the Docker build process or accidentally included by future changes.

### 7.4 Docker Hub Publication

The image was published to the public Docker Hub repository:

**Repository:** [`zouhour13/prodml-api`](https://hub.docker.com/r/zouhour13/prodml-api)

Published tags:

- `zouhour13/prodml-api:0.1.0`
- `zouhour13/prodml-api:latest`

The public image was pulled and tested independently, without cloning the Git repository or mounting the local model directory.

### 7.5 Public Image Acceptance Tests

| Acceptance check | Result |
|---|---|
| Public Docker Hub image pulled and launched without cloning | Passed |
| `/health` | HTTP 200 |
| `/metadata` | HTTP 200 |
| `/predict` | Successful prediction |
| `/predict/batch` | HTTP 200 |
| Container user | `appuser` |
| Docker health status | `healthy` |
| Model loading at startup | Confirmed in logs |

The single-prediction and batch-prediction endpoints both returned successful responses. Container inspection confirmed that the service ran as `appuser` and reached the `healthy` state. Application logs confirmed that the model was loaded during startup.

### 7.6 Docker Milestone Outcome

The multi-stage image was built, tested, and published to Docker Hub. The public image passed the endpoint and runtime acceptance checks, completing the Docker packaging and distribution milestone.

---

## 8. MLOps Maturity Self-Assessment

**Current level: Level 1 — Manual Process.**

The repository has evolved beyond notebooks into a reusable Python package with a tested prediction API, structured logging, model serialization, and a publicly published Docker image, but training and deployment are still manually triggered. To reach **Level 2 — ML Pipeline**, the project needs automated training, experiment tracking (e.g. MLflow), and basic CI; these are the main objectives of Module 2.

### Maturity Model Reference

| Level | Name | Main characteristics |
|---|---|---|
| Level 0 | No MLOps | Notebooks only; manual processes and no reproducibility |
| Level 1 | Manual Process | Scripts exist, but deployments are manual and there is no experiment tracking |
| Level 2 | ML Pipeline | Automated training, experiment tracking, and basic CI |
| Level 3 | CD for ML | Automatic retraining on triggers, model registry, and monitoring |
| Level 4 | Full MLOps | Zero-touch pipelines, automatic rollback, and full observability |

The repository has moved beyond the notebook-only stage, but it does not yet meet the Level 2 requirements. The next module addresses this gap.

---

## 9. Module 1 — Overall Progress and Remaining Deliverables

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
| Coverage enforcement | 72.46% — passed |
| Mutation experiment | Regression detected |
| Pre-commit quality checks | Passed |
| Latest branch commit | `1505217` — Multi-stage Docker build |
| Multi-stage Docker build | Completed |
| Single-stage vs multi-stage image comparison | Completed — 0.78% reduction |
| `.dockerignore` image-size comparison | Completed — 701-byte difference |
| Docker Hub publication | Completed — public image published and pulled |
| Public image acceptance tests | Passed — 8 checks documented |
| README rewrite | Completed — quick start and repository structure verified |
| Maturity self-assessment | Completed — Level 1 |
| Definition of Done review | Pending |
| Pull request and `v0.1.0` release | Pending |

---
## 10. Definition of Done

This checklist summarizes the acceptance criteria defined in the *MLOps Practitioner — Mini Projects & Final Project Handbook* (ITI × MLOps MENA Community).

Each criterion is classified according to the available evidence and its current completion status.

| # | Acceptance criterion | Status and evidence |
|---|---|---|
| 1 | Public GitHub repository with more than one commit | **Verified** — the repository is public and contains multiple commits. |
| 2 | Docker Hub access token stored securely | **Verified** — Docker Hub authentication uses a personal access token stored through the Docker Desktop credential helper (`desktop.exe`). |
| 3 | `docker run hello-world` executes successfully | **Verified** — Docker installation smoke test. |
| 4 | WSL2 runs with version 2 on Windows | **Verified** — confirm the WSL distribution version. |
| 5 | `pip install -e .` succeeds in a clean virtual environment | **Verified** — test the installation in a fresh environment. |
| 6 | Linting passes and pre-commit hooks are installed | **Verified** — Ruff, formatting, and pre-commit checks passed. |
| 7 | Automated tests pass with at least 70% coverage | **Verified** — 25 tests passed with 72.46% total coverage. |
| 8 | No `print()` statements remain in `src/` | **Verified** — inspect the source code. |
| 9 | `/health`, `/metadata`, `/predict`, and `/predict/batch` respond correctly | **Verified** — all four endpoints were tested against the public Docker image. |
| 10 | ONNX predictions match the reference predictions | **Verified** — the serialization parity test passed with an absolute tolerance of `1e-4`. |
| 11 | Docker image is public, pullable by another user, and runs as a non-root user | **Verified** — the public image was pulled and tested independently; the container ran as `appuser`. |
| 12 | README enables a new user to obtain a prediction in three commands | **Verified** — the public Docker image was started and the health and prediction endpoints were tested successfully. |
| 13 | `reports/module-1.md` documents MAE, latency comparison, image-size comparison, serialization comparison, and MLOps maturity self-assessment | **Verified** — all required topics are included in this report. |
| 14 | Pull request is opened, reviewed by two peers, and merged | **Pending** — the pull request and peer reviews remain to be completed. |
| 15 | Version `v0.1.0` is tagged | **Pending** — create the release tag after the pull request is reviewed and merged. |

### Completion Status

The core technical implementation and acceptance checks for Module 1 have been completed and verified, including packaging, automated testing, API validation, serialization parity, and Docker deployment.

The remaining work is the collaboration and release workflow: open the pull request, obtain two peer reviews, address any feedback, merge the approved pull request, and create the `v0.1.0` release tag.

Module 1 will be considered fully complete once these remaining deliverables have been fulfilled.

---

## 11. Conclusion and Next Steps

Module 1 transformed the initial NYC Green Taxi regression notebook into a reusable Python package and a publicly accessible, containerized prediction service.

The project now includes a baseline model, a modular application structure, structured logging, a FastAPI interface, serialization benchmarks, and automated tests with an enforced coverage threshold. The manual mutation-testing experiment also demonstrated that the feature tests can detect an intentionally introduced regression.

The application was packaged using a multi-stage Docker build, configured to run as a non-root user, and published to Docker Hub. The public image was independently pulled and validated through its health, metadata, single-prediction, and batch-prediction endpoints.

### MLOps Maturity Self-Assessment

Based on the assessment presented in Section 8, I classify this project at **Level 1 — Manual Process**.

Although the project has a reusable package structure, automated tests, structured logging, a prediction API, and containerized deployment, important parts of the ML lifecycle remain manual.

To progress toward **Level 2 — ML Pipeline**, the next objectives are:

- Automate the model training pipeline.
- Introduce experiment tracking.
- Implement basic Continuous Integration (CI) workflows.

### Next Steps

The technical acceptance criteria for Module 1 have been verified. The remaining steps are to complete the collaboration and release workflow:

1. Open the Module 1 pull request against `main`.
2. Obtain reviews from two peers and address their feedback.
3. Merge the pull request after approval.
4. Create the `v0.1.0` release tag.

Module 2 will build on this foundation by introducing automated ML pipelines and experiment-tracking capabilities, moving the project toward a more reproducible and mature MLOps workflow.
