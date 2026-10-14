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

## Module 04 — Model Serialization

### Pickle vs ONNX: inference benchmark

Both formats were evaluated on the same 500 validation examples. The benchmark performed 100 timed batch-inference repetitions after a warm-up. Feature vectorization and ONNX dense-array conversion were performed before timing, so preprocessing is excluded.

| Metric | Pickle / scikit-learn | ONNX Runtime |
|---|---:|---:|
| Validation batch size | 500 rows | 500 rows |
| Mean latency (ms/batch) | 0.1325 | 1.3905 |
| p95 latency (ms/batch) | 0.1719 | 2.6175 |

The parity test passed with an absolute tolerance of 1e-4. In this local inference-only benchmark, Pickle/scikit-learn was faster than ONNX Runtime. These measurements describe this environment and workload; they do not establish that one format is universally faster. In particular, preprocessing costs are excluded, and ONNX receives dense input while scikit-learn uses sparse input.

### Serialization format comparison

| Format | Human-readable | Cross-language | Schema-enforced | Safe to load from an untrusted source |
|---|---|---|---|---|
| JSON | Yes | Yes | No built-in schema; JSON Schema can provide validation | Generally safe to parse, but validate the data |
| Protobuf | No, normally binary (a text format also exists) | Yes | Yes, using a `.proto` schema | Generally safer than executable object formats; validate input |
| Pickle | No | No, Python-specific | No | **No** — loading can execute arbitrary code |
| ONNX | No, normally a binary model graph | Yes, through compatible runtimes | Partially — defines typed tensors and graph structure, not a complete application-level schema | Safer than Pickle against Python-code execution, but only use trusted, validated model artifacts |

### Service format decision

The prediction service will use **ONNX for regression inference** because it supports interoperable inference across languages and runtimes; the trusted Pickle artifact is retained for the fitted `DictVectorizer` needed to prepare input features.

### Security note

Never load a Pickle artifact from an untrusted source: deserialization can execute arbitrary code. Model artifacts should come from a trusted build or training process.
