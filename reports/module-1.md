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
