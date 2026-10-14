"""Benchmark pickle and ONNX inference on the same validation batch."""

import pickle
import time
from collections.abc import Callable

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features


def measure_latency(
    predict: Callable[[], object],
    repetitions: int = 100,
) -> tuple[float, float]:
    """Return mean and p95 inference latency in milliseconds per batch."""
    predict()  # Warm up the runtime.

    latencies = []
    for _ in range(repetitions):
        start = time.perf_counter()
        predict()
        latencies.append((time.perf_counter() - start) * 1000)

    return float(np.mean(latencies)), float(np.percentile(latencies, 95))


def main() -> None:
    with settings.model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    df = load_data()
    _, X_val, _, _ = split_data(df)
    X_val = create_features(X_val).head(500)

    if len(X_val) != 500:
        raise ValueError(f"Expected 500 validation rows, got {len(X_val)}")

    vectorized = artifact["dv"].transform(X_val.to_dict(orient="records"))
    dense_features = vectorized.toarray().astype(np.float32)

    onnx_path = settings.model_path.with_suffix(".onnx")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    def predict_pickle() -> np.ndarray:
        return artifact["model"].predict(vectorized)

    def predict_onnx() -> np.ndarray:
        return session.run(None, {input_name: dense_features})[0].ravel()

    if not np.allclose(predict_pickle(), predict_onnx(), atol=1e-4):
        raise ValueError("Pickle and ONNX predictions do not match")

    for name, predict in (
        ("Pickle / scikit-learn", predict_pickle),
        ("ONNX Runtime", predict_onnx),
    ):
        mean_ms, p95_ms = measure_latency(predict)
        print(f"{name}: mean={mean_ms:.4f} ms/batch, p95={p95_ms:.4f} ms/batch")


if __name__ == "__main__":
    main()
