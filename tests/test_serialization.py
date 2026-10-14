"""Parity tests for pickle and ONNX model artifacts."""

import pickle

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import create_features


def test_pickle_and_onnx_predictions_match() -> None:
    """The two model formats should agree on 500 validation examples."""
    with settings.model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    df = load_data()
    _, X_val, _, _ = split_data(df)
    X_val = create_features(X_val).head(500)

    feature_dicts = X_val.to_dict(orient="records")
    vectorized = artifact["dv"].transform(feature_dicts)

    predictions_pickle = artifact["model"].predict(vectorized)

    onnx_path = settings.model_path.with_suffix(".onnx")
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    predictions_onnx = session.run(
        None,
        {input_name: vectorized.toarray().astype(np.float32)},
    )[0].ravel()

    assert len(predictions_pickle) == 500
    assert np.allclose(
        predictions_pickle,
        predictions_onnx,
        atol=1e-4,
    ), (
        "Pickle and ONNX predictions differ. "
        f"Maximum absolute difference: "
        f"{np.max(np.abs(predictions_pickle - predictions_onnx))}"
    )
