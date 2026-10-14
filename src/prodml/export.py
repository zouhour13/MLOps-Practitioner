"""Export the trained regression estimator to ONNX."""

import logging
import pickle
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings
from prodml.logging_conf import configure_logging

logger = logging.getLogger(__name__)


def export_model(output_path: Path | None = None) -> Path:
    """Export the fitted regression estimator with a dynamic batch dimension."""
    with settings.model_path.open("rb") as model_file:
        artifact = pickle.load(model_file)

    dv = artifact["dv"]
    model = artifact["model"]
    feature_count = len(dv.feature_names_)

    initial_types = [
        ("features", FloatTensorType([None, feature_count])),
    ]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_types,
        target_opset=17,
    )

    destination = output_path or settings.model_path.with_suffix(".onnx")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(onnx_model.SerializeToString())

    logger.info(
        "ONNX model exported to %s with %d input features",
        destination,
        feature_count,
    )
    return destination


if __name__ == "__main__":
    configure_logging()
    export_model()
