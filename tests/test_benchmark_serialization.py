"""Unit tests for serialization benchmarking."""

from unittest.mock import Mock, mock_open
import pytest
from pathlib import Path
from prodml import benchmark_serialization
import numpy as np
import pandas as pd


def test_measure_latency_returns_mean_and_p95(monkeypatch) -> None:
    """Measure repeated calls and return latency statistics in milliseconds."""
    predict = Mock(return_value=[1.0])

    # Simulate three measured durations: 1 ms, 3 ms, and 2 ms.
    timestamps = iter([0.0, 0.001, 1.0, 1.003, 2.0, 2.002])
    monkeypatch.setattr(
        benchmark_serialization.time,
        "perf_counter",
        lambda: next(timestamps),
    )

    mean_ms, p95_ms = benchmark_serialization.measure_latency(
        predict,
        repetitions=3,
    )

    assert mean_ms == pytest.approx(2.0)
    assert p95_ms == pytest.approx(2.9)
    assert predict.call_count == 4  # One warm-up + three measurements.


def test_measure_latency_rejects_zero_repetitions() -> None:
    """Reject a benchmark that cannot produce latency measurements."""
    predict = Mock(return_value=[1.0])

    with pytest.raises(
        ValueError,
        match="repetitions must be at least 1",
    ):
        benchmark_serialization.measure_latency(
            predict,
            repetitions=0,
        )

    predict.assert_not_called()


def test_main_runs_pickle_and_onnx_benchmarks(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    """Run main with mocked data, model artifacts, and ONNX Runtime."""
    model_path = tmp_path / "model.pkl"
    fake_file = mock_open(read_data=b"fake model")

    artifact = {
        "dv": Mock(),
        "model": Mock(),
    }

    vectorized = Mock()
    vectorized.toarray.return_value = np.ones((500, 2), dtype=np.float32)

    artifact["dv"].transform.return_value = vectorized
    artifact["model"].predict.return_value = np.ones(500)

    session = Mock()
    session.get_inputs.return_value = [Mock(name="features")]
    session.run.return_value = [np.ones(500)]

    monkeypatch.setattr(
        benchmark_serialization,
        "settings",
        Mock(model_path=model_path),
    )
    monkeypatch.setattr(
        benchmark_serialization.pickle,
        "load",
        Mock(return_value=artifact),
    )
    monkeypatch.setattr(
        benchmark_serialization,
        "load_data",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        benchmark_serialization,
        "split_data",
        Mock(return_value=(None, pd.DataFrame(index=range(500)), None, None)),
    )
    monkeypatch.setattr(
        benchmark_serialization,
        "create_features",
        Mock(side_effect=lambda data: data),
    )
    monkeypatch.setattr(
        benchmark_serialization.ort,
        "InferenceSession",
        Mock(return_value=session),
    )
    monkeypatch.setattr(
        benchmark_serialization,
        "measure_latency",
        Mock(return_value=(1.5, 2.0)),
    )

    monkeypatch.setattr(
        Path,
        "open",
        lambda self, *args, **kwargs: fake_file(),
    )

    benchmark_serialization.main()

    output = capsys.readouterr().out
    assert "Pickle / scikit-learn" in output
    assert "ONNX Runtime" in output
    assert "mean=1.5000 ms/batch" in output
    assert "p95=2.0000 ms/batch" in output
