from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

from ml.pipeline import load_split_data, onnx_predict


ROOT = Path(__file__).resolve().parents[1]


def test_quantized_model_meets_quality_gate() -> None:
    data = load_split_data()
    predictions = onnx_predict(ROOT / "models" / "digits_mlp_int8.onnx", data.x_test)
    assert accuracy_score(data.y_test, predictions) >= 0.95


def test_quantization_prediction_drift_is_bounded() -> None:
    data = load_split_data()
    fp32 = onnx_predict(ROOT / "models" / "digits_mlp_fp32.onnx", data.x_test)
    int8 = onnx_predict(ROOT / "models" / "digits_mlp_int8.onnx", data.x_test)
    disagreement = float(np.mean(fp32 != int8))
    assert disagreement <= 0.02

