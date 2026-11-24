from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray


def load_params(project_root: Path | None = None) -> dict[str, object]:
    root = project_root or PROJECT_ROOT
    with (root / "params.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_split_data(project_root: Path | None = None) -> SplitData:
    params = load_params(project_root)
    seed = int(params["seed"])
    validation_share = float(params["split"]["validation"])
    test_share = float(params["split"]["test"])
    holdout_share = validation_share + test_share
    dataset = load_digits()
    features = dataset.data.astype(np.float32) / np.float32(16.0)
    labels = dataset.target.astype(np.int64)
    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        labels,
        test_size=holdout_share,
        random_state=seed,
        stratify=labels,
    )
    relative_test_share = test_share / holdout_share
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_share,
        random_state=seed,
        stratify=y_temp,
    )
    return SplitData(x_train, x_validation, x_test, y_train, y_validation, y_test)


def build_model(params: dict[str, object]) -> MLPClassifier:
    model_params = params["model"]
    seed = int(params["seed"])
    return MLPClassifier(
        hidden_layer_sizes=tuple(int(value) for value in model_params["hidden_layers"]),
        activation=str(model_params["activation"]),
        solver="adam",
        alpha=float(model_params["alpha"]),
        batch_size=int(model_params["batch_size"]),
        learning_rate_init=float(model_params["learning_rate_init"]),
        max_iter=int(model_params["max_iter"]),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
        verbose=False,
    )


def evaluate(model: MLPClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro")),
    }


def export_onnx(model: MLPClassifier, output_path: Path, opset: int) -> None:
    initial_types = [("input", FloatTensorType([None, 64]))]
    options = {id(model): {"zipmap": False}}
    exported = convert_sklearn(
        model,
        initial_types=initial_types,
        options=options,
        target_opset=opset,
    )
    exported.graph.name = "digits_mlp_v1"
    onnx.save_model(exported, output_path)


def quantize_onnx(source: Path, destination: Path) -> None:
    preprocessed = source.with_name(f"{source.stem}_preprocessed.onnx")
    try:
        quant_pre_process(
            input_model=source,
            output_model_path=preprocessed,
            skip_optimization=False,
            skip_onnx_shape=False,
            skip_symbolic_shape=True,
        )
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(destination),
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    finally:
        preprocessed.unlink(missing_ok=True)


def onnx_predict(model_path: Path, features: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features.astype(np.float32)})
    return np.asarray(outputs[0], dtype=np.int64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_and_export(project_root: Path) -> dict[str, object]:
    model_dir = project_root / "models"
    results_dir = project_root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(project_root)
    data = load_split_data(project_root)
    model = build_model(params)
    model.fit(data.x_train, data.y_train)

    joblib_path = model_dir / "digits_mlp.joblib"
    fp32_path = model_dir / "digits_mlp_fp32.onnx"
    int8_path = model_dir / "digits_mlp_int8.onnx"
    joblib.dump(model, joblib_path, compress=0)
    export_onnx(model, fp32_path, int(params["optimization"]["onnx_opset"]))
    quantize_onnx(fp32_path, int8_path)

    fp32_predictions = onnx_predict(fp32_path, data.x_test)
    int8_predictions = onnx_predict(int8_path, data.x_test)
    version = f"digits-mlp-v1-{sha256_file(int8_path)[:12]}"
    (model_dir / "MODEL_VERSION").write_text(version + "\n", encoding="utf-8")

    metrics: dict[str, object] = {
        "model_version": version,
        "random_seed": int(params["seed"]),
        "dataset": {
            "name": "sklearn.datasets.load_digits",
            "samples": int(sum(map(len, [data.x_train, data.x_validation, data.x_test]))),
            "features": 64,
            "classes": 10,
            "split": {
                "train": int(len(data.x_train)),
                "validation": int(len(data.x_validation)),
                "test": int(len(data.x_test)),
            },
        },
        "architecture": [64, 128, 64, 32, 10],
        "training": {
            "iterations": int(model.n_iter_),
            "loss": float(model.loss_),
            "validation": evaluate(model, data.x_validation, data.y_validation),
            "test_sklearn": evaluate(model, data.x_test, data.y_test),
        },
        "onnx_fp32": {
            "accuracy": float(accuracy_score(data.y_test, fp32_predictions)),
            "f1_macro": float(f1_score(data.y_test, fp32_predictions, average="macro")),
        },
        "onnx_int8": {
            "accuracy": float(accuracy_score(data.y_test, int8_predictions)),
            "f1_macro": float(f1_score(data.y_test, int8_predictions, average="macro")),
        },
        "artifacts": {
            path.name: {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (joblib_path, fp32_path, int8_path)
        },
    }
    metrics_path = results_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

