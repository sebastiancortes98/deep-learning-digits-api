from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort

from app.config import DEFAULT_MODEL_PATH, MODEL_VERSION_PATH


class ModelNotReadyError(RuntimeError):
    pass


class DigitClassifier:
    def __init__(self, model_path: Path | None = None) -> None:
        configured = os.getenv("MODEL_PATH")
        self.model_path = Path(configured) if configured else (model_path or DEFAULT_MODEL_PATH)
        self.session: ort.InferenceSession | None = None
        self.input_name = "input"
        self.output_name = "probabilities"
        self.version = "unknown"

    @property
    def ready(self) -> bool:
        return self.session is not None

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotReadyError(f"Model artifact not found: {self.model_path}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("ORT_INTRA_OP_THREADS", "1")))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[1].name
        if MODEL_VERSION_PATH.exists():
            self.version = MODEL_VERSION_PATH.read_text(encoding="utf-8").strip()

    def predict(self, raw_pixels: np.ndarray) -> tuple[np.ndarray, float]:
        if self.session is None:
            raise ModelNotReadyError("Model session is not loaded")
        pixels = np.asarray(raw_pixels, dtype=np.float32)
        if pixels.ndim != 2 or pixels.shape[1] != 64:
            raise ValueError("Expected a two-dimensional array with 64 features")
        normalized = pixels / np.float32(16.0)
        started = perf_counter()
        outputs = self.session.run(None, {self.input_name: normalized})
        elapsed_ms = (perf_counter() - started) * 1000.0
        probabilities = np.asarray(outputs[1], dtype=np.float32)
        return probabilities, elapsed_ms

