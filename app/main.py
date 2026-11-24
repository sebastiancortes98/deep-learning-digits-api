from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, status

from app.inference import DigitClassifier, ModelNotReadyError
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
)

classifier = DigitClassifier()


@asynccontextmanager
async def lifespan(_: FastAPI):
    classifier.load()
    yield


app = FastAPI(
    title="Optimized Digit Classifier API",
    version="1.0.0",
    description="Real-time ONNX INT8 inference for 8 x 8 handwritten digits.",
    lifespan=lifespan,
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready", tags=["operations"])
def ready() -> dict[str, str]:
    if not classifier.ready:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not ready")
    return {"status": "ready", "model_version": classifier.version}


def _to_prediction(probabilities: np.ndarray) -> Prediction:
    digit = int(np.argmax(probabilities))
    return Prediction(
        digit=digit,
        confidence=float(probabilities[digit]),
        probabilities=[float(value) for value in probabilities],
    )


@app.post("/v1/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        probabilities, elapsed_ms = classifier.predict(np.asarray([payload.pixels], dtype=np.float32))
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return PredictResponse(
        model_version=classifier.version,
        inference_ms=round(elapsed_ms, 4),
        prediction=_to_prediction(probabilities[0]),
    )


@app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    matrix = np.asarray([item.pixels for item in payload.instances], dtype=np.float32)
    try:
        probabilities, elapsed_ms = classifier.predict(matrix)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return BatchPredictResponse(
        model_version=classifier.version,
        inference_ms=round(elapsed_ms, 4),
        predictions=[_to_prediction(row) for row in probabilities],
    )

