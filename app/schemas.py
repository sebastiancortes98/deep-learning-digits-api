from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

Pixel = Annotated[float, Field(ge=0.0, le=16.0)]


class PredictRequest(BaseModel):
    """One flattened 8 x 8 grayscale image from the sklearn digits scale."""

    model_config = ConfigDict(extra="forbid")
    pixels: list[Pixel] = Field(min_length=64, max_length=64)


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PredictRequest] = Field(min_length=1, max_length=256)


class Prediction(BaseModel):
    digit: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: list[float] = Field(min_length=10, max_length=10)


class PredictResponse(BaseModel):
    model_version: str
    inference_ms: float = Field(ge=0.0)
    prediction: Prediction


class BatchPredictResponse(BaseModel):
    model_version: str
    inference_ms: float = Field(ge=0.0)
    predictions: list[Prediction]

