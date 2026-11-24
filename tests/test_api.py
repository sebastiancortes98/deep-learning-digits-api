from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from app.main import app


def test_health_and_ready() -> None:
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_single_prediction() -> None:
    dataset = load_digits()
    with TestClient(app) as client:
        response = client.post("/v1/predict", json={"pixels": dataset.data[0].tolist()})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"]["digit"] == int(dataset.target[0])
    assert len(body["prediction"]["probabilities"]) == 10
    assert body["inference_ms"] >= 0


def test_batch_prediction_and_validation() -> None:
    dataset = load_digits()
    instances = [{"pixels": row.tolist()} for row in dataset.data[:4]]
    with TestClient(app) as client:
        response = client.post("/v1/predict/batch", json={"instances": instances})
        invalid = client.post("/v1/predict", json={"pixels": [0.0] * 63})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 4
    assert invalid.status_code == 422

