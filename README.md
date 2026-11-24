# Optimized Deep Learning Inference API

This project trains a multilayer neural network on the handwritten digits dataset, exports it to ONNX, applies dynamic INT8 quantization, benchmarks FP32 and INT8 inference, and serves the optimized artifact through FastAPI.

## Reproduce locally

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-train.txt
.venv/bin/python scripts/train.py
.venv/bin/python scripts/benchmark.py
.venv/bin/python scripts/smoke_test.py
.venv/bin/pytest -q
.venv/bin/python scripts/check_gates.py
```

On Windows PowerShell, replace `.venv/bin/python` with `.venv\\Scripts\\python.exe`.

## Run the API

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the interactive OpenAPI interface.

Example request:

```json
{
  "pixels": [0, 0, 5, 13, 9, 1, 0, 0, 0, 0, 13, 15, 10, 15, 5, 0, 0, 3, 15, 2, 0, 11, 8, 0, 0, 4, 12, 0, 0, 8, 8, 0, 0, 5, 8, 0, 0, 9, 8, 0, 0, 4, 11, 0, 1, 12, 7, 0, 0, 2, 14, 5, 10, 12, 0, 0, 0, 0, 6, 13, 10, 0, 0, 0]
}
```

## Run with Docker

```bash
docker compose up --build
curl http://127.0.0.1:8000/ready
```

The runtime container executes as a non-root user, has a health check, and contains only the INT8 model and inference dependencies.

## Evidence

- `results/training_metrics.json` records dataset splits, validation and test metrics, artifact sizes, and SHA-256 checksums.
- `results/benchmark.json` records the environment and p50/p95 latency for batches 1, 16, and 64.
- `models/MODEL_VERSION` binds the deployed version to the quantized artifact checksum.
- `dvc.yaml` describes reproducible training and benchmarking stages.
- `.github/workflows/ci.yml` reproduces, tests, benchmarks, packages, and smoke-tests the container.
