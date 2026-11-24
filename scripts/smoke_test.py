from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx
from sklearn.datasets import load_digits

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    port = 8013
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(trust_env=False, timeout=5.0) as client:
            ready = None
            for _ in range(30):
                try:
                    candidate = client.get(f"http://127.0.0.1:{port}/ready")
                    if candidate.status_code == 200:
                        ready = candidate.json()
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            if ready is None:
                raise RuntimeError("API did not become ready")
            features, labels = load_digits(return_X_y=True)
            started = time.perf_counter()
            response = client.post(
                f"http://127.0.0.1:{port}/v1/predict",
                json={"pixels": features[0].tolist()},
            )
            end_to_end_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            payload = response.json()
            evidence = {
                "ready": ready,
                "expected_digit": int(labels[0]),
                "predicted_digit": payload["prediction"]["digit"],
                "confidence": payload["prediction"]["confidence"],
                "reported_inference_ms": payload["inference_ms"],
                "client_end_to_end_ms": end_to_end_ms,
            }
            if evidence["predicted_digit"] != evidence["expected_digit"]:
                raise RuntimeError("Smoke-test prediction mismatch")
            destination = PROJECT_ROOT / "results" / "smoke_test.json"
            destination.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
            print(destination)
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
