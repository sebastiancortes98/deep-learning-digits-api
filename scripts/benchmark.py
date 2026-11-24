from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import load_split_data


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def benchmark(session: ort.InferenceSession, source: np.ndarray, batch_size: int) -> dict[str, float]:
    input_name = session.get_inputs()[0].name
    batches = [
        np.resize(source[index % len(source)], (batch_size, source.shape[1])).astype(np.float32)
        for index in range(120)
    ]
    for batch in batches[:20]:
        session.run(None, {input_name: batch})
    latencies_ms: list[float] = []
    for batch in batches[20:]:
        started = time.perf_counter_ns()
        session.run(None, {input_name: batch})
        latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    ordered = sorted(latencies_ms)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    median = statistics.median(latencies_ms)
    throughput = (batch_size * 1000.0) / median
    return {
        "batch_size": batch_size,
        "runs": len(latencies_ms),
        "latency_p50_ms": median,
        "latency_p95_ms": p95,
        "throughput_items_per_second_at_p50": throughput,
    }


def measure_session_rss_delta_kib(path: Path) -> int:
    code = """
import json
import sys
import numpy as np
import onnxruntime as ort
def rss_kib():
    with open('/proc/self/status', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return 0
before = rss_kib()
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(sys.argv[1], sess_options=options, providers=['CPUExecutionProvider'])
name = session.get_inputs()[0].name
session.run(None, {name: np.zeros((1, 64), dtype=np.float32)})
after = rss_kib()
print(json.dumps({'delta_kib': max(0, int(after - before))}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(completed.stdout)["delta_kib"])


def main() -> None:
    data = load_split_data()
    models = {
        "onnx_fp32": PROJECT_ROOT / "models" / "digits_mlp_fp32.onnx",
        "onnx_int8": PROJECT_ROOT / "models" / "digits_mlp_int8.onnx",
    }
    results: dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or "not reported",
            "onnxruntime": ort.__version__,
            "threads_per_session": 1,
        },
        "method": {
            "warmup_runs": 20,
            "measured_runs": 100,
            "batch_sizes": [1, 16, 64],
        },
        "models": {},
    }
    for name, path in models.items():
        session = create_session(path)
        results["models"][name] = {
            "artifact_bytes": path.stat().st_size,
            "session_peak_rss_delta_kib": measure_session_rss_delta_kib(path),
            "batches": [benchmark(session, data.x_test, size) for size in (1, 16, 64)],
        }
    destination = PROJECT_ROOT / "results" / "benchmark.json"
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

