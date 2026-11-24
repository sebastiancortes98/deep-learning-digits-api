import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
training = json.loads((ROOT / "results" / "training_metrics.json").read_text(encoding="utf-8"))
benchmark = json.loads((ROOT / "results" / "benchmark.json").read_text(encoding="utf-8"))

accuracy = training["onnx_int8"]["accuracy"]
p95_batch_one = benchmark["models"]["onnx_int8"]["batches"][0]["latency_p95_ms"]

if accuracy < 0.95:
    raise SystemExit(f"Quality gate failed: INT8 accuracy {accuracy:.4f} < 0.95")
if p95_batch_one > 10.0:
    raise SystemExit(f"Latency gate failed: INT8 p95 {p95_batch_one:.4f} ms > 10 ms")

print(f"Gates passed: accuracy={accuracy:.4f}, p95_batch1={p95_batch_one:.4f} ms")
