import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MODEL_PATH", str(PROJECT_ROOT / "models" / "digits_mlp_int8.onnx"))

