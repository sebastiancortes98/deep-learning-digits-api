from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "digits_mlp_int8.onnx"
MODEL_VERSION_PATH = MODEL_DIR / "MODEL_VERSION"

