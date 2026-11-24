import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.pipeline import train_and_export


if __name__ == "__main__":
    result = train_and_export(PROJECT_ROOT)
    print(f"Created {result['model_version']}")

