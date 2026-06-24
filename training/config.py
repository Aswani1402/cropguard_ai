from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "kaggle"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}