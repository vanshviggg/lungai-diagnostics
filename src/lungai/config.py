from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2
RANDOM_SEED = 42

DISEASE_LABELS = [
    "Cardiomegaly",
    "Pneumonia",
    "Pneumothorax",
]
