"""Shared paths and artifact filenames."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_PATH = DATA_DIR / "emails.csv"
SAMPLE_MBOX = DATA_DIR / "sample_inbox.mbox"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

MODEL_FILENAME = "model.joblib"
VECTORIZER_FILENAME = "vectorizer.joblib"
METRICS_FILENAME = "metrics.json"
METRICS_CSV_FILENAME = "metrics.csv"

LABELS = ("ham", "spam")
POSITIVE_LABEL = "spam"
