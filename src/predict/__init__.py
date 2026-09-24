"""Inference: load saved artifacts and score email text or mbox files."""

from src.predict.mbox_io import classify_mbox, read_mbox
from src.predict.service import EmailClassifier, load_classifier

__all__ = [
    "EmailClassifier",
    "classify_mbox",
    "load_classifier",
    "read_mbox",
]
