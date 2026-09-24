"""Load the saved model and score email text."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.config import (
    METRICS_FILENAME,
    MODEL_FILENAME,
    OUTPUT_DIR,
    VECTORIZER_FILENAME,
)


class EmailClassifier:
    """Spam versus ham scorer backed by a fitted estimator and vectorizer."""

    def __init__(self, model, vectorizer, metadata: dict, output_dir: Path):
        self.model = model
        self.vectorizer = vectorizer
        self.metadata = metadata
        self.output_dir = Path(output_dir)

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("best_model") or "unknown")

    def predict_many(self, texts: list[str]) -> list[dict]:
        """Score each message. Every item needs non-empty text."""
        if not texts:
            return []
        cleaned: list[str] = []
        for text in texts:
            if text is None or not str(text).strip():
                raise ValueError("Email text is empty.")
            cleaned.append(str(text))

        features = self.vectorizer.transform(cleaned)
        labels = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        classes = [str(label) for label in self.model.classes_]
        results: list[dict] = []
        for label, scores in zip(labels, probabilities):
            prob_map = {name: float(score) for name, score in zip(classes, scores)}
            label_name = str(label)
            results.append(
                {
                    "label": label_name,
                    "confidence": float(prob_map[label_name]),
                    "probabilities": prob_map,
                }
            )
        return results

    def predict_text(self, text: str) -> dict:
        """Return label, confidence, and per-class probabilities."""
        return self.predict_many([text])[0]


def load_classifier(output_dir: Path | None = None) -> EmailClassifier:
    """Load ``model.joblib`` and ``vectorizer.joblib``.

    Raises ``FileNotFoundError`` with a train-first hint when artifacts are missing.
    """
    directory = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    model_path = directory / MODEL_FILENAME
    vectorizer_path = directory / VECTORIZER_FILENAME
    missing = [path.name for path in (model_path, vectorizer_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained artifact(s) in {directory}: {', '.join(missing)}. "
            "Train a model first with: python -m src.train"
        )

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load artifacts from {directory}. Retrain with: python -m src.train"
        ) from exc

    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise RuntimeError(
            f"Saved model in {model_path} does not support predict_proba. "
            "Retrain with: python -m src.train"
        )
    if not hasattr(vectorizer, "transform"):
        raise RuntimeError(
            f"Saved vectorizer in {vectorizer_path} is unusable. "
            "Retrain with: python -m src.train"
        )

    metadata: dict = {}
    metrics_path = directory / METRICS_FILENAME
    if metrics_path.is_file():
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    return EmailClassifier(
        model=model,
        vectorizer=vectorizer,
        metadata=metadata,
        output_dir=directory,
    )
