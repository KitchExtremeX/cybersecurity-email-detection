"""Fit four classifiers, keep the best spam-F1 model, and write artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import (
    DATA_PATH,
    LOG_DIR,
    METRICS_CSV_FILENAME,
    METRICS_FILENAME,
    MODEL_FILENAME,
    OUTPUT_DIR,
    POSITIVE_LABEL,
    ROOT,
    VECTORIZER_FILENAME,
)
from src.logging_utils import configure_logging
from src.train.dataset import load_dataset
from src.train.features import build_vectorizer
from src.train.metrics import compute_metrics, format_metrics_table, rank_models, select_best
from src.train.models import build_models
from src.train.split import split_frame

LOGGER_NAME = "email_detection.train"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def run_training(
    data_path: Path,
    output_dir: Path,
    log_file: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train, compare, and save the best model plus the TF-IDF vectorizer."""
    logger = configure_logging(LOGGER_NAME, log_file, mode="w")

    logger.info("Training started")
    logger.info("Data: %s", _display_path(Path(data_path)))
    frame = load_dataset(data_path)
    counts = frame["label"].value_counts().to_dict()
    logger.info("Loaded %s rows. Label counts: %s", len(frame), counts)
    if len(frame) < 100:
        logger.warning(
            "Dataset has only %s rows. Metrics will be noisy; prefer hundreds of emails.",
            len(frame),
        )
    small = [label for label, count in counts.items() if count < 20]
    if small:
        raise ValueError(
            "Need at least 20 rows for each label before splitting. "
            f"Short labels: {small}."
        )

    train_frame, test_frame, split_info = split_frame(frame, test_size=test_size, seed=seed)
    x_train = train_frame["text"].tolist()
    y_train = train_frame["label"].tolist()
    x_test = test_frame["text"].tolist()
    y_test = test_frame["label"].tolist()
    logger.info(
        "Split: strategy=%s train=%s test=%s test_size=%s seed=%s",
        split_info["strategy"],
        len(x_train),
        len(x_test),
        test_size,
        seed,
    )
    if split_info["test_categories"]:
        logger.info("Held-out categories: %s", ", ".join(split_info["test_categories"]))

    vectorizer = build_vectorizer()
    logger.info("Fitting TF-IDF vectorizer (word unigrams and bigrams)")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    logger.info("Vocabulary size: %s", len(vectorizer.vocabulary_))

    rows: list[dict] = []
    for name, model in build_models(seed).items():
        logger.info("Training %s", name)
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        metrics = compute_metrics(name, y_test, predictions)
        rows.append(metrics)
        logger.info(
            "%s accuracy=%.4f precision_spam=%.4f recall_spam=%.4f f1_spam=%.4f f1_macro=%.4f",
            name,
            metrics["accuracy"],
            metrics["precision_spam"],
            metrics["recall_spam"],
            metrics["f1_spam"],
            metrics["f1_macro"],
        )

    ranked = rank_models(rows)
    best = select_best(rows)
    best_name = best["model"]
    table = format_metrics_table(rows, best_name)
    logger.info("Metric table:\n%s", table)

    margin = 0.0
    runner_up = None
    if len(ranked) > 1:
        runner_up = ranked[1]["model"]
        margin = round(best["f1_spam"] - ranked[1]["f1_spam"], 6)
    logger.info(
        "Best model: %s (spam F1=%.4f, runner-up=%s, margin=%.4f)",
        best_name,
        best["f1_spam"],
        runner_up,
        margin,
    )
    logger.info(
        "Selection: highest spam-class F1; ties break on macro F1, then accuracy, "
        "then alphabetical model name. Positive class: %s.",
        POSITIVE_LABEL,
    )

    logger.info(
        "Refitting %s and the TF-IDF vectorizer on all %s rows for the saved artifact",
        best_name,
        len(frame),
    )
    deployed_vectorizer = build_vectorizer()
    deployed_features = deployed_vectorizer.fit_transform(frame["text"].tolist())
    deployed_model = build_models(seed)[best_name]
    deployed_model.fit(deployed_features, frame["label"].tolist())

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    vectorizer_path = output_dir / VECTORIZER_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    metrics_csv_path = output_dir / METRICS_CSV_FILENAME

    joblib.dump(deployed_model, model_path, compress=3)
    joblib.dump(deployed_vectorizer, vectorizer_path, compress=3)

    summary = {
        "task": "spam_vs_ham",
        "positive_label": POSITIVE_LABEL,
        "selection_metric": "f1_spam",
        "tie_break": ["f1_macro", "accuracy", "model_name_alphabetical"],
        "best_model": best_name,
        "runner_up": runner_up,
        "f1_margin": margin,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "seed": seed,
        "test_size": test_size,
        "split": split_info,
        "artifact_fit": "full_corpus_after_selection",
        "n_rows": int(len(frame)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "label_counts": {key: int(value) for key, value in counts.items()},
        "vocabulary_size": int(len(deployed_vectorizer.vocabulary_)),
        "evaluation_vocabulary_size": int(len(vectorizer.vocabulary_)),
        "preprocessing": (
            "lowercase; HTML stripped; URLs replaced with 'url' plus host words; "
            "email addresses replaced with 'emailaddr'; digits replaced with 'num'"
        ),
        "vectorizer": {
            "type": "tfidf",
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "sublinear_tf": True,
        },
        "models": rows,
        "artifacts": {
            "model": MODEL_FILENAME,
            "vectorizer": VECTORIZER_FILENAME,
            "metrics": METRICS_FILENAME,
        },
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(metrics_csv_path, index=False)

    logger.info("Saved model: %s", _display_path(model_path))
    logger.info("Saved vectorizer: %s", _display_path(vectorizer_path))
    logger.info("Saved metrics: %s", _display_path(metrics_path))
    logger.info("Training finished")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train spam vs ham classifiers and save the best model by spam F1.",
    )
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Labeled CSV path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for model.joblib, vectorizer.joblib, and metrics.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=LOG_DIR / "train.log",
        help="Training log path.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of categories per label to hold out, or fraction of rows if there is no category column.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 0.1 <= args.test_size <= 0.5:
        print("--test-size must be between 0.1 and 0.5", file=sys.stderr)
        return 2
    logger = logging.getLogger(LOGGER_NAME)
    try:
        run_training(
            data_path=args.data,
            output_dir=args.output_dir,
            log_file=args.log_file,
            test_size=args.test_size,
            seed=args.seed,
        )
    except (FileNotFoundError, ValueError) as exc:
        if logger.handlers:
            logger.error("%s", exc)
        else:
            print(exc, file=sys.stderr)
        return 1
    return 0
