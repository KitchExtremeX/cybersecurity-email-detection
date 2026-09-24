"""CLI: python -m src.predict --text "..." | --mbox inbox.mbox"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.config import LOG_DIR, OUTPUT_DIR
from src.logging_utils import configure_logging
from src.predict.mbox_io import classify_mbox
from src.predict.service import load_classifier

LOGGER_NAME = "email_detection.predict"
CSV_FIELDS = ["subject", "from", "date", "snippet", "label", "confidence"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify email text or an mbox file.")
    parser.add_argument("--text", help="A single email (subject and body).")
    parser.add_argument("--mbox", type=Path, help="Path to an .mbox file.")
    parser.add_argument(
        "--csv",
        type=Path,
        help="Where to write mbox results. Defaults to stdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory with model.joblib and vectorizer.joblib.",
    )
    parser.add_argument("--json", action="store_true", help="Print a single-text result as JSON.")
    return parser.parse_args(argv)


def _write_csv(rows: list[dict], destination) -> None:
    writer = csv.DictWriter(destination, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.text) == bool(args.mbox):
        print("Provide exactly one of --text or --mbox.", file=sys.stderr)
        return 2

    logger = configure_logging(LOGGER_NAME, LOG_DIR / "predict.log")
    try:
        classifier = load_classifier(args.output_dir)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1

    logger.info("Loaded model %s from %s", classifier.model_name, classifier.output_dir)

    try:
        if args.text:
            result = classifier.predict_text(args.text)
            logger.info(
                "Scored one message: label=%s confidence=%.4f",
                result["label"],
                result["confidence"],
            )
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"label: {result['label']}")
                print(f"confidence: {result['confidence']:.4f}")
                for name, score in sorted(result["probabilities"].items()):
                    print(f"{name}: {score:.4f}")
            return 0

        rows = classify_mbox(classifier, args.mbox)
        logger.info("Classified %s messages from %s", len(rows), args.mbox)
        if args.csv:
            args.csv.parent.mkdir(parents=True, exist_ok=True)
            with args.csv.open("w", encoding="utf-8", newline="") as handle:
                _write_csv(rows, handle)
            print(f"Wrote {len(rows)} rows to {args.csv}")
        else:
            _write_csv(rows, sys.stdout)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
