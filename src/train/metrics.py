"""Holdout metrics and best-model selection."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(model_name: str, y_true, y_pred) -> dict:
    """Precision, recall, and F1 for spam, plus accuracy and macro F1.

    Spam is the positive class. Confusion counts use label order ham, spam.
    """
    precision_spam, recall_spam, f1_spam, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["spam"],
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=["ham", "spam"])
    return {
        "model": model_name,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_spam": round(float(precision_spam[0]), 6),
        "recall_spam": round(float(recall_spam[0]), 6),
        "f1_spam": round(float(f1_spam[0]), 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "ham_predicted_ham": int(matrix[0, 0]),
        "ham_predicted_spam": int(matrix[0, 1]),
        "spam_predicted_ham": int(matrix[1, 0]),
        "spam_predicted_spam": int(matrix[1, 1]),
    }


def rank_models(rows: list[dict]) -> list[dict]:
    """Sort by spam F1, then macro F1, then accuracy, then model name."""
    return sorted(
        rows,
        key=lambda row: (
            -row["f1_spam"],
            -row["f1_macro"],
            -row["accuracy"],
            row["model"],
        ),
    )


def select_best(rows: list[dict]) -> dict:
    """Pick the winning row. See ``rank_models`` for the tie-break."""
    if not rows:
        raise ValueError("No model metrics to compare.")
    return rank_models(rows)[0]


def format_metrics_table(rows: list[dict], best_model: str) -> str:
    """Plain-text comparison table. The winner is marked with ``*``."""
    ordered = sorted(rows, key=lambda row: row["model"])
    header = (
        f"{'model':<22} {'accuracy':>10} {'prec_spam':>10} "
        f"{'rec_spam':>10} {'f1_spam':>10} {'f1_macro':>10}"
    )
    lines = [header, "-" * len(header)]
    for row in ordered:
        mark = "*" if row["model"] == best_model else " "
        lines.append(
            f"{mark}{row['model']:<21} "
            f"{row['accuracy']:10.4f} {row['precision_spam']:10.4f} "
            f"{row['recall_spam']:10.4f} {row['f1_spam']:10.4f} "
            f"{row['f1_macro']:10.4f}"
        )
    lines.append("* best by spam-class F1 (tie-break: macro F1, accuracy, name)")
    return "\n".join(lines)
