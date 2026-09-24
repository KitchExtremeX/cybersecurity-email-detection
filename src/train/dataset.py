"""Load the labeled email CSV used by training.

``category`` is kept when present so the trainer can hold out whole scenarios.
It is not a model feature.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LABEL_MAP = {
    "spam": "spam",
    "ham": "ham",
    "1": "spam",
    "0": "ham",
}


def load_dataset(path: Path) -> pd.DataFrame:
    """Return columns ``text`` and ``label`` (``spam`` or ``ham``).

    Required CSV columns are ``text`` and ``label``.     ``label`` may be ``spam``/``ham`` or ``1``/``0`` (1 = spam). An optional
    ``category`` column is preserved for a scenario holdout and is not vectorized.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Expected a CSV with columns text, label."
        )

    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset {path} is missing column(s) {sorted(missing)}. "
            "Required schema: text, label (spam/ham or 1/0). Optional: category."
        )

    texts = frame["text"].fillna("").astype(str).str.strip()
    labels: list[str] = []
    errors: list[str] = []
    for index, raw in enumerate(frame["label"].tolist(), start=1):
        if pd.isna(raw):
            errors.append(f"row {index}: missing label")
            labels.append("")
            continue
        key = str(raw).strip().lower()
        mapped = LABEL_MAP.get(key)
        if mapped is None:
            errors.append(f"row {index}: {raw!r}")
            labels.append("")
        else:
            labels.append(mapped)

    if errors:
        preview = "; ".join(errors[:8])
        extra = "" if len(errors) <= 8 else f" (+{len(errors) - 8} more)"
        raise ValueError(
            "Unknown or missing labels. Use spam/ham or 1/0 (1 = spam). "
            f"{preview}{extra}"
        )

    if "category" in frame.columns:
        categories = frame["category"].fillna("").astype(str).str.strip().tolist()
    else:
        categories = [""] * len(frame)

    cleaned = pd.DataFrame(
        {"text": texts.tolist(), "label": labels, "category": categories}
    )
    cleaned = cleaned[cleaned["text"] != ""].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"Dataset {path} has no non-empty text rows.")

    counts = cleaned["label"].value_counts()
    missing_labels = [label for label in ("ham", "spam") if label not in counts.index]
    if missing_labels:
        raise ValueError(
            f"Dataset {path} is missing label(s) {missing_labels}. "
            "Include both spam and ham rows."
        )
    return cleaned
