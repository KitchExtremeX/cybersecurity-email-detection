"""Train/test splits for the model comparison.

When every row has a category and each label has at least two categories, the
holdout reserves whole categories. That stops shared template wording from
sitting on both sides of a random row split. Otherwise the split is stratified
by label.
"""

from __future__ import annotations

import random

import pandas as pd
from sklearn.model_selection import train_test_split


def grouped_holdout_ready(frame: pd.DataFrame) -> bool:
    if "category" not in frame.columns:
        return False
    categories = frame["category"].fillna("").astype(str).str.strip()
    if categories.eq("").any():
        return False
    counts = frame.groupby("label")["category"].nunique()
    if len(counts) < 2:
        return False
    return bool((counts >= 2).all())


def split_frame(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return train rows, test rows, and a JSON-ready description of the split."""
    if grouped_holdout_ready(frame):
        train, test, test_categories = _split_by_category(frame, test_size, seed)
        return train, test, {
            "strategy": "category_holdout",
            "test_size": test_size,
            "seed": seed,
            "test_categories": test_categories,
            "note": (
                "Whole categories are held out, with about test_size of the "
                "categories for each label reserved for testing. A category never "
                "appears in both train and test."
            ),
        }

    train, test = _split_rows(frame, test_size, seed)
    return train, test, {
        "strategy": "stratified_row",
        "test_size": test_size,
        "seed": seed,
        "test_categories": [],
        "note": (
            "Stratified row split. Provide a category column with at least two "
            "categories per label to hold out whole scenarios instead."
        ),
    }


def _split_by_category(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = random.Random(seed)
    test_categories: list[str] = []
    for label in ("ham", "spam"):
        categories = sorted(frame.loc[frame["label"] == label, "category"].unique())
        rng.shuffle(categories)
        n_test = max(1, int(round(len(categories) * test_size)))
        n_test = min(n_test, len(categories) - 1)
        test_categories.extend(categories[:n_test])

    test_mask = frame["category"].isin(test_categories)
    train = frame.loc[~test_mask].reset_index(drop=True)
    test = frame.loc[test_mask].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Category holdout produced an empty train or test set.")
    for part, name in ((train, "train"), (test, "test")):
        present = set(part["label"])
        if present != {"ham", "spam"}:
            raise ValueError(
                f"Category holdout left {name} without both labels. Use another --seed."
            )
    return train, test, sorted(test_categories)


def _split_rows(
    frame: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
