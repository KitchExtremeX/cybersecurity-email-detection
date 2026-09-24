import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.split import split_frame


class SplitTests(unittest.TestCase):
    def test_shipped_corpus_holds_out_whole_categories(self):
        frame = pd.read_csv(ROOT / "data" / "emails.csv")
        train, test, info = split_frame(frame, test_size=0.25, seed=42)
        self.assertEqual(info["strategy"], "category_holdout")
        self.assertEqual(set(train["category"]) & set(test["category"]), set())
        self.assertEqual(set(train["label"]), {"ham", "spam"})
        self.assertEqual(set(test["label"]), {"ham", "spam"})
        self.assertGreater(len(test), 100)

    def test_blank_category_uses_stratified_rows(self):
        frame = pd.DataFrame(
            {
                "text": [f"message {index}" for index in range(20)],
                "label": ["ham"] * 10 + ["spam"] * 10,
                "category": [""] * 20,
            }
        )
        _train, test, info = split_frame(frame, test_size=0.25, seed=1)
        self.assertEqual(info["strategy"], "stratified_row")
        self.assertEqual(set(test["label"]), {"ham", "spam"})


if __name__ == "__main__":
    unittest.main()
