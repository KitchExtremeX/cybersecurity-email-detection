import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_dataset import PER_LABEL, build_rows, validate_templates
from src.train.dataset import load_dataset


class DatasetTests(unittest.TestCase):
    def test_generator_is_deterministic(self):
        validate_templates()
        self.assertEqual(build_rows(seed=42, per_label=15), build_rows(seed=42, per_label=15))

    def test_shipped_csv_schema_and_size(self):
        frame = pd.read_csv(ROOT / "data" / "emails.csv")
        self.assertGreaterEqual(len(frame), PER_LABEL * 2)
        self.assertTrue({"text", "label", "category"} <= set(frame.columns))
        self.assertEqual(set(frame["label"]), {"ham", "spam"})
        self.assertTrue(frame["text"].astype(str).str.strip().ne("").all())
        self.assertTrue(frame["text"].astype(str).str.startswith("Subject:").all())
        counts = frame["label"].value_counts()
        self.assertGreaterEqual(int(counts["ham"]), PER_LABEL)
        self.assertGreaterEqual(int(counts["spam"]), PER_LABEL)

    def test_binary_labels_and_extra_columns(self):
        frame = pd.DataFrame(
            {
                "text": [f"Subject: note {index}\n\nMeet at noon." for index in range(4)]
                + [f"Subject: prize {index}\n\nClaim it." for index in range(4)],
                "label": [0, 0, 0, 0, 1, 1, 1, 1],
                "category": ["ignore"] * 8,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.csv"
            frame.to_csv(path, index=False)
            loaded = load_dataset(path)
        self.assertEqual(list(loaded.columns), ["text", "label", "category"])
        self.assertEqual(set(loaded["label"]), {"ham", "spam"})
        self.assertTrue((loaded["category"] == "ignore").all())

    def test_unknown_label_and_missing_column(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_label = Path(directory) / "bad_label.csv"
            bad_column = Path(directory) / "bad_column.csv"
            pd.DataFrame({"text": ["hello"], "label": ["unknown"]}).to_csv(bad_label, index=False)
            pd.DataFrame({"body": ["hello"], "label": ["ham"]}).to_csv(bad_column, index=False)
            with self.assertRaises(ValueError):
                load_dataset(bad_label)
            with self.assertRaises(ValueError):
                load_dataset(bad_column)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(ROOT / "data" / "does-not-exist.csv")


if __name__ == "__main__":
    unittest.main()
