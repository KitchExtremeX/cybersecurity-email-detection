import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.mbox_io import classify_mbox
from src.predict.service import load_classifier
from src.train.metrics import select_best
from src.train.pipeline import run_training


class PipelineTests(unittest.TestCase):
    def test_training_smoke_and_prediction(self):
        frame = pd.read_csv(ROOT / "data" / "emails.csv")
        small = frame.groupby("label", group_keys=False).sample(n=80, random_state=7)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            data_path = work / "small.csv"
            output_dir = work / "artifacts"
            log_file = work / "train.log"
            small.to_csv(data_path, index=False)
            summary = run_training(
                data_path=data_path,
                output_dir=output_dir,
                log_file=log_file,
                test_size=0.25,
                seed=7,
            )
            self.assertEqual(summary["best_model"], select_best(summary["models"])["model"])
            winner = next(row for row in summary["models"] if row["model"] == summary["best_model"])
            self.assertGreaterEqual(winner["f1_spam"], max(row["f1_spam"] for row in summary["models"]))
            self.assertEqual(
                {row["model"] for row in summary["models"]},
                {"svm", "logistic_regression", "decision_tree", "random_forest"},
            )
            self.assertTrue((output_dir / "model.joblib").is_file())
            self.assertTrue((output_dir / "vectorizer.joblib").is_file())
            saved = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["best_model"], summary["best_model"])
            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("Metric table:", log_text)
            self.assertIn("Best model:", log_text)
            self.assertIn(summary["best_model"], log_text)

            classifier = load_classifier(output_dir)
            result = classifier.predict_text("Subject: hello\n\nThis is a short note.")
            self.assertIn(result["label"], {"ham", "spam"})
            self.assertGreater(result["confidence"], 0)
            self.assertLessEqual(result["confidence"], 1)
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1, places=5)

            rows = classify_mbox(classifier, ROOT / "data" / "sample_inbox.mbox")
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                list(rows[0].keys()),
                ["subject", "from", "date", "snippet", "label", "confidence"],
            )


if __name__ == "__main__":
    unittest.main()
