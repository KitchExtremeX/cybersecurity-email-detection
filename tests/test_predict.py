import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.mbox_io import classify_mbox
from src.predict.service import load_classifier

HAM = """Subject: Design review Thursday

Hi team,

Can we keep Thursday at 15:00 for the API design review? The notes are already on the shared drive.

Thanks,
Priya
"""

SPAM = """Subject: Mailbox locks in 2 hours

Dear customer,

Verify your password now or the account will be suspended.

http://mailbox-verify-login.example/secure

This link expires in 2 hours.
"""


class PredictTests(unittest.TestCase):
    def test_missing_artifacts_tell_you_to_train(self):
        missing = ROOT / "outputs" / "_missing_model_dir"
        with self.assertRaises(FileNotFoundError) as caught:
            load_classifier(missing)
        self.assertIn("python -m src.train", str(caught.exception))

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.predict",
                "--text",
                "Subject: hi\n\nHello",
                "--output-dir",
                str(missing),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("python -m src.train", result.stderr)

    def test_shipped_model_scores_obvious_mail(self):
        classifier = load_classifier(ROOT / "outputs")
        ham = classifier.predict_text(HAM)
        spam = classifier.predict_text(SPAM)
        self.assertEqual(ham["label"], "ham")
        self.assertEqual(spam["label"], "spam")
        self.assertGreater(spam["probabilities"]["spam"], 0.5)
        self.assertGreater(ham["probabilities"]["ham"], 0.5)

        rows = classify_mbox(classifier, ROOT / "data" / "sample_inbox.mbox")
        by_subject = {row["subject"]: row for row in rows}
        self.assertEqual(by_subject["Design review Thursday"]["label"], "ham")
        self.assertEqual(by_subject["Mailbox locks in 2 hours"]["label"], "spam")
        self.assertEqual(by_subject["You won 7500 USD in the Austin draw"]["label"], "spam")
        self.assertEqual(by_subject["Lab VPN passwords rotate Friday"]["label"], "ham")


if __name__ == "__main__":
    unittest.main()