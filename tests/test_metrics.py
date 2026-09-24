import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train.metrics import format_metrics_table, select_best


def _row(model, f1_spam, f1_macro, accuracy):
    return {
        "model": model,
        "f1_spam": f1_spam,
        "f1_macro": f1_macro,
        "accuracy": accuracy,
        "precision_spam": f1_spam,
        "recall_spam": f1_spam,
    }


class MetricsTests(unittest.TestCase):
    def test_higher_spam_f1_wins(self):
        rows = [
            _row("svm", 0.91, 0.5, 0.5),
            _row("logistic_regression", 0.8, 0.99, 0.99),
        ]
        self.assertEqual(select_best(rows)["model"], "svm")

    def test_tie_breaks_on_macro_f1_then_accuracy_then_name(self):
        tied_f1 = [
            _row("svm", 0.9, 0.8, 0.99),
            _row("logistic_regression", 0.9, 0.85, 0.7),
        ]
        self.assertEqual(select_best(tied_f1)["model"], "logistic_regression")

        tied_macro = [
            _row("random_forest", 0.9, 0.85, 0.7),
            _row("decision_tree", 0.9, 0.85, 0.8),
        ]
        self.assertEqual(select_best(tied_macro)["model"], "decision_tree")

        full_tie = [
            _row("svm", 0.9, 0.9, 0.9),
            _row("decision_tree", 0.9, 0.9, 0.9),
        ]
        self.assertEqual(select_best(full_tie)["model"], "decision_tree")

    def test_table_marks_the_winner(self):
        rows = [_row("svm", 0.5, 0.5, 0.5), _row("decision_tree", 0.7, 0.7, 0.7)]
        table = format_metrics_table(rows, "decision_tree")
        self.assertIn("*decision_tree", table)
        self.assertIn("prec_spam", table)
        self.assertIn("f1_spam", table)


if __name__ == "__main__":
    unittest.main()
