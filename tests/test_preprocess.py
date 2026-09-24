import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocess import compose_email_text, normalize_email_text, snippet


class PreprocessTests(unittest.TestCase):
    def test_url_keeps_host_words(self):
        text = normalize_email_text(
            "See http://paypal-secure-login.example/verify now"
        )
        tokens = text.split()
        self.assertIn("url", tokens)
        self.assertIn("paypal", tokens)
        self.assertIn("secure", tokens)
        self.assertIn("login", tokens)
        self.assertNotIn("http", tokens)

    def test_email_and_digits_collapse(self):
        text = normalize_email_text("Write ada@northwind.example about ticket 44021.")
        self.assertIn("emailaddr", text)
        self.assertIn("num", text)
        self.assertNotIn("44021", text)
        self.assertNotIn("@", text)

    def test_html_is_stripped(self):
        text = normalize_email_text("<p>Hello <b>team</b></p><script>alert(1)</script>")
        self.assertIn("hello", text)
        self.assertIn("team", text)
        self.assertNotIn("script", text)
        self.assertNotIn("alert", text)

    def test_compose_includes_attachment(self):
        text = compose_email_text("Invoice", "Please review.", ["Q3.pdf"])
        self.assertTrue(text.startswith("Subject: Invoice"))
        self.assertIn("Attachment: Q3.pdf", text)

    def test_snippet_collapses_whitespace(self):
        self.assertEqual(snippet("one\n\n two"), "one two")
        self.assertTrue(snippet("word " * 80).endswith("..."))


if __name__ == "__main__":
    unittest.main()
