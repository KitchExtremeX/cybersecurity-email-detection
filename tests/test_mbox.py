import mailbox
import sys
import tempfile
import unittest
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict.mbox_io import read_mbox


class MboxTests(unittest.TestCase):
    def test_sample_inbox_has_eight_messages(self):
        messages = read_mbox(ROOT / "data" / "sample_inbox.mbox")
        self.assertEqual(len(messages), 8)
        for message in messages:
            self.assertIn("Subject:", message["text"])
            self.assertTrue(message["subject"])
            self.assertTrue(message["from"])
            self.assertTrue(message["date"])

    def test_html_and_attachment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mbox"
            box = mailbox.mbox(path)
            try:
                plain = EmailMessage()
                plain["From"] = "Ada Lovelace <ada@northwind.example>"
                plain["Subject"] = "Notes"
                plain["Date"] = "Mon, 21 Sep 2026 09:00:00 +0000"
                plain.set_content("See you at the review.")
                plain.add_attachment(
                    b"not a real file\n",
                    maintype="application",
                    subtype="octet-stream",
                    filename="Q3-invoice.pdf",
                )
                html = EmailMessage()
                html["From"] = "Sam <sam@northwind.example>"
                html["Subject"] = "Rendered note"
                html["Date"] = "Tue, 22 Sep 2026 09:00:00 +0000"
                html.set_content("<p>Hello <b>team</b></p>", subtype="html")
                box.add(plain)
                box.add(html)
            finally:
                box.close()
            messages = read_mbox(path)
        self.assertEqual(messages[0]["subject"], "Notes")
        self.assertIn("Ada Lovelace", messages[0]["from"])
        self.assertIn("review", messages[0]["text"])
        self.assertIn("Attachment: Q3-invoice.pdf", messages[0]["text"])
        self.assertIn("Q3-invoice.pdf", messages[0]["snippet"])
        self.assertIn("Hello", messages[1]["text"])
        self.assertIn("team", messages[1]["text"])
        self.assertNotIn("<b>", messages[1]["text"])

    def test_missing_and_empty_mbox(self):
        with self.assertRaises(FileNotFoundError):
            read_mbox(ROOT / "data" / "missing.mbox")
        with tempfile.TemporaryDirectory() as directory:
            empty = Path(directory) / "empty.mbox"
            empty.write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_mbox(empty)


if __name__ == "__main__":
    unittest.main()
