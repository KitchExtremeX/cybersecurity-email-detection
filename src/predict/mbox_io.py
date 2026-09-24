"""Parse mbox mailboxes and classify each message."""

from __future__ import annotations

import mailbox
from email.header import decode_header, make_header
from pathlib import Path

from src.preprocess import compose_email_text, html_to_text, snippet


def decode_mime(value: str | None) -> str:
    """Decode an RFC 2047 header into a plain string."""
    if value is None:
        return ""
    try:
        return str(make_header(decode_header(str(value))))
    except Exception:
        return str(value)


def _decode_part(part) -> str:
    payload = part.get_payload(decode=True)
    if payload is None:
        raw = part.get_payload()
        return raw if isinstance(raw, str) else ""
    charset = part.get_content_charset() or "utf-8"
    return payload.decode(charset, errors="replace")


def extract_body_and_attachments(message) -> tuple[str, list[str]]:
    """Return visible body text and attachment filenames."""
    attachments: list[str] = []
    plain_chunks: list[str] = []
    html_chunks: list[str] = []

    def consume(part) -> None:
        disposition = str(part.get("Content-Disposition") or "")
        filename = part.get_filename()
        if filename or "attachment" in disposition.lower():
            if filename:
                attachments.append(decode_mime(filename))
            return
        content_type = part.get_content_type()
        if content_type == "text/plain":
            text = _decode_part(part).strip()
            if text:
                plain_chunks.append(text)
        elif content_type == "text/html":
            text = html_to_text(_decode_part(part)).strip()
            if text:
                html_chunks.append(text)

    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            consume(part)
    else:
        consume(message)

    body = "\n\n".join(plain_chunks or html_chunks).strip()
    return body, attachments


def read_mbox(path: Path) -> list[dict]:
    """Parse an mbox file into subject, from, date, snippet, and model text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mbox file not found: {path}")

    box = mailbox.mbox(path)
    messages: list[dict] = []
    try:
        for message in box:
            subject = decode_mime(message.get("subject"))
            sender = decode_mime(message.get("from"))
            date = decode_mime(message.get("date"))
            body, attachments = extract_body_and_attachments(message)
            text = compose_email_text(subject, body, attachments)
            preview_source = body
            if attachments:
                names = " ".join(f"[attachment: {name}]" for name in attachments)
                preview_source = f"{body} {names}".strip()
            messages.append(
                {
                    "subject": subject,
                    "from": sender,
                    "date": date,
                    "snippet": snippet(preview_source or subject),
                    "text": text,
                }
            )
    finally:
        box.close()

    if not messages:
        raise ValueError(f"No messages found in {path}.")
    return messages


def classify_mbox(classifier, path: Path) -> list[dict]:
    """Classify each mbox message into a CSV-ready row."""
    messages = read_mbox(path)
    predictions = classifier.predict_many([message["text"] for message in messages])
    rows: list[dict] = []
    for message, prediction in zip(messages, predictions):
        rows.append(
            {
                "subject": message["subject"],
                "from": message["from"],
                "date": message["date"],
                "snippet": message["snippet"],
                "label": prediction["label"],
                "confidence": round(float(prediction["confidence"]), 4),
            }
        )
    return rows
