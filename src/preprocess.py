"""Text normalization shared by training and inference.

The vectorizer stores a reference to ``normalize_email_text``. Keep this
function importable at ``src.preprocess.normalize_email_text`` so saved
artifacts load.
"""

from __future__ import annotations

import re

_HTML_BLOCK = re.compile(r"(?is)<(script|style)\b.*?>.*?</\1>")
_HTML_BREAK = re.compile(r"(?i)<br\s*/?>|</p>|</div>|</tr>|</li>")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"(https?://|www\.)[^\s<>\"']+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b")
_NUMBER = re.compile(r"\d+")
_NON_WORD = re.compile(r"[^a-z\s]+")
_SPACE = re.compile(r"\s+")
_HOST_SPLIT = re.compile(r"[^a-z0-9]+")
_URL_DROP = {"www", "http", "https", "com", "net", "org"}


def html_to_text(value: str) -> str:
    """Drop tags and keep readable text from an HTML part."""
    text = _HTML_BLOCK.sub(" ", value)
    text = _HTML_BREAK.sub("\n", text)
    text = _HTML_TAG.sub(" ", text)
    return text


def _url_token(match: re.Match[str]) -> str:
    raw = match.group(0).lower()
    host = re.sub(r"^https?://", "", raw)
    host = host.split("/")[0].split("?")[0].split("@")[-1]
    parts = [part for part in _HOST_SPLIT.split(host) if part and part not in _URL_DROP]
    tail = " ".join(parts)
    if tail:
        return f" url {tail} "
    return " url "


def normalize_email_text(text: str) -> str:
    """Lowercase, strip HTML, and collapse URLs, addresses, and digits."""
    if text is None:
        return ""
    value = html_to_text(str(text)).lower()
    value = _URL.sub(_url_token, value)
    value = _EMAIL.sub(" emailaddr ", value)
    value = _NUMBER.sub(" num ", value)
    value = _NON_WORD.sub(" ", value)
    return _SPACE.sub(" ", value).strip()


def compose_email_text(
    subject: str,
    body: str,
    attachments: list[str] | None = None,
) -> str:
    """Build the same Subject/body/attachment layout used in the training CSV."""
    lines = [f"Subject: {(subject or '').strip()}", "", (body or "").strip()]
    for name in attachments or []:
        cleaned = (name or "").strip()
        if cleaned:
            lines.append(f"Attachment: {cleaned}")
    return "\n".join(lines).strip()


def snippet(text: str, limit: int = 180) -> str:
    """Single-line preview for batch results."""
    collapsed = _SPACE.sub(" ", text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3].rstrip() + "..."
