"""Load a document's text from a PDF or plain-text file.

Kept intentionally small: real pipelines add OCR for scanned PDFs, but for a
text-based demo ``pypdf`` plus a plain-text fallback is enough.
"""
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


def load_document(path: str | Path) -> str:
    """Return the extracted text of a ``.pdf`` or ``.txt`` file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = path.read_text(encoding="utf-8")

    text = text.strip()
    if not text:
        raise ValueError(
            f"No extractable text in {path}. If it is a scanned PDF, run OCR first."
        )
    return text
