"""File parsers: PDF / DOCX / TXT / MD / code."""
from __future__ import annotations

import io

from docx import Document as DocxDocument
from pypdf import PdfReader

ALLOWED_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx",
                ".csv", ".log", ".py", ".js", ".ts", ".json"}


def parse_bytes(filename: str, mime: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf") or mime == "application/pdf":
        return _pdf(data)
    if name.endswith(".docx") or "wordprocessingml" in mime:
        return _docx(data)
    # Everything else: best-effort UTF-8 decode.
    return data.decode("utf-8", errors="replace")


def _pdf(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001
            continue
    return "\n\n".join(parts)


def _docx(data: bytes) -> str:
    d = DocxDocument(io.BytesIO(data))
    return "\n".join(p.text for p in d.paragraphs)
