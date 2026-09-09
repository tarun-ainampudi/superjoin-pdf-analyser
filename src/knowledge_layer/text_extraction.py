"""Raw text extraction from PDFs and chunking utilities.

Document-agnostic: no file names, layouts or document types are assumed.
"""

import re
from typing import Any, Dict, List

import pymupdf


def extract_pdf_pages(path: str) -> List[Dict[str, Any]]:
    """Return ``[{"page": int, "text": str}, ...]`` for every page of a PDF."""
    doc = pymupdf.open(path)
    pages: List[Dict[str, Any]] = []
    try:
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text("text", sort=True)
            text = re.sub(r"[ \t]+", " ", text)
            text = text.replace("\u00a0", " ")
            pages.append({"page": i + 1, "text": text})
    finally:
        doc.close()
    return pages


def chunk_text(text: str, max_chars: int) -> List[str]:
    """Split text into non-overlapping chunks on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            chunks.append(current)
            current = sent
        else:
            current = current + (" " if current else "") + sent
    if current:
        chunks.append(current)
    return chunks


def evidence_snippet(text: str, fragment: str, before: int = 140, after: int = 220) -> str:
    """Return a window of source text around ``fragment`` for grounding."""
    idx = text.find(fragment)
    if idx == -1:
        return re.sub(r"\s+", " ", text[:400])
    start = max(0, idx - before)
    end = min(len(text), idx + len(fragment) + after)
    return re.sub(r"\s+", " ", text[start:end])