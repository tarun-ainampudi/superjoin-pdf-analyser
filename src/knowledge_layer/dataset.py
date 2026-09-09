"""Dataset discovery helpers."""

from pathlib import Path
from typing import List

from .fact_extraction import extract_facts_from_pdf


def load_default_documents(base_dir: str) -> List[str]:
    """Discover all starter PDFs under ``base_dir`` (excluding the assignment PDF)."""
    pdfs = []
    for path in sorted(Path(base_dir).rglob("*.pdf")):
        if "superjoin-vit-2026-assignment" not in path.name:
            pdfs.append(str(path))
    return pdfs


def load_all_facts(base_dir: str, progress_cb=None, max_pages: int = None) -> List:
    """Extract facts from every starter PDF in ``base_dir``."""
    all_facts = []
    for pdf_path in load_default_documents(base_dir):
        all_facts.extend(extract_facts_from_pdf(pdf_path, progress_cb=progress_cb, max_pages=max_pages))
    return all_facts