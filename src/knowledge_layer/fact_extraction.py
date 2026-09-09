"""Fact extraction orchestration.

Text pages are chunked, sent to the local Ollama model (in parallel), the
responses are repaired and validated into ``Fact`` objects, and results are
deduplicated. If the model is unavailable a generic heuristic takes over.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .config import EXTRACTION_CHUNK_CHARS, MAX_FACTS_PER_CHUNK, MAX_PAGES_PER_PDF, OLLAMA_WORKERS
from .heuristic import heuristic_extract_all_pages
from .llm import ollama_available, ollama_chat, extract_json_objects
from .models import Fact
from .normalization import (
    safe_float,
    generate_id,
    normalize_unit,
    infer_period,
    infer_basis,
    infer_concept,
    dedupe,
)
from .text_extraction import extract_pdf_pages, chunk_text, evidence_snippet

EXTRACT_SYSTEM = (
    "You are a precise fact extractor for business and financial documents. "
    "From the given text, extract every meaningful numerical or semantic fact. "
    "Respond with ONLY a JSON array (or concatenated JSON objects) of objects, no prose, no markdown. "
    "Each object MUST have exactly these keys: "
    '{"subject": "what the fact is about, concise", '
    '"value": "the number or short claim as a string", '
    '"unit": "unit if present (e.g. Cr, Mn, %, tons) else empty string", '
    '"period": "time period if present (e.g. FY24, Q1 FY25) else empty string", '
    '"basis": "basis if present (e.g. adjusted, reported, real) else empty string"}. '
    "Skip generic marketing sentences. Include concrete facts only."
)


def _parse_model_facts(page: int, raw: str, filename: str, source_path: str,
                       chunk_text_ref: str, start_counter: int) -> List[Fact]:
    """Convert a repaired model response into validated Fact objects."""
    facts: List[Fact] = []
    counter = start_counter
    for obj in extract_json_objects(raw):
        subject = str(obj.get("subject", "")).strip()
        value = str(obj.get("value", "")).strip()
        if not subject or not value or len(subject) > 120:
            continue
        counter += 1
        unit = normalize_unit(str(obj.get("unit", "")), subject, value)
        period = str(obj.get("period", "")).strip() or infer_period(subject + " " + chunk_text_ref)
        basis = str(obj.get("basis", "")).strip() or infer_basis(subject)
        fragment = value if value in chunk_text_ref else (chunk_text_ref[:20] or chunk_text_ref)
        facts.append(Fact(
            id=generate_id(filename, page, counter),
            source=filename,
            source_path=source_path,
            page=page,
            subject=subject,
            value=value,
            unit=unit,
            period=period,
            basis=basis,
            qualifiers=period,
            evidence=evidence_snippet(chunk_text_ref, fragment),
            concept=infer_concept(subject),
            normalized_value=safe_float(value),
            confidence=0.9,
            extracted_by="ollama",
        ))
    return facts


def extract_facts_with_ollama(filename: str, pages: List[Dict[str, Any]], source_path: str,
                              progress_cb: Optional[Callable[[int, int], None]] = None) -> List[Fact]:
    tasks: List[tuple] = []
    for page_info in pages:
        page = page_info["page"]
        text = page_info["text"]
        if not text.strip():
            continue
        for chunk in chunk_text(text, EXTRACTION_CHUNK_CHARS):
            if chunk.strip():
                tasks.append((page, chunk))

    total = len(tasks)
    results: List[Optional[tuple]] = [None] * total
    done = 0

    def work(task: tuple) -> tuple:
        page, chunk = task
        try:
            raw = ollama_chat(
                EXTRACT_SYSTEM,
                f"Extract all facts from this document text:\n\n{chunk}",
                use_json_format=False,
            )
            return page, chunk, raw
        except Exception as e:  # noqa: BLE001 - a bad chunk should not kill a run
            print(f"[ollama] extraction chunk failed (page {page}): {e}")
            return page, chunk, None

    with ThreadPoolExecutor(max_workers=OLLAMA_WORKERS) as executor:
        future_map = {executor.submit(work, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_map):
            idx = future_map[future]
            page, chunk, raw = future.result()
            done += 1
            if progress_cb:
                progress_cb(done, total)
            if raw:
                results[idx] = (page, chunk, raw)

    facts: List[Fact] = []
    counter = 0
    for item in results:
        if item is None:
            continue
        page, chunk, raw = item
        parsed = _parse_model_facts(page, raw, filename, source_path, chunk, counter)
        counter += len(parsed)
        if len(parsed) > MAX_FACTS_PER_CHUNK:
            parsed = parsed[:MAX_FACTS_PER_CHUNK]
        facts.extend(parsed)
    return dedupe(facts)


def extract_facts_from_pdf(path: str,
                           progress_cb: Optional[Callable[[str, int, int], None]] = None,
                           max_pages: Optional[int] = None) -> List[Fact]:
    """Extract grounded facts from any PDF.

    ``progress_cb`` is an optional ``(stage, done, total)`` callback used by
    the UI to show progress. It is passed through into the LLM stage.

    ``max_pages`` caps the number of pages processed (defaults to the
    ``MAX_PAGES_PER_PDF`` config value) so large PDFs stay responsive.
    """
    pages = extract_pdf_pages(path)
    if len(pages) > (max_pages or MAX_PAGES_PER_PDF):
        pages = pages[: max_pages or MAX_PAGES_PER_PDF]
    filename = os.path.basename(path)

    def passthrough(done: int, total: int) -> None:
        if progress_cb:
            progress_cb("extracting", done, total)

    if ollama_available():
        facts = extract_facts_with_ollama(filename, pages, path, progress_cb=passthrough)
        if not facts:
            # Model produced nothing usable - fall back to heuristics.
            facts = heuristic_extract_all_pages(filename, pages, path)
        return dedupe(facts)

    return dedupe(heuristic_extract_all_pages(filename, pages, path))