"""Fact extraction orchestration.

Text pages are chunked and sent to the model one request at a time (serial).
Responses are repaired and validated into ``Fact`` objects, and results are
deduplicated. If the model is unavailable, too slow, or quota-limited, a
generic heuristic takes over.
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .cache import load_cached_facts, save_cached_facts
from .config import EXTRACTION_CHUNK_CHARS, MAX_FACTS_PER_CHUNK, MAX_PAGES_PER_PDF
from .heuristic import heuristic_extract_all_pages
from .llm import (
    ModelTimeoutError,
    ModelUnavailableError,
    call_model,
    extract_json_objects,
    gemini_available,
    get_used_backend,
    ollama_available,
    set_used_backend,
)
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

logger = logging.getLogger("superjoin.fact_extraction")

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
                       chunk_text_ref: str, start_counter: int,
                       extracted_by: str) -> List[Fact]:
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
            extracted_by=extracted_by,
        ))
    return facts


def extract_facts_with_ollama(filename: str, pages: List[Dict[str, Any]], source_path: str,
                              progress_cb: Optional[Callable[[int, int], None]] = None,
                              status_cb: Optional[Callable[[str, int, str], None]] = None) -> List[Fact]:
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
    results: List[tuple] = []

    # Send one request at a time (serial) so we never hammer the model and burn
    # through its quota. This is a test prototype, so low-and-slow is preferred.
    for idx, (page, chunk) in enumerate(tasks, start=1):
        if progress_cb:
            progress_cb(idx, total)
        try:
            logger.info("Processing extraction chunk %s/%s for page %s (%s chars)", idx, total, page, len(chunk))
            raw = call_model(
                EXTRACT_SYSTEM,
                f"Extract all facts from this document text:\n\n{chunk}",
                use_json_format=False,
            )
            logger.info("Extraction chunk %s/%s succeeded for page %s; response length=%s", idx, total, page, len(raw or ""))
            if status_cb and raw:
                status_cb(filename, page, raw.strip())
            if raw:
                results.append((page, chunk, raw, get_used_backend()))
        except ModelTimeoutError:
            # The model is too slow - hand the whole document off to the
            # standard (heuristic) extractor instead of waiting further.
            logger.warning("Extraction timed out on chunk %s/%s for page %s; falling back to heuristic extractor", idx, total, page)
            raise
        except ModelUnavailableError:
            # Retrying every remaining chunk after both backends are known to
            # be unavailable is slow and can leave a misleading partial result.
            logger.warning("All model backends became unavailable on chunk %s/%s; using heuristic extraction", idx, total)
            raise
        except Exception as e:  # noqa: BLE001 - a bad chunk should not kill a run
            logger.warning("Extraction chunk %s/%s failed for page %s: %s", idx, total, page, e)

    facts: List[Fact] = []
    counter = 0
    for page, chunk, raw, backend in results:
        parsed = _parse_model_facts(page, raw, filename, source_path, chunk, counter, backend)
        counter += len(parsed)
        if len(parsed) > MAX_FACTS_PER_CHUNK:
            parsed = parsed[:MAX_FACTS_PER_CHUNK]
        facts.extend(parsed)
    return dedupe(facts)


def extract_facts_from_pdf(path: str,
                           progress_cb: Optional[Callable[[str, int, int], None]] = None,
                           max_pages: Optional[int] = None,
                           cache_dir: Optional[Union[str, Path]] = None,
                           use_cache: bool = True,
                           status_cb: Optional[Callable[[str, int, str], None]] = None) -> List[Fact]:
    """Extract grounded facts from any PDF.

    ``progress_cb`` is an optional ``(stage, done, total)`` callback used by
    the UI to show progress. It is passed through into the LLM stage.

    ``max_pages`` caps the number of pages processed (defaults to the
    ``MAX_PAGES_PER_PDF`` config value) so large PDFs stay responsive.
    """
    logger.info("Starting PDF fact extraction for %s (max_pages=%s, cache_dir=%s)", path, max_pages, cache_dir)
    cache_path = Path(cache_dir) if cache_dir else None
    if use_cache and cache_path is not None:
        cached = load_cached_facts(path, cache_path)
        if cached is not None:
            logger.info("Using cached facts for %s from %s", path, cache_path)
            set_used_backend("cache")
            return dedupe(cached)

    pages = extract_pdf_pages(path)
    if len(pages) > (max_pages or MAX_PAGES_PER_PDF):
        pages = pages[: max_pages or MAX_PAGES_PER_PDF]
    filename = os.path.basename(path)

    def passthrough(done: int, total: int) -> None:
        if progress_cb:
            progress_cb("extracting", done, total)

    if ollama_available() or gemini_available():
        logger.info("AI backend available for %s; extracting facts with LLM path", path)
        try:
            facts = extract_facts_with_ollama(filename, pages, path, progress_cb=passthrough, status_cb=status_cb)
        except (ModelTimeoutError, ModelUnavailableError):
            logger.warning("Model extraction failed for %s; using heuristic extractor", path)
            facts = []
        if not facts:
            logger.warning("LLM extraction returned no usable facts for %s; falling back to heuristic extractor", path)
            facts = heuristic_extract_all_pages(filename, pages, path)
        deduped = dedupe(facts)
    else:
        logger.warning("No AI backend available for %s; using heuristic extractor", path)
        deduped = dedupe(heuristic_extract_all_pages(filename, pages, path))

    if use_cache and cache_path is not None:
        saved = save_cached_facts(path, deduped, cache_path)
        logger.info("Saved %s facts for %s to cache file %s", len(deduped), path, saved)
    return deduped
