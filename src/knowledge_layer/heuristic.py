"""Generic offline fallback extractor.

Used only when the local Ollama model is unreachable. This parser matches
labelled metric patterns found in any business/financial text. It is purely
generic - it does NOT know about Delhivery, India macroeconomics, or any
specific document, so it generalises to new PDFs by design.
"""

import re
from typing import Any, Dict, List

from .models import Fact
from .normalization import (
    safe_float,
    generate_id,
    normalize_unit,
    infer_period,
    infer_basis,
    infer_concept,
)
from .text_extraction import evidence_snippet

# Value-first regexes. We find a numeric value (currency, percent, or a
# scaled count) and then derive the label from the text immediately before
# it. This avoids ambiguous "label: value" parsing and its catastrophic
# backtracking on long lines of dense tables.
_VALUE_PATTERNS = [
    (r"[₹$€]\s*(?P<value>[0-9,]+(?:\.\d+)?)\s*(?P<unit>[A-Za-z%]+)?", "currency"),
    (r"(?P<value>\d+(?:\.\d+)?)\s*per\s*cent", "percent"),
    (r"(?P<value>\d+(?:\.\d+)?)\s*%", "percent"),
    (r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>Mn|Bn|Cr|Million|Billion|Crore)", "count"),
]

# Characters that mark a reasonable label boundary when scanning backwards.
_LABEL_BOUNDARY = r"[.!?:;|•\n]"
_MAX_LABEL_BACK = 160


def _derive_label(combined: str, start: int) -> str:
    """Take up to one phrase of text before ``start`` as the subject label."""
    snippet = combined[max(0, start - _MAX_LABEL_BACK): start]
    # Cut at the nearest sentence/segment boundary so the label stays compact.
    boundaries = [m.start() for m in re.finditer(_LABEL_BOUNDARY, snippet)]
    if boundaries:
        snippet = snippet[boundaries[-1] + 1:]
    label = re.sub(r"\s+", " ", snippet).strip(" :;|-\u2022")
    if len(label) > 80:
        # Still too long: keep a trailing window instead of the whole sentence.
        label = " ".join(label.split()[-6:])
    return label


def heuristic_extract_page(filename: str, page_text: str, page: int, source_path: str) -> List[Fact]:
    facts: List[Fact] = []
    combined = re.sub(r"\s+", " ", page_text)
    for pattern, kind in _VALUE_PATTERNS:
        for match in re.finditer(pattern, combined, flags=re.IGNORECASE):
            value = match.group("value")
            if not value:
                continue
            label = _derive_label(combined, match.start())
            if not label:
                continue
            unit_raw = match.groupdict().get("unit") or ""
            unit = normalize_unit(unit_raw, label, value)
            facts.append(Fact(
                id=generate_id(filename, page, len(facts) + 1),
                source=filename,
                source_path=source_path,
                page=page,
                subject=label,
                value=value,
                unit=unit,
                period=infer_period(label + " " + combined),
                basis=infer_basis(label),
                qualifiers="",
                evidence=evidence_snippet(combined, match.group(0)),
                concept=infer_concept(label),
                normalized_value=safe_float(value),
                confidence=0.4,
                extracted_by="heuristic",
            ))
    return facts


def heuristic_extract_all_pages(filename: str, pages: List[Dict[str, Any]], source_path: str) -> List[Fact]:
    facts: List[Fact] = []
    for page_info in pages:
        facts.extend(heuristic_extract_page(filename, page_info["text"], page_info["page"], source_path))
    return facts