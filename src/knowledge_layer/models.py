"""Data model for grounded facts."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Fact:
    """A fact grounded in a source PDF.

    Every Fact carries the evidence snippet it was extracted from, so the
    system can always explain *why* a fact is believed and point to its
    exact location in the source document.
    """

    id: str
    source: str                     # file name (e.g. "03-...-earnings-presentation.pdf")
    source_path: str                # absolute/relative path for re-opening
    page: int                       # 1-based page number
    subject: str                    # what the fact is about (e.g. "FY24 EBITDA")
    value: str                      # raw value as written in the document
    unit: str = ""                  # normalised unit (%, crore, million, billion, tons, ...)
    period: str = ""                # FY24, Q1 FY25, 2023-24, ...
    basis: str = ""                 # reported / adjusted / real / nominal ...
    qualifiers: str = ""            # free-form extra context
    evidence: str = ""              # snippet of source text supporting the fact
    concept: str = ""               # semantic bucket used for comparison
    normalized_value: Optional[float] = None
    confidence: float = 0.0
    extracted_by: str = ""          # "ollama" or "heuristic"

    def short(self) -> str:
        parts = [p for p in (self.value, self.unit, self.period) if p]
        return f"{self.subject}={''.join(parts)}".strip()