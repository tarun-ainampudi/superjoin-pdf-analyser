"""Value normalisation helpers shared by the extractors and comparer."""

import re
from typing import Optional

from .models import Fact


def safe_float(value: str) -> Optional[float]:
    try:
        cleaned = value.replace(",", "").replace("₹", "").replace("$", "").replace("%", "")
        return float(cleaned)
    except Exception:
        return None


def generate_id(filename: str, page: int, idx: int) -> str:
    return f"{filename}-p{page}-{idx}"


def normalize_unit(unit: str, subject: str = "", value: str = "") -> str:
    u = (unit or "").strip().lower()
    s = (subject or "").lower()
    if u in ("%", "percent", "per cent"):
        return "%"
    if u in ("mn", "million"):
        return "million"
    if u in ("bn", "billion"):
        return "billion"
    if u in ("cr", "crore"):
        return "crore"
    if u in ("mn tons", "mn tonnes", "kt", "tons", "tonnes", "mt"):
        return "tons"
    if u in ("shares", "equity shares"):
        return "shares"
    if u in ("days", "day"):
        return "days"
    if u:
        return u
    # Infer from the subject when no unit was stated.
    if "percent" in s or "%" in s or "growth" in s or "margin" in s or "inflation" in s:
        return "%"
    if any(k in s for k in ("revenue", "ebitda", "profit", "loss", "expense",
                            "cad", "debt", "income", "offer", "equity", "pledged")):
        return "currency"
    if any(k in u for k in ("ton", "shipment", "parcel")):
        return "tons"
    return ""


def infer_period(text: str) -> str:
    m = re.search(
        r"\b(FY\s?(?:20\d{2}|\d{2})(?:/\d{2,4})?|Q[1-4]\s?F[Yy]\s?(?:20\d{2}|\d{2})|20\d{2}\s*[-–]\s*20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if m:
        return re.sub(r"\s+", "", m.group(1)).replace("fy", "FY").replace("Fy", "FY").upper()
    return ""


def infer_basis(text: str) -> str:
    lower = (text or "").lower()
    if "adjusted" in lower or "adj." in lower:
        return "adjusted"
    if "real" in lower:
        return "real"
    if "nominal" in lower:
        return "nominal"
    return "reported"


def infer_concept(subject: str) -> str:
    s = subject.lower()
    for kw, concept in [
        ("ebitda", "ebitda"),
        ("revenue", "revenue"),
        ("gdp", "gdp"),
        ("economy", "gdp"),
        ("economic", "gdp"),
        ("inflation", "inflation"),
        ("tonnage", "volume_tonnage"),
        ("tonne", "volume_tonnage"),
        ("tons", "volume_tonnage"),
        ("shipment", "volume_shipments"),
        ("parcel", "volume_shipments"),
        ("current account", "current_account"),
        ("fiscal", "fiscal"),
        ("pat", "profit"),
        ("profit", "profit"),
        ("loss", "profit"),
        ("margin", "margin"),
        ("nwc", "working_capital"),
        ("net working capital", "working_capital"),
        ("growth", "growth"),
        ("offer", "offer"),
        ("equity share", "shares"),
        ("shares", "shares"),
    ]:
        if kw in s:
            return concept
    return "general_metric"


# Stop-words and noise tokens removed when building a comparison key.
_STOPWORDS = {
    "the", "and", "of", "for", "in", "on", "to", "at", "with", "was", "were",
    "is", "are", "be", "by", "from", "as", "a", "an", "its", "per", "during",
    "over", "vs", "vs.", "net", "total", "from", "since", "our", "of",
    "of", "by", "or", "rs", "inr", "cr", "mn", "bn", "usd", "inr",
}


def subject_key(subject: str) -> tuple:
    """Normalise a subject into a comparison key.

    Strips parentheticals, period markers (FY24, Q1FY25, years) and stop
    words so that facts expressed differently still group together, e.g.
    "FY24 EBITDA" and "EBITDA" both collapse to ``("ebitda",)``.
    """
    s = (subject or "").lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"\bfy\s?\d{2,4}(?:/\d{2,4})?\b", " ", s)
    s = re.sub(r"\bq[1-4]\s?fy\s?\d+\b", " ", s)
    s = re.sub(r"\b(20\d{2}|19\d{2})\b", " ", s)
    s = re.sub(r"\b(rs\.?|inr|usd|\.)\b", " ", s)
    tokens = re.findall(r"[a-z0-9%]+", s)
    return tuple(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


def dedupe(facts: list) -> list:
    """Drop exact duplicates, keeping the first occurrence of each key."""
    seen = {}
    for fact in facts:
        key = (fact.source, fact.page, fact.subject.lower(), str(fact.value), fact.unit)
        seen.setdefault(key, fact)
    return list(seen.values())