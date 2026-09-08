import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import pymupdf


@dataclass
class Fact:
    id: str
    source: str
    source_path: str
    page: int
    fact_type: str
    subject: str
    value: str
    unit: str
    qualifiers: str
    evidence: str
    normalized_value: float | None = None
    concept: str = ""
    period: str = ""
    basis: str = ""


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_float(value: str) -> float | None:
    try:
        cleaned = value.replace(",", "").replace("₹", "").replace("%", "")
        return float(cleaned)
    except Exception:
        return None


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_facts_from_pdf(path: str) -> List[Fact]:
    doc = pymupdf.open(path)
    facts: List[Fact] = []
    filename = os.path.basename(path)

    # Extract candidate snippets from each page and handle common patterns.
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text", sort=True)
        lines = [one_line(line) for line in text.splitlines() if one_line(line)]

        # Consolidate a few lines around likely metric sentences.
        combined = " ".join(lines)

        # Shared patterns
        patterns = [
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*₹\s*(?P<value>[0-9,]+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]+)", "currency"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*per\s*cent", "percent"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*percent", "percent"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*%", "percent"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*Mn", "currency"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*Bn", "currency"),
            (r"(?P<label>[A-Za-z0-9/()\- ,&]+?)\s*[:\-]\s*(?P<value>\d+(?:\.\d+)?)\s*Cr", "currency"),
        ]

        for pattern, kind in patterns:
            for match in re.finditer(pattern, combined, flags=re.IGNORECASE):
                label = normalize_space(match.group("label"))
                value = match.group("value")
                unit = match.groupdict().get("unit", "")
                if not label or not value:
                    continue
                if len(label) > 140:
                    continue

                concept = infer_concept(label)
                subject = label.lower().replace("fy24", "FY24").replace("fy25", "FY25")
                evidence = sentence_from_text(combined, match.group(0))
                fact = Fact(
                    id=f"{filename}-{page_index+1}-{len(facts)+1}",
                    source=filename,
                    source_path=path,
                    page=page_index + 1,
                    fact_type=kind,
                    subject=subject,
                    value=value,
                    unit=unit if unit else infer_unit(label, value),
                    qualifiers=qualify_sentence(subject, combined),
                    evidence=evidence,
                    normalized_value=safe_float(value),
                    concept=concept,
                    period=infer_period(subject + " " + combined),
                    basis=infer_basis(subject),
                )
                facts.append(fact)

        # Special targeted extraction for the more important facts from these docs.
        facts.extend(extract_semantic_facts(filename, combined, page_index + 1, path))

    deduped: Dict[str, Fact] = {}
    for fact in facts:
        key = (fact.source, fact.page, fact.subject, fact.value, fact.fact_type)
        deduped.setdefault(key, fact)
    return list(deduped.values())


def infer_concept(label: str) -> str:
    lower = label.lower()
    if "gdp" in lower:
        return "gdp"
    if "inflation" in lower:
        return "inflation"
    if "ebitda" in lower:
        return "ebitda"
    if "revenue" in lower:
        return "revenue"
    if "current account" in lower or "cad" in lower:
        return "current_account"
    if "fiscal" in lower:
        return "fiscal"
    if "parcel" in lower or "shipment" in lower or "tonnage" in lower:
        return "shipment_volume"
    if "growth" in lower:
        return "growth"
    return "general_metric"


def infer_unit(label: str, value: str) -> str:
    l = label.lower()
    if "percent" in l or "%" in l or "growth" in l or "inflation" in l:
        return "%"
    if "revenue" in l or "ebitda" in l or "cad" in l or "balance" in l or "debt" in l:
        return "currency"
    if "tonnage" in l or "shipments" in l:
        return "units"
    if "mn" in l or "million" in l:
        return "million"
    if "bn" in l or "billion" in l:
        return "billion"
    return ""


def sentence_from_text(text: str, match_fragment: str) -> str:
    if match_fragment in text:
        start = max(0, text.index(match_fragment) - 120)
        end = min(len(text), text.index(match_fragment) + len(match_fragment) + 180)
        snippet = text[start:end]
        return normalize_space(snippet)
    return normalize_space(text[:400])


def qualify_sentence(subject: str, text: str) -> str:
    subject_l = subject.lower()
    for keyword in ["fy24", "fy25", "fy2024/25", "q1", "q4", "real gdp", "headline inflation", "revenue from services"]:
        if keyword in subject_l:
            return keyword
    return "not specified"


def infer_period(text: str) -> str:
    match = re.search(r"\b(FY\s?(?:20\d{2}|\d{2})(?:/\d{2,4})?|Q[1-4]\s?FY\s?(?:20\d{2}|\d{2}))\b", text, re.IGNORECASE)
    return normalize_space(match.group(1).upper()) if match else ""


def infer_basis(text: str) -> str:
    lower = text.lower()
    if "adjusted ebitda" in lower or "adj. ebitda" in lower:
        return "adjusted"
    if "real gdp" in lower:
        return "real"
    if "nominal gdp" in lower:
        return "nominal"
    return "reported"


def extract_semantic_facts(filename: str, text: str, page_number: int, source_path: str) -> List[Fact]:
    facts: List[Fact] = []
    doc_prefix = filename.lower()

    # Delhivery facts
    if "delhivery" in doc_prefix:
        patterns = {
            "revenue_from_services": r"FY24\s+revenue\s+from\s+services\s*[\w\s/()]*?₹?\s*([0-9,]+)\s*Cr",
            "ebitda": r"FY24\s+EBITDA\s*[:/\-]?\s*₹?\s*([0-9,]+)\s*Cr",
            "adjusted_ebitda": r"Adj\.\s*EBITDA\s*[:/\-]?\s*₹?\s*([0-9,]+)\s*Cr",
            "express_shipments": r"Express\s+parcel\s+shipments\s+in\s+FY24\s*[:\-]?\s*([0-9,]+)\s*Mn",
            "ptl_tonnage": r"PTL\s+freight\s+tonnage\s+in\s+FY24\s*[:\-]?\s*([0-9,]+)\s*Mn",
            "nwc_days": r"NWC\s+days\s*[:\-]?\s*([0-9]+)\s*to\s*([0-9]+)",
            "pat_improvement": r"PAT\s+loss\s+reduced\s+by\s+₹?\s*([0-9,]+)\s*Cr",
        }

        for label, pattern in patterns.items():
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1)
            if label == "nwc_days":
                value = f"{m.group(1)} to {m.group(2)}"
            subject = label.replace("_", " ")
            facts.append(Fact(
                id=f"{filename}-{page_number}-{label}",
                source=filename,
                source_path=source_path,
                page=page_number,
                fact_type="currency" if "cr" in text.lower() or "₹" in text else "number",
                subject=subject,
                value=value,
                unit="Cr" if "cr" in pattern.lower() else "Mn" if "mn" in pattern.lower() else "days" if label == "nwc_days" else "",
                qualifiers="FY24",
                evidence=normalize_space(text[max(0, text.find(m.group(0))-80): min(len(text), text.find(m.group(0))+220)]),
                normalized_value=safe_float(value.replace(" to ", " ").split()[0]) if label != "nwc_days" else None,
                concept=infer_concept(subject),
                period=infer_period(text),
                basis=infer_basis(subject),
            ))

    # Macro facts
    if "india" in doc_prefix or "rbi" in doc_prefix or "imf" in doc_prefix:
        macro_patterns = {
            "real_gdp_fy25": r"real\s+GDP\s+is\s+estimated\s+to\s+grow\s+by\s+(\d+\.\d+)\s+per\s+cent\s+in\s+FY25",
            "real_gdp_fy2024_25": r"Following\s+economic\s+growth\s+of\s+(\d+\.\d+)\s+percent\s+in\s+FY2024/25",
            "headline_inflation": r"Headline\s+inflation\s+has\s+declined\s+markedly.*?(\d+\.\d+)\s+percent",
            "gdp_projection_2025_26": r"real\s+GDP\s+is\s+projected\s+to\s+grow\s+at\s+(\d+\.\d+)\s+percent\s+in\s+FY2025/26",
            "gdp_projection_2026_27": r"FY2026/27\s+.*?(\d+\.\d+)\s+percent",
            "current_account_deficit": r"current\s+account\s+deficit\s+has\s+been\s+contained",
            "rbi_fiscal": r"the\s+central\s+government\s+continued\s+with\s+its\s+efforts\s+towards\s+fiscal\s+consolidation",
        }

        for label, pattern in macro_patterns.items():
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1) if m.groups() else "yes"
            subject = label.replace("_", " ")
            facts.append(Fact(
                id=f"{filename}-{page_number}-{label}",
                source=filename,
                source_path=source_path,
                page=page_number,
                fact_type="percent" if re.search(r"\d+\.\d+", value) else "claim",
                subject=subject,
                value=value,
                unit="%" if re.search(r"\d+\.\d+", value) else "",
                qualifiers="macro",
                evidence=normalize_space(text[max(0, text.find(m.group(0))-100): min(len(text), text.find(m.group(0))+220)]),
                normalized_value=safe_float(value),
                concept=infer_concept(subject),
                period=infer_period(text),
                basis=infer_basis(subject),
            ))

    return facts


def compare_facts(all_facts: List[Fact]) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    normalized = [(fact.concept, fact) for fact in all_facts if fact.concept]
    seen = set()

    for i in range(len(normalized)):
        concept_a, fact_a = normalized[i]
        if concept_a == "general_metric" or fact_a.normalized_value is None:
            continue
        for j in range(i + 1, len(normalized)):
            concept_b, fact_b = normalized[j]
            if concept_a != concept_b:
                continue
            if concept_b == "general_metric" or fact_b.normalized_value is None:
                continue
            pair_key = (fact_a.id, fact_b.id)
            if pair_key in seen or (fact_b.id, fact_a.id) in seen:
                continue
            seen.add(pair_key)

            val_a = fact_a.normalized_value
            val_b = fact_b.normalized_value
            if val_a is None or val_b is None:
                kind = "same_concept"
            elif abs(val_a - val_b) <= 0.2:
                kind = "corroborated"
            elif fact_a.subject.lower() != fact_b.subject.lower():
                kind = "likely_contradiction"
            else:
                kind = "contextual_reconciliation"

            relations.append({
                "fact_a": fact_a.subject,
                "fact_b": fact_b.subject,
                "source_a": fact_a.source,
                "source_b": fact_b.source,
                "page_a": fact_a.page,
                "page_b": fact_b.page,
                "relation": kind,
                "explanation": explain_relation(fact_a, fact_b, kind),
            })

    return relations


def explain_relation(fact_a: Fact, fact_b: Fact, relation: str) -> str:
    if relation == "corroborated":
        return "The two facts describe the same concept and values are close enough to be consistent after rounding or report timing differences."
    if relation == "likely_contradiction":
        return "The values differ materially and likely reflect different periods, definitions, or measurement bases; the evidence should be read together before treating it as a true contradiction."
    return "The facts appear different because they refer to different scopes, periods, or bases such as FY25 vs FY2024/25 or adjusted vs reported metrics."


def load_default_documents(base_dir: str) -> List[str]:
    pdfs = []
    for path in sorted(Path(base_dir).rglob("*.pdf")):
        if "superjoin-vit-2026-assignment" not in path.name:
            pdfs.append(str(path))
    return pdfs


def load_all_facts(base_dir: str) -> List[Fact]:
    all_facts: List[Fact] = []
    for pdf_path in load_default_documents(base_dir):
        all_facts.extend(extract_facts_from_pdf(pdf_path))
    return all_facts
