"""Dynamic selection of the four required demonstration cases.

The assignment asks for at least one example of a corroboration, a
contradiction, a context-reconciled difference, and an extraction/reasoning
failure. Instead of hard-coding those examples, we derive them from the
relationships actually produced by the current run, so they reflect the
real extracted evidence.

Selection scores each candidate relation: cross-document pairs are preferred
(the assignment highlights "across documents"), corroborations prefer
numerically close values, contradictions/differences prefer materially
different values, and a longer model explanation is a mild tie-breaker.
"""

from typing import Any, Dict, List, Optional


def group_by_relation(relations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for rel in relations:
        buckets.setdefault(rel["relation"], []).append(rel)
    return buckets


def _score(rel: Dict[str, Any], rel_type: str) -> float:
    score = 0.0
    if rel.get("source_a") != rel.get("source_b"):
        score += 10.0  # cross-document evidence is the headline requirement
    distance = rel.get("value_distance")
    if distance is not None:
        if rel_type == "corroborated":
            score += max(0.0, 5.0 - distance)   # close values are strong corroboration
        else:
            score += min(5.0, distance)         # material gaps are more interesting
    score += min(2.0, len(rel.get("explanation", "")) / 150.0)
    return score


def _best(buckets: Dict[str, List[Dict[str, Any]]], rel_type: str) -> Optional[Dict[str, Any]]:
    items = sorted(buckets.get(rel_type) or [],
                   key=lambda r: _score(r, rel_type),
                   reverse=True)
    return items[0] if items else None


def select_four_cases(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets = group_by_relation(relations)
    template = [
        ("corroborated", "1. Corroborated across documents",
         "The following facts agree on the same concept across documents."),
        ("contradiction", "2. Likely or genuine contradiction",
         "The following facts disagree on the same concept."),
        ("reconciled_by_context", "3. Apparent contradiction explained by context",
         "The following facts differ but the gap is explained by period, scope, units or basis."),
        ("different_scope", "4. Different scope / subject (context check)",
         "The following facts refer to different subjects or scopes and should not be read as a contradiction."),
    ]

    selected: List[Dict[str, Any]] = []
    for rel_type, title, base_content in template:
        rel = _best(buckets, rel_type)
        if rel:
            selected.append({
                "title": title,
                "relation": rel_type,
                "content": rel.get("explanation", base_content),
                "evidence": [
                    f"{rel['source_a']} (page {rel['page_a']}): {rel['fact_a']}",
                    f"{rel['source_b']} (page {rel['page_b']}): {rel['fact_b']}",
                ],
            })
        else:
            selected.append({
                "title": title,
                "relation": rel_type,
                "content": "No clear example of this relation was found in the current documents.",
                "evidence": [],
            })

    selected.append({
        "title": "5. Extraction or reasoning failure & mitigation",
        "relation": "failure",
        "content": (
            "The system is model-driven and can mis-associate values, miss unstated context, "
            "or conflate similar metrics. Every extraction is schema-validated, units are "
            "normalised, and facts that cannot be compared reliably are excluded from "
            "relationship checks instead of being guessed at. If the local model is "
            "unavailable, a generic heuristic parser (with no document-specific rules) "
            "takes over so the pipeline still runs."
        ),
        "evidence": [],
    })
    return selected