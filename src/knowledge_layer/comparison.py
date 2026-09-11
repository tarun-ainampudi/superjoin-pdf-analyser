"""Cross-document fact comparison and reasoning.

Facts sharing a semantic concept (and subject) are paired and classified as
corroborated, a contradiction, reconciled by context, or different scope.
The configured model chain writes the human-readable explanation; a
document-agnostic heuristic covers the offline case.
"""

import logging
from typing import Any, Dict, List, Optional

from .config import RELATION_BUDGET
from .llm import call_model, extract_json_objects, get_active_backend
from .models import Fact
from .normalization import subject_key

RELATION_SYSTEM = (
    "You compare two facts from possibly different documents. "
    "Respond with ONLY a JSON object: {\"relation\": \"one of "
    "corroborated|contradiction|reconciled_by_context|different_scope\", "
    '"explanation": "one or two sentences explaining the relationship using specific '
    'context, units, periods or bases"}. No prose outside the JSON.'
)

VALID_RELATIONS = {"corroborated", "contradiction", "reconciled_by_context", "different_scope"}
logger = logging.getLogger("superjoin.comparison")


def _fact_repr(fact: Fact) -> str:
    return (f"{fact.subject} = {fact.value}{fact.unit} "
            f"(period={fact.period or 'unspecified'}, basis={fact.basis})")


def _llm_relation(fact_a: Fact, fact_b: Fact) -> Optional[Dict[str, Any]]:
    user = (
        f"Fact A (from {fact_a.source}, page {fact_a.page}): {_fact_repr(fact_a)}\n"
        f"Fact B (from {fact_b.source}, page {fact_b.page}): {_fact_repr(fact_b)}\n"
        "How are these two facts related?"
    )
    try:
        # Use the same Gemini -> Ollama fallback chain as extraction. Relation
        # checks used to bypass Gemini entirely and silently lost their model
        # fallback whenever the local runtime was unavailable.
        raw = call_model(RELATION_SYSTEM, user, use_json_format=False)
        objs = extract_json_objects(raw)
        if objs:
            rel = str(objs[0].get("relation", "")).lower().strip()
            expl = str(objs[0].get("explanation", "")).strip()
            if rel in VALID_RELATIONS:
                return {"relation": rel, "explanation": expl}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Model relation comparison failed; using heuristic: %s", exc)
    return None


def _heuristic_relation(fact_a: Fact, fact_b: Fact) -> Dict[str, Any]:
    """Simple offline classifier: corroborate near-identical values, otherwise
    explain via context (period/unit/basis/subject)."""
    va = fact_a.normalized_value
    vb = fact_b.normalized_value

    if va is not None and vb is not None:
        delta = abs(va - vb)
        if delta <= 0.2:
            same_subject = subject_key(fact_a.subject) == subject_key(fact_b.subject)
            return {
                "relation": "corroborated" if same_subject else "corroborated",
                "explanation": (
                    "Values are close enough to be consistent after rounding or "
                    "report-timing differences."
                ),
            }
        if delta <= 2.0:
            return {
                "relation": "reconciled_by_context",
                "explanation": (
                    "Values are near but not identical; the gap is plausibly explained "
                    "by differing periods, units, rounding, or measurement bases."
                ),
            }
        return {
            "relation": "contradiction",
            "explanation": (
                "The numeric values disagree by a material margin and the facts "
                "appear to be measuring the same thing differently."
            ),
        }

    # Non-numeric or partially numeric facts: context decides.
    if fact_a.period and fact_b.period and fact_a.period != fact_b.period:
        return {
            "relation": "reconciled_by_context",
            "explanation": f"The facts refer to different periods ({fact_a.period} vs {fact_b.period}).",
        }
    if fact_a.basis and fact_b.basis and fact_a.basis != fact_b.basis:
        return {
            "relation": "reconciled_by_context",
            "explanation": f"The facts use different measurement bases ({fact_a.basis} vs {fact_b.basis}).",
        }
    return {
        "relation": "different_scope",
        "explanation": "The facts describe different subjects or scopes and should be read in their own context.",
    }


def _relation_entry(fa: Fact, fb: Fact, rel: Dict[str, Any]) -> Dict[str, Any]:
    distance = None
    if fa.normalized_value is not None and fb.normalized_value is not None:
        distance = round(abs(fa.normalized_value - fb.normalized_value), 4)
    return {
        "relation": rel["relation"],
        "fact_a": fa.short(),
        "fact_b": fb.short(),
        "source_a": fa.source,
        "source_b": fb.source,
        "page_a": fa.page,
        "page_b": fb.page,
        "subject_a": fa.subject,
        "subject_b": fb.subject,
        "value_distance": distance,
        "explanation": rel["explanation"],
    }


def _candidate_pairs(facts: List[Fact]) -> List[tuple]:
    """Yield fact pairs that are plausibly comparable.

    Two facts are candidates when their normalised subject keys share at
    least one meaningful token (so paraphrased subjects still pair up) AND
    they differ in source, period, or basis (so we avoid near-duplicate
    comparisons of the same statement).
    """
    pairs: List[tuple] = []
    seen = set()
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            fa, fb = facts[i], facts[j]
            pair = frozenset((fa.id, fb.id))
            if pair in seen:
                continue
            if fa.source == fb.source and fa.period == fb.period and fa.basis == fb.basis:
                # Identical context and source - redundant, skip.
                continue
            ka = set(subject_key(fa.subject))
            kb = set(subject_key(fb.subject))
            if not ka or not kb or not (ka & kb):
                continue
            seen.add(pair)
            pairs.append((fa, fb))
    return pairs


def compare_facts(all_facts: List[Fact], use_llm: Optional[bool] = None) -> List[Dict[str, Any]]:
    """Compare same-concept facts across documents and return relationship
    descriptors with human-readable explanations.

    Facts within a concept bucket are paired when they share a meaningful
    subject token. Each pair is classified by the local model (or the
    heuristic fallback), and the four demonstration cases are derived later
    from the resulting relations.
    """
    if use_llm is None:
        use_llm = get_active_backend() != "none"

    buckets: Dict[str, List[Fact]] = {}
    skipped = 0
    for fact in all_facts:
        if fact.concept in ("general_metric", ""):
            skipped += 1
            continue
        buckets.setdefault(fact.concept, []).append(fact)

    cases = 0
    relations: List[Dict[str, Any]] = []
    for concept in sorted(buckets):
        facts = buckets[concept]
        if len(facts) < 2:
            continue
        pairs = _candidate_pairs(facts)
        for fa, fb in pairs:
            rel = None
            if use_llm and cases < RELATION_BUDGET:
                rel = _llm_relation(fa, fb)
            rel = rel or _heuristic_relation(fa, fb)
            if use_llm and cases < RELATION_BUDGET:
                cases += 1
            relations.append(_relation_entry(fa, fb, rel))
    return relations
