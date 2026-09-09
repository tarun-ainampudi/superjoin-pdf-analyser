"""Superjoin VIT 2026 - Fact Knowledge Layer.

A document-agnostic engine that extracts grounded facts from PDFs using a
local Ollama LLM (with a generic heuristic fallback), compares them across
documents, and explains corroboration / contradiction / reconciliation.
"""

from .models import Fact
from .llm import ollama_available
from .fact_extraction import extract_facts_from_pdf
from .comparison import compare_facts
from .cases import select_four_cases
from .dataset import load_all_facts, load_default_documents

__all__ = [
    "Fact",
    "ollama_available",
    "extract_facts_from_pdf",
    "compare_facts",
    "select_four_cases",
    "load_all_facts",
    "load_default_documents",
]