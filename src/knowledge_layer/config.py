"""Configuration for the knowledge layer (reads from environment with sane defaults)."""

import os


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer setting without making app startup fragile."""
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _positive_float(name: str, default: float) -> float:
    """Read a positive timeout setting without making app startup fragile."""
    try:
        return max(0.1, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default

# Ollama backend
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.1:3b")
OLLAMA_RETRIES = _positive_int("OLLAMA_RETRIES", 3)
# Max wall-clock time a single Ollama response may take before we fall back
# to the standard (heuristic) extraction path.
OLLAMA_CALL_TIMEOUT = _positive_float("OLLAMA_CALL_TIMEOUT", 60.0)

# Gemini backend (priority backend)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_TIMEOUT = _positive_float("GEMINI_TIMEOUT", 180.0)

# Extraction tuning
EXTRACTION_CHUNK_CHARS = _positive_int("EXTRACTION_CHUNK_CHARS", 1200)
MAX_FACTS_PER_CHUNK = _positive_int("MAX_FACTS_PER_CHUNK", 40)
# Cap total pages processed per PDF so large files stay responsive.
MAX_PAGES_PER_PDF = _positive_int("MAX_PAGES_PER_PDF", 20)
# Max number of LLM relationship-comparison calls per run.
RELATION_BUDGET = _positive_int("RELATION_BUDGET", 40)
