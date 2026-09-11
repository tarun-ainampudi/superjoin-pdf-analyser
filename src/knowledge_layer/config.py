"""Configuration for the knowledge layer (reads from environment with sane defaults)."""

import os

# Ollama backend
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.1:3b")
OLLAMA_RETRIES = max(1, int(os.environ.get("OLLAMA_RETRIES", "3")))
# Max wall-clock time a single Ollama response may take before we fall back
# to the standard (heuristic) extraction path.
OLLAMA_CALL_TIMEOUT = float(os.environ.get("OLLAMA_CALL_TIMEOUT", "60"))

# Gemini backend (priority backend)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
GEMINI_API_URL = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com/v1beta")
GEMINI_TIMEOUT = float(os.environ.get("GEMINI_TIMEOUT", "180"))

# Extraction tuning
EXTRACTION_CHUNK_CHARS = int(os.environ.get("EXTRACTION_CHUNK_CHARS", "1200"))
MAX_FACTS_PER_CHUNK = int(os.environ.get("MAX_FACTS_PER_CHUNK", "40"))
# Cap total pages processed per PDF so large files stay responsive.
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "20"))
# Max number of LLM relationship-comparison calls per run.
RELATION_BUDGET = int(os.environ.get("RELATION_BUDGET", "40"))