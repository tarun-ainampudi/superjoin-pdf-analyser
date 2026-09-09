"""Configuration for the knowledge layer (reads from environment with sane defaults)."""

import os

# Ollama backend
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "granite4.1:3b")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "600"))
OLLAMA_WORKERS = int(os.environ.get("OLLAMA_WORKERS", "2"))

# Extraction tuning
EXTRACTION_CHUNK_CHARS = int(os.environ.get("EXTRACTION_CHUNK_CHARS", "1200"))
MAX_FACTS_PER_CHUNK = int(os.environ.get("MAX_FACTS_PER_CHUNK", "40"))
# Cap total pages processed per PDF so large files stay responsive.
MAX_PAGES_PER_PDF = int(os.environ.get("MAX_PAGES_PER_PDF", "20"))
# Max number of LLM relationship-comparison calls per run.
RELATION_BUDGET = int(os.environ.get("RELATION_BUDGET", "40"))