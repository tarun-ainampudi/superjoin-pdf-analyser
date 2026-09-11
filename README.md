# Superjoin VIT 2026 — Fact Knowledge Layer

Extracts grounded facts from any PDF, compares them across documents, and explains corroboration, contradiction, and contextual reconciliation — powered by the local Ollama granite4.1:3b model.

## Setup and Run Instructions

### Prerequisites

1. **Python 3.10+**
2. **Ollama** (local LLM backend) — install from [ollama.com](https://ollama.com), then pull the model:
   ```bash
   ollama pull granite4.1:3b
   ```
   Verify it is running: `ollama list` should show `granite4.1:3b`.

### Install and Run

```bash
cd superjoin-pdf-analyser
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt

streamlit run app.py
```

Open http://localhost:8501. Upload any PDF or click through the starter dataset under `dataset/`.

### Environment Variables (optional)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `granite4.1:3b` | Model to use for extraction |
| `OLLAMA_CALL_TIMEOUT` | `60` | Max seconds for a single Ollama response before falling back to the heuristic extractor |
| `EXTRACTION_CHUNK_CHARS` | `1200` | Max chars per LLM prompt chunk |
| `MAX_PAGES_PER_PDF` | `20` | Cap on pages processed per PDF |

## Project Structure

```
superjoin-pdf-analyser/
├── app.py                              # Streamlit UI
├── src/knowledge_layer/                # Core engine (modular package)
│   ├── config.py                       # Env-var configuration
│   ├── models.py                       # Fact dataclass
│   ├── text_extraction.py              # PyMuPDF page extraction + chunking
│   ├── llm.py                          # Ollama + Gemini clients, fallback, JSON repair
│   ├── normalization.py                # Unit/period/basis/concept inference
│   ├── heuristic.py                    # Generic offline fallback extractor
│   ├── fact_extraction.py              # Orchestrator (serial model calls)
│   ├── comparison.py                   # Cross-document relationship reasoning
│   ├── cases.py                        # Dynamic four-case selection
│   └── dataset.py                      # Dataset file discovery
├── dataset/                            # Starter PDFs
│   ├── delhivery/
│   └── india-macroeconomy/
├── requirements.txt
└── README.md
```

## Approach

### Core Idea

Each fact is a grounded statement carrying: source document, page, subject, value, unit, period, basis, an evidence snippet, and a concept tag. This lets the system always explain *why* a fact is believed and point to its location in the source.

### Extraction Pipeline

1. **PyMuPDF** extracts raw text from every PDF page.
2. Text is split into sentence-aware chunks (default ~1200 chars).
3. Each chunk is sent to the active model **one request at a time (serial)**, asking for structured JSON facts.
4. The model's response — which may be a JSON array, a single object, or concatenated objects — is **repaired and validated** into the strict `Fact` schema.
5. Facts are deduplicated across chunks and normalised (units, periods, concepts).
6. Fallbacks (in order): **Gemini API** (if a key is configured) → **Ollama** → **generic heuristic extractor** (no document-specific rules).
   - A Gemini **429 (quota exceeded)** hands straight off to Ollama.
   - If an **Ollama response takes more than one minute**, extraction falls back to the heuristic extractor.

This is completely document-agnostic: no facts, filenames, schemas, or document-specific regex patterns are hard-coded.

### Comparison and Reasoning

Facts sharing a semantic concept are paired and classified as:

- **Corroborated** — same concept, values agree within rounding
- **Contradiction** — same concept, values disagree
- **Reconciled by context** — values differ but the gap is explained by period, units, basis, or scope
- **Different scope** — facts describe different subjects entirely

The local model writes the human-readable explanation. A heuristic fallback covers the offline case.

### Dynamic Four Cases

The four required demonstration cases are **not hard-coded**. They are selected from the relationships actually produced during the current run, so the demo always reflects real extracted evidence.

### AI Tools Used

- **Ollama granite4.1:3b** — local LLM for extraction and reasoning (no API key required)
- **PyMuPDF** — PDF text extraction
- **Streamlit** — upload-and-inspect interface
- **Python stdlib** — no external LLM SDKs needed

## Limitations and Next Steps

- **3b model constraints**: granite4.1:3b sometimes produces shallow or noisy extractions and may miss implicit relationships. A larger model (e.g. 8-13B) would improve coverage.
- **Period inference**: the model sometimes omits the `period` field; this is patched automatically via regex inference on the subject text.
- **Slow per-chunk**: each chunk takes ~5-15s on a local 3b model, so a 30-page PDF takes a few minutes. Requests are made serially on purpose so a free/quota-limited model is never hammered; an Ollama call that exceeds `OLLAMA_CALL_TIMEOUT` (default 1 minute) automatically falls back to the heuristic extractor.
- **No incremental ingestion**: facts are re-extracted from scratch each time; a persistent store would help for large document sets.

Next steps:
- Upgrade to a larger Ollama model (e.g. `llama3.1:8b` or `mistral:7b`)
- Add a persistent fact store with incremental ingestion
- Add graph visualisation of concept relationships
- Add user feedback loop for extraction quality

## Additional Notes

The system is built to generalise beyond the starter documents. It accepts any PDF through the UI, uses no hard-coded facts or document-specific rules, and dynamically derives the four required cases from the extracted evidence. Every fact is grounded in a source snippet and compared with reasoning — not just raw numeric matching.
