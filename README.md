# Superjoin — PDF Fact Knowledge Layer

A Streamlit application that extracts grounded facts from PDFs, compares related facts across documents, and presents evidence-backed relationship checks. It ships with a starter dataset and also accepts external PDF uploads.

The application uses a resilient model chain:

1. Gemini, when a configured Gemini model accepts a live probe request.
2. Local Ollama, when the configured local model accepts a live probe request.
3. A generic offline heuristic extractor when neither model can serve the request.

## Features

- Extracts text from PDFs with PyMuPDF and processes it in sentence-aware chunks.
- Produces grounded facts containing a source, page number, value, unit, period, basis, concept, confidence, and evidence snippet.
- Repairs JSON arrays, individual objects, and concatenated JSON objects returned by models.
- Compares facts as corroborated, contradictory, reconciled by context, or different in scope.
- Selects the four demonstration cases from the relationships actually found in the current document set.
- Shows the active backend in the sidebar and updates it as model responses arrive.
- Groups live extraction output by page, with the latest processed page first; the live output is removed when final results are ready.
- Persists dataset and upload fact caches, including compatibility with the previous cache-key format.

## Requirements

- Python 3.10 or later
- One of the following is optional but recommended for model-assisted extraction:
  - A Gemini API key
  - [Ollama](https://ollama.com/) with a pulled model, such as `granite4.1:3b`

Without either model backend, the app still works with the offline heuristic extractor.

## Install and run

```bash
git clone https://github.com/tarun-ainampudi/superjoin-pdf-analyser.git
cd superjoin-pdf-analyser

python -m venv .venv
.venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

The app loads the PDFs in `dataset/` by default. Use the sidebar uploader to analyse one or more external PDFs instead.

## Model setup

### Gemini

Set `GEMINI_API_KEY` in your environment or in a local `.env` file:

```env
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-flash-latest
```

Gemini availability is not inferred from the presence of a key. The application sends a short `generateContent` probe to the configured model and treats an HTTP `200` response as available.

### Ollama

Install Ollama, then pull and serve the model:

```bash
ollama pull granite4.1:3b
ollama serve
```

Ollama availability is also verified with a short `/api/chat` request to the configured model. A running server without the requested model is therefore treated as unavailable.

## Backend selection and fallbacks

For a fresh extraction, Gemini is preferred. If it cannot respond, the app tries Ollama. If both fail, or an Ollama response times out, it extracts generic numerical facts locally.

- A Gemini `400`, `401`, `403`, `404`, or `429` disables Gemini for the current run and proceeds to Ollama.
- Ollama applies the configured retry budget; after it is exhausted, the rest of the document uses the heuristic path rather than repeatedly waiting for failed requests.
- A fully cached document set does not call Gemini or Ollama merely to regenerate relationship explanations. It uses the deterministic comparison heuristic so cached loads remain fast.

The sidebar reports the backend that actually supplied the current result: Gemini, Ollama, cache, or the generic heuristic.

## Caching

Fact extraction is cached as JSON under `.cache/`:

```text
.cache/
├── dataset/                 # Persistent cache for the bundled starter PDFs
└── session/<session-id>/    # Cache for PDFs uploaded during one app session
```

Cache keys use a SHA-256 hash of the PDF content, so the same upload can be reused even when Streamlit writes it to a different temporary directory. Cache writes are atomic, corrupted entries are discarded safely, and an empty extraction result is still a valid cache hit.

Existing cache files written by the earlier filename/size/mtime/path key are read when the original PDF remains at the same location, then migrated automatically to the content-hash key.

## Configuration

All settings are optional environment variables. Values below are defaults.

| Variable | Default | Description |
|---|---:|---|
| `GEMINI_API_KEY` | empty | Gemini API key; enables Gemini probing and requests. |
| `GEMINI_MODEL` | `gemini-flash-latest` | Gemini model used for probing and generation. |
| `GEMINI_API_URL` | Google Generative Language v1beta URL | Gemini API base URL. |
| `GEMINI_TIMEOUT` | `180` | Maximum seconds for a full Gemini generation request. |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_MODEL` | `granite4.1:3b` | Local model used for probing and generation. |
| `OLLAMA_RETRIES` | `3` | Retry count for failed Ollama chat requests. |
| `OLLAMA_CALL_TIMEOUT` | `60` | Maximum seconds for one Ollama generation request. |
| `MODEL_PROBE_TIMEOUT` | `15` | Maximum seconds for each lightweight model availability probe. |
| `EXTRACTION_CHUNK_CHARS` | `1200` | Maximum characters in one extraction prompt chunk. |
| `MAX_FACTS_PER_CHUNK` | `40` | Maximum accepted facts from a model response chunk. |
| `MAX_PAGES_PER_PDF` | `20` | Maximum pages processed for each PDF. |
| `RELATION_BUDGET` | `40` | Maximum model-assisted relationship checks per run. |

Invalid numeric values use defaults; non-positive values are coerced to safe positive limits.

## Project structure

```text
superjoin-pdf-analyser/
├── app.py                              # Streamlit user interface
├── src/knowledge_layer/
│   ├── cache.py                        # Content-addressed cache and migration
│   ├── comparison.py                   # Fact relationship classification
│   ├── config.py                       # Environment configuration
│   ├── dataset.py                      # Starter-PDF discovery
│   ├── fact_extraction.py              # Extraction orchestration and fallback
│   ├── heuristic.py                    # Offline generic numeric extractor
│   ├── llm.py                          # Gemini/Ollama clients and probes
│   ├── models.py                       # Fact data model
│   ├── normalization.py                # Value, unit, period, and concept helpers
│   └── text_extraction.py              # PyMuPDF extraction and chunking
├── dataset/                            # Delhivery and India macroeconomy PDFs
├── tests/                              # Cache and backend regression tests
├── requirements.txt
└── README.md
```

## Testing

Run the test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The tests cover backend-probe caching, Gemini-to-Ollama fallback, timeout handling, cache key portability, corrupt-cache recovery, legacy-cache migration, and cache backend reporting.

## Limitations

- PDF quality, tables, and scanned pages can affect text extraction and fact quality.
- The heuristic fallback focuses on generic numeric patterns and is less precise than a working model backend.
- Facts and relationships should be reviewed against their displayed evidence before being used for important decisions.
- The app processes model requests serially to reduce local load and API quota pressure; large fresh PDFs can therefore take time.
