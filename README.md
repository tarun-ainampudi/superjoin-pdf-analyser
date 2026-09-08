# Superjoin VIT 2026 — Fact Knowledge Layer

This project implements a lightweight fact knowledge layer for PDF documents. It extracts numerical and semantic facts from PDFs, grounds them in source evidence, and compares them across documents to identify corroboration, contradiction, and contextual reconciliation.

## Setup and Run Instructions

1. Clone or open this repository.
2. Create a Python environment and install dependencies:

```bash
cd superjoin-pdf-analyser
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal (usually http://localhost:8501).
5. Upload one or more PDFs or use the starter dataset under the `dataset/` folder.
6. The app will display extracted facts, source evidence, and cross-document relationship checks.

## Video Demo

A short demo video is required for the assignment. This environment does not include a GitHub account or video-hosting setup, so the final external video link must be created and added after recording locally.

Suggested demo flow (under 3 minutes):

1. Open the app.
2. Upload a PDF or open the dataset.
3. Show the extracted facts table.
4. Explain the four required cases:
   - corroborated fact across documents
   - likely contradiction
   - context-based reconciliation
   - extraction/logic failure and mitigation
5. Show the evidence snippets used for reasoning.

## Approach

### Core idea
The application treats each fact as a grounded statement with:

- source document
- page number
- subject/concept
- value and unit
- evidence snippet
- qualifiers such as date, period, or scope

This lets the system explain why a fact is believed, where it came from, and how it compares with other facts.

### Architecture

- `knowledge_layer.py` handles PDF extraction, fact normalization, and cross-document comparison.
- `app.py` provides the Streamlit UI for upload, inspection, and explanation.
- `dataset/` contains starter PDFs for the assignment.

### Important decisions

- We extracted facts using a hybrid rule-based approach rather than a fully hard-coded document schema.
- Numeric metrics are normalized conservatively and grouped by concept to reduce false mismatches.
- Relationship analysis checks whether facts are corroborated, contradictory, or only different because of different periods, units, or adjustments.

### AI tools and libraries used

- Python
- PyMuPDF for PDF text extraction
- Streamlit for a simple upload-and-inspect interface
- Lightweight regex-driven fact extraction and reasoning heuristics

## Limitations and Next Steps

This is a prototype, not a production-grade knowledge graph. The current implementation still has limits:

- it relies on pattern-based extraction and may miss unusual fact formats
- not all ambiguous statements are resolved automatically
- relationship grouping is heuristic and can be improved with entity linking and richer metadata

Next, I would add:

- a real semantic parser or LLM-based fact extractor
- better entity normalization and resolution
- a database-backed fact store for incremental PDF ingestion
- richer graph relationships and confidence scores

## Additional Notes

The assignment emphasizes not just extraction, but grounded reasoning. The strongest part of this project is that every fact is associated with evidence from the source PDF and compared in context, rather than treated as a disconnected number.

The system is built to generalize beyond the starter documents and can accept new PDFs through the UI without relying on hard-coded facts or names.
