import logging
import sys
import tempfile
import uuid
import hashlib
from pathlib import Path

import streamlit as st

# Ensure src is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.knowledge_layer import (
    get_active_backend,
    get_used_backend,
    reset_availability,
    extract_facts_from_pdf,
    compare_facts,
    select_four_cases,
    load_default_documents,
)
from src.knowledge_layer.cache import get_dataset_cache_dir, get_session_cache_dir
from src.knowledge_layer.config import GEMINI_MODEL, MAX_PAGES_PER_PDF, OLLAMA_MODEL

st.set_page_config(page_title="Superjoin Fact Knowledge Layer", layout="wide")

ROOT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = ROOT_DIR / "dataset"
DATASET_CACHE_DIR = get_dataset_cache_dir(ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if "session_id" not in st.session_state:
    st.session_state["session_id"] = uuid.uuid4().hex
    logging.info("Initialized app session %s", st.session_state["session_id"])

if "model_events" not in st.session_state:
    st.session_state["model_events"] = []

SESSION_CACHE_DIR = get_session_cache_dir(ROOT_DIR, st.session_state["session_id"])
logging.info("Dataset cache dir: %s", DATASET_CACHE_DIR)
logging.info("Session cache dir: %s", SESSION_CACHE_DIR)

# Re-probe each model's availability once per run so the UI reflects backend
# changes (e.g. Ollama started/stopped). Within a run the results are cached,
# so a model is never probed more than once per run.
reset_availability()


def _backend_label(backend_name: str) -> str:
    if backend_name == "gemini":
        return f"Gemini API ({GEMINI_MODEL})"
    if backend_name == "ollama":
        return f"Ollama ({OLLAMA_MODEL})"
    if backend_name == "cache":
        return "Loaded from cache"
    return "Generic heuristic"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    backend_status = st.empty()
    st.info(
        "Upload any PDF. "
        "The system automatically extracts grounded facts, compares them "
        "across documents, and shows the four required cases."
    )


def _render_backend_status() -> None:
    # The sidebar is created before extraction. Updating its placeholder after
    # loading prevents it from showing the previous run's backend.
    backend_name = st.session_state.get("used_backend", get_used_backend())
    backend_status.success(f"Backend: {_backend_label(backend_name)}")


_render_backend_status()

# ── Data loading ─────────────────────────────────────────────────────────────
if uploaded:
    # Include content, not just name and size: distinct PDFs can share both.
    upload_key = tuple(sorted(
        (u.name, u.size, hashlib.sha256(u.getvalue()).hexdigest()) for u in uploaded
    ))
    with tempfile.TemporaryDirectory(prefix="superjoin-pdfs-") as temp_dir:
        temp_path = Path(temp_dir)
        for upload in uploaded:
            target = temp_path / Path(upload.name).name
            target.write_bytes(upload.getvalue())
        docs = [str(p) for p in sorted(temp_path.glob("*.pdf"))]

        if st.session_state.get("upload_key") != upload_key:
            st.session_state["model_events"] = []
            progress = st.progress(0.0, text="Extracting facts from uploaded PDFs...")
            log_placeholder = st.empty()
            n_docs = len(docs)
            counter = {"doc": 0}

            def on_extract(_stage, done, total):
                progress.progress(
                    min(1.0, (counter["doc"] + done / max(1, total)) / max(1, n_docs)),
                    text=f"Extracting page {done}/{total} of document {counter['doc'] + 1}/{n_docs}...",
                )

            def on_model_status(pdf_name: str, page: int, raw: str):
                # call_model records the backend that produced this exact
                # response. Persist it and repaint the sidebar immediately,
                # rather than waiting for the full PDF to complete.
                backend = get_used_backend() or "heuristic"
                st.session_state["used_backend"] = backend
                _render_backend_status()
                page_name = Path(pdf_name).name
                events = st.session_state["model_events"]
                # One PDF page can contain multiple extraction chunks. Keep
                # those responses together and move the just-updated page to
                # the top instead of rendering duplicate page rows.
                event = next(
                    (item for item in events if item["pdf_name"] == page_name and item["page"] == page),
                    None,
                )
                if event is None:
                    event = {"pdf_name": page_name, "page": page, "backend": backend, "responses": []}
                else:
                    events.remove(event)
                    event["backend"] = backend
                event["responses"].append(raw[:1200])
                events.insert(0, event)
                with log_placeholder.container():
                    for item in events:
                        st.markdown(
                            f"**{item['pdf_name']}** — page {item['page']} — "
                            f"**{item['backend']}** ({len(item['responses'])} chunk(s))"
                        )
                        st.code("\n\n".join(item["responses"]) or "No response returned", language=None)

            session_facts = []
            document_backends = []
            for pdf_path in docs:
                logging.info("Processing uploaded PDF %s with session cache %s", pdf_path, SESSION_CACHE_DIR)
                session_facts.extend(
                    extract_facts_from_pdf(
                        pdf_path,
                        progress_cb=on_extract,
                        max_pages=MAX_PAGES_PER_PDF,
                        cache_dir=SESSION_CACHE_DIR,
                        status_cb=on_model_status,
                    )
                )
                used_backend = get_used_backend() or get_active_backend()
                document_backends.append(used_backend)
                st.session_state["used_backend"] = used_backend
                counter["doc"] += 1
            progress.empty()
            st.session_state["upload_key"] = upload_key
            st.session_state["facts"] = session_facts
            # Cache hits already contain extracted facts. Do not turn a fast
            # cached load into a slow Gemini request merely to regenerate
            # explanations; the deterministic relationship heuristic is used.
            all_from_cache = bool(document_backends) and all(
                backend == "cache" for backend in document_backends
            )
            st.session_state["relations"] = compare_facts(
                session_facts,
                use_llm=False if all_from_cache else get_active_backend() != "none",
            )
            # The per-page model output is only a live processing aid. Remove
            # it once the final knowledge-layer output is ready.
            log_placeholder.empty()
            st.session_state["model_events"] = []
            st.success(f"Finished processing {len(docs)} uploaded document(s) using {st.session_state.get('used_backend', get_active_backend())}.")
        facts = st.session_state.get("facts", [])
else:
    docs = load_default_documents(str(DATASET_ROOT))
    if not st.session_state.get("default_loaded"):
        st.session_state["model_events"] = []
        progress = st.progress(0.0, text="Extracting facts from starter PDFs...")
        log_placeholder = st.empty()
        n_docs = len(docs)
        counter = {"doc": 0}

        def on_extract(_stage, done, total):
            progress.progress(
                min(1.0, (counter["doc"] + done / max(1, total)) / max(1, n_docs)),
                text=f"Extracting page {done}/{total} of document {counter['doc'] + 1}/{n_docs}...",
            )

        def on_model_status(pdf_name: str, page: int, raw: str):
            backend = get_used_backend() or "heuristic"
            st.session_state["used_backend"] = backend
            _render_backend_status()
            page_name = Path(pdf_name).name
            events = st.session_state["model_events"]
            event = next(
                (item for item in events if item["pdf_name"] == page_name and item["page"] == page),
                None,
            )
            if event is None:
                event = {"pdf_name": page_name, "page": page, "backend": backend, "responses": []}
            else:
                events.remove(event)
                event["backend"] = backend
            event["responses"].append(raw[:1200])
            events.insert(0, event)
            with log_placeholder.container():
                for item in events:
                    st.markdown(
                        f"**{item['pdf_name']}** — page {item['page']} — "
                        f"**{item['backend']}** ({len(item['responses'])} chunk(s))"
                    )
                    st.code("\n\n".join(item["responses"]) or "No response returned", language=None)

        session_facts = []
        document_backends = []
        for pdf_path in docs:
            logging.info("Processing default dataset PDF %s with dataset cache %s", pdf_path, DATASET_CACHE_DIR)
            session_facts.extend(
                extract_facts_from_pdf(
                    pdf_path,
                    progress_cb=on_extract,
                    max_pages=MAX_PAGES_PER_PDF,
                    cache_dir=DATASET_CACHE_DIR,
                    status_cb=on_model_status,
                )
            )
            used_backend = get_used_backend() or get_active_backend()
            document_backends.append(used_backend)
            st.session_state["used_backend"] = used_backend
            counter["doc"] += 1
        progress.empty()
        st.session_state["default_loaded"] = True
        st.session_state["facts"] = session_facts
        # Cached extraction should not make any external model calls. Use the
        # deterministic comparison path until at least one PDF is freshly
        # extracted in this run.
        all_from_cache = bool(document_backends) and all(
            backend == "cache" for backend in document_backends
        )
        st.session_state["relations"] = compare_facts(
            session_facts,
            use_llm=False if all_from_cache else get_active_backend() != "none",
        )
        log_placeholder.empty()
        st.session_state["model_events"] = []
        st.success(f"Finished processing {len(docs)} starter document(s) using {st.session_state.get('used_backend', get_active_backend())}.")
    facts = st.session_state.get("facts", [])

if not docs:
    st.warning("No PDFs found. Upload a PDF to begin.")
    st.stop()

# Refresh the already-rendered sidebar with the path that actually supplied
# facts in this run (cache, Gemini, Ollama, or the heuristic fallback).
_render_backend_status()

relations = st.session_state.get("relations", [])
four_cases = select_four_cases(relations)

# ── Header metrics ───────────────────────────────────────────────────────────
st.title("Superjoin Fact Knowledge Layer")
st.caption("Upload PDFs or start with the starter dataset to ground facts, compare them, and explain differences.")

cols = st.columns(4)
cols[0].metric("Documents processed", len(docs))
cols[1].metric("Facts extracted", len(facts))
cols[2].metric("Corroborations", sum(1 for r in relations if r["relation"] == "corroborated"))
cols[3].metric("Relationships found", len(relations))

# ── Four required cases (dynamically produced) ───────────────────────────────
st.subheader("Four required cases (derived from actual extraction results)")

for case in four_cases:
    with st.expander(case["title"], expanded=True):
        st.write(case["content"])
        for ev in case["evidence"]:
            st.markdown(f"- `{ev}`")
        if not case["evidence"]:
            st.caption("No example found in the current document set.")

# ── Fact table ───────────────────────────────────────────────────────────────
st.subheader("Extracted facts")
if facts:
    fact_df = []
    for fact in facts:
        fact_df.append({
            "source": fact.source,
            "page": fact.page,
            "concept": fact.concept,
            "subject": fact.subject,
            "value": fact.value,
            "unit": fact.unit,
            "period": fact.period,
            "basis": fact.basis,
            "evidence": fact.evidence[:180],
            "by": fact.extracted_by,
        })
    st.dataframe(fact_df, width="stretch", height=400)

# ── Relationships ────────────────────────────────────────────────────────────
st.subheader("Cross-document relationship checks")
if relations:
    for rel in relations[:20]:
        label = rel["relation"].replace("_", " ").title()
        st.markdown(
            f"- **{label}**: {rel['fact_a']}  vs  {rel['fact_b']}  \n"
            f"  _{rel['explanation']}_"
        )
else:
    st.info("No meaningful cross-document relationships detected yet.")

# ── Evidence viewer ──────────────────────────────────────────────────────────
st.subheader("Evidence viewer")
if facts:
    labels = [f"{f.source}  p{f.page}  {f.subject}" for f in facts]
    selected_label = st.selectbox("Pick a fact to inspect", labels)
    if selected_label:
        fact = next(f for f in facts if f"{f.source}  p{f.page}  {f.subject}" == selected_label)
        st.write(f"**Subject:** {fact.subject}")
        st.write(f"**Value:** {fact.value} {fact.unit}")
        st.write(f"**Period:** {fact.period or 'not specified'}")
        st.write(f"**Basis:** {fact.basis or 'reported'}")
        st.write(f"**Source:** {fact.source}, page {fact.page}")
        st.code(fact.evidence, language=None)

st.caption("Prototype built for the Superjoin VIT 2026 Engineering Intern assignment.")
