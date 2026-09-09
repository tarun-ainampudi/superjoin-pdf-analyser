import sys
import tempfile
from pathlib import Path

import streamlit as st

# Ensure src is importable when running from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.knowledge_layer import (
    ollama_available,
    extract_facts_from_pdf,
    compare_facts,
    select_four_cases,
    load_all_facts,
    load_default_documents,
)
from src.knowledge_layer.config import MAX_PAGES_PER_PDF

st.set_page_config(page_title="Superjoin Fact Knowledge Layer", layout="wide")

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    backend = "Ollama (granite4.1:3b)" if ollama_available() else "Generic heuristic"
    st.success(f"Backend: {backend}")
    st.info(
        "Upload any PDF or use the starter dataset. "
        "The system automatically extracts grounded facts, compares them "
        "across documents, and shows the four required cases."
    )

# ── Data loading ─────────────────────────────────────────────────────────────
if uploaded:
    # Cache extraction per unique set of uploaded files (name+size fingerprint).
    upload_key = tuple(sorted(f"{u.name}|{u.size}" for u in uploaded))
    with tempfile.TemporaryDirectory(prefix="superjoin-pdfs-") as temp_dir:
        temp_path = Path(temp_dir)
        for upload in uploaded:
            target = temp_path / Path(upload.name).name
            target.write_bytes(upload.getvalue())
        docs = [str(p) for p in sorted(temp_path.glob("*.pdf"))]

        if st.session_state.get("upload_key") != upload_key:
            progress = st.progress(0.0, text="Extracting facts from uploaded PDFs...")
            n_docs = len(docs)
            counter = {"doc": 0}

            def on_extract(_stage, done, total):
                progress.progress(
                    min(1.0, (counter["doc"] + done / max(1, total)) / max(1, n_docs)),
                    text=f"Extracting page {done}/{total} of document {counter['doc'] + 1}/{n_docs}...",
                )

            session_facts = []
            for pdf_path in docs:
                session_facts.extend(extract_facts_from_pdf(pdf_path, progress_cb=on_extract,
                                                            max_pages=MAX_PAGES_PER_PDF))
                counter["doc"] += 1
            progress.empty()
            st.session_state["upload_key"] = upload_key
            st.session_state["facts"] = session_facts
            st.session_state["relations"] = compare_facts(session_facts, use_llm=ollama_available())
        facts = st.session_state.get("facts", [])
else:
    docs = load_default_documents(str(DATASET_ROOT))
    if not st.session_state.get("default_loaded"):
        progress = st.progress(0.0, text="Extracting facts from starter PDFs...")
        n_docs = len(docs)
        counter = {"doc": 0}

        def on_extract(_stage, done, total):
            progress.progress(
                min(1.0, (counter["doc"] + done / max(1, total)) / max(1, n_docs)),
                text=f"Extracting page {done}/{total} of document {counter['doc'] + 1}/{n_docs}...",
            )

        session_facts = []
        for pdf_path in docs:
            session_facts.extend(extract_facts_from_pdf(pdf_path, progress_cb=on_extract,
                                                        max_pages=MAX_PAGES_PER_PDF))
            counter["doc"] += 1
        progress.empty()
        st.session_state["default_loaded"] = True
        st.session_state["facts"] = session_facts
        st.session_state["relations"] = compare_facts(session_facts, use_llm=ollama_available())
    facts = st.session_state.get("facts", [])

if not docs:
    st.warning("No PDFs found. Upload a PDF to begin.")
    st.stop()

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
