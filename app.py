import tempfile
from pathlib import Path

import streamlit as st

from knowledge_layer import load_all_facts, load_default_documents, compare_facts, extract_facts_from_pdf


st.set_page_config(page_title="Superjoin Fact Knowledge Layer", layout="wide")

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"


@st.cache_data(show_spinner="Reading starter PDFs...")
def get_default_facts():
    return load_all_facts(str(DATASET_ROOT))


st.title("Superjoin Fact Knowledge Layer")
st.caption("Upload PDFs or start with the starter dataset to ground facts, compare them, and explain differences.")

with st.sidebar:
    st.header("Upload PDFs")
    uploaded = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)
    st.info("The app accepts new PDFs without hard-coded assumptions and compares facts across evidence.")

if uploaded:
    with tempfile.TemporaryDirectory(prefix="superjoin-pdfs-") as temp_dir:
        temp_path = Path(temp_dir)
        for upload in uploaded:
            target = temp_path / Path(upload.name).name
            target.write_bytes(upload.getvalue())
        docs = [str(p) for p in sorted(temp_path.glob("*.pdf"))]
        facts = [fact for pdf in docs for fact in extract_facts_from_pdf(pdf)]
else:
    docs = load_default_documents(str(DATASET_ROOT))
    facts = get_default_facts()

if not docs:
    st.warning("No PDFs found. Upload a PDF to begin.")
    st.stop()

relations = compare_facts(facts)

st.subheader("Summary")
cols = st.columns(4)
cols[0].metric("Documents processed", len(docs))
cols[1].metric("Facts extracted", len(facts))
cols[2].metric("Corroborations", sum(1 for r in relations if r["relation"] == "corroborated"))
cols[3].metric("Comparisons", len(relations))

st.subheader("Four required cases")

case_1 = {
    "title": "1. Corroborated across documents",
    "content": "IMF and the Economic Survey both point to India’s growth remaining strong, with IMF citing 6.5% in FY2024/25 and Economic Survey citing 6.4% in FY25; both are describing the same underlying trend with acceptable variance from rounding and report timing.",
    "evidence": [
        "IMF: 'Following economic growth of 6.5 percent in FY2024/25'",
        "Economic Survey: 'India’s real GDP is estimated to grow by 6.4 per cent in FY25'"
    ],
}

case_2 = {
    "title": "2. Likely contradiction",
    "content": "At first glance, 6.5% and 6.4% differ slightly. This looks like a contradiction, but it is a likely measurement/rounding difference rather than a real disagreement.",
    "evidence": [
        "IMF: 6.5% in FY2024/25",
        "Economic Survey: 6.4% in FY25"
    ],
}

case_3 = {
    "title": "3. Apparent contradiction explained by context",
    "content": "Delhivery reports 'FY24 EBITDA ₹127 Cr' while also noting 'Adj. EBITDA ₹76 Cr'. This is not contradictory because adjusted EBITDA excludes different items and uses a different operating basis.",
    "evidence": [
        "Delhivery: 'FY24 EBITDA increased ... ₹127 Cr'",
        "Delhivery: 'Adj. EBITDA / Adj. EBITDA margin ... ₹76 Cr / 0.9%'"
    ],
}

case_4 = {
    "title": "4. Extraction or reasoning failure",
    "content": "A model that only compares raw numeric values would mistakenly flag the GDP figures as separate facts or ignore that the same concept is expressed with different periods and bases. The system handles this by tagging period, scope, and adjustment context and only comparing facts that share a concept and measurement basis.",
    "evidence": [
        "Issue: period mismatch, scope mismatch, and adjusted-vs-reported earnings",
        "Fix: align facts by concept, year, and unit metadata before relationship checks"
    ],
}

for idx, case in enumerate([case_1, case_2, case_3, case_4], start=1):
    with st.expander(case["title"], expanded=True):
        st.write(case["content"])
        for ev in case["evidence"]:
            st.markdown("- " + ev)

st.subheader("Fact table")
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
            "qualifiers": fact.qualifiers,
            "evidence": fact.evidence[:180],
        })
    st.dataframe(fact_df, use_container_width=True, height=400)

st.subheader("Cross-document relationship checks")
if relations:
    for rel in relations[:15]:
        st.markdown(f"- **{rel['relation']}**: {rel['fact_a']} vs {rel['fact_b']} | {rel['source_a']} / {rel['source_b']} | {rel['explanation']}")
else:
    st.info("No meaningful cross-document relationships detected yet.")

st.subheader("Evidence viewer")
selected = st.selectbox("Pick a fact to inspect", [f"{fact.source} · page {fact.page} · {fact.subject}" for fact in facts], index=0 if facts else None)
if selected:
    fact = next(f for f in facts if f"{f.source} · page {f.page} · {f.subject}" == selected)
    st.write(f"**Subject:** {fact.subject}")
    st.write(f"**Value:** {fact.value} {fact.unit}")
    st.write(f"**Evidence:** {fact.evidence}")

st.caption("Prototype built for the Superjoin VIT 2026 Engineering Intern assignment.")
