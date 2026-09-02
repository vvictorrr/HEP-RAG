"""
app.py — HEP-RAG Streamlit frontend.

Four tabs: Ask (default), Corpus Browser, Benchmark & Ablation Results,
How It Works. Design intent: a casual visitor gets a clean answer in one
click; an ML engineer reading closely can go three clicks deep into
retrieval internals and hallucination signal justifications.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from retriever import HybridRetriever, ALL_MODES
from generator import Generator
from hallucination_detector import HallucinationDetector

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"

# Zero-config demo: if the pre-built index isn't present (e.g. fresh Streamlit
# Cloud deploy), fetch it once rather than forcing every visitor to wait on
# a full re-ingest. No-op if data/chroma_db already exists (committed index).
if not (DATA_DIR / "chroma_db").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from download_index import download_index
        download_index()
    except Exception as e:
        print(f"download_index() skipped/failed: {e}")

st.set_page_config(
    page_title="HEP-RAG | Physics Literature Q&A",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.hep-header {
    background: #1B2A4A;
    color: #FFFFFF;
    padding: 1.1rem 1.6rem;
    border-radius: 10px;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.hep-header h1 { font-size: 1.3rem; margin: 0; font-weight: 600; }
.hep-header span { font-size: 0.85rem; opacity: 0.75; }

.support-label {
    display: inline-block;
    padding: 0.05rem 0.55rem;
    border-radius: 5px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.support-ok { background: #DCF3E1; color: #146C2E; border: 1px solid #A9E3B7; }
.support-uncertain { background: #FDF3D3; color: #8A6100; border: 1px solid #F3D97F; }
.support-flag { background: #FBDEDE; color: #A31414; border: 1px solid #F3AFAF; }

.chunk-card {
    border: 1px solid #E3E7F0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.6rem;
    background: #FAFBFD;
}
.arxiv-badge {
    display: inline-block;
    background: #EEF1F8;
    color: #1B2A4A;
    padding: 0.1rem 0.5rem;
    border-radius: 12px;
    font-size: 0.75rem;
    text-decoration: none;
    border: 1px solid #CDD5E8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """<div class="hep-header"><h1>⚛ HEP-RAG — Physics Literature Q&A</h1>
    <span>Dark matter searches at the LHC · citation-grounded · dual-signal hallucination detection</span></div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### API Key")
    st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        key="user_api_key",
        help="Get one at console.anthropic.com. Your questions are answered "
             "using this key, billed to your own account.",
    )
    st.caption(
        "Not stored or logged anywhere — held only in this browser session's "
        "memory and sent directly to Anthropic's API to answer your questions. "
        "If left blank, the app uses the deployer's own key if one is "
        "configured, or won't be able to answer until you provide one."
    )


# ---------------------------------------------------------------------------
# Cached resource loading — models load once per server process, not per query
# ---------------------------------------------------------------------------
#
# BRING-YOUR-OWN-KEY DESIGN NOTE: st.cache_resource's cache is shared across
# ALL visitors to this app on the same server process — it is not per-session.
# If load_generator()/load_detector() took no arguments, the first visitor's
# API key would get baked into the cached client and silently reused for
# every subsequent visitor, regardless of what key *they* entered. To avoid
# this, both functions below take api_key as an explicit parameter, so
# Streamlit's cache correctly keys on it — a different key gets a different
# cached client, and the same key (e.g. two visitors who both leave the
# field blank and fall through to the deployer's own default, if configured)
# correctly shares one. The expensive local models (retriever, NLI,
# reranker) are NOT API-key-dependent and are loaded once, shared by
# everyone, regardless of whose key is active — see _load_detector_models().
def _get_default_api_key() -> str | None:
    """The deployer's own key, if they configured one in secrets/env — used
    only as a fallback when a visitor hasn't entered their own key. On a
    public deployment, the deployer may deliberately leave this unset so
    that every visitor must supply their own key (see README's deployment
    notes)."""
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def get_active_api_key() -> str | None:
    """Whatever key should actually be used for this session: the visitor's
    own entry (sidebar), if provided, else the deployer's default, if any."""
    user_key = st.session_state.get("user_api_key", "").strip()
    return user_key or _get_default_api_key()


@st.cache_resource
def load_retriever():
    return HybridRetriever(data_dir=str(DATA_DIR))


@st.cache_resource
def _load_detector_models():
    """The NLI cross-encoder and relevance-scoring cross-encoder are large,
    slow-to-load, and completely independent of any API key — load them
    exactly once, shared across every visitor regardless of whose key is
    active, rather than reloading per distinct key."""
    from sentence_transformers import CrossEncoder
    from transformers import pipeline
    nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base", top_k=None)
    relevance_scorer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return nli, relevance_scorer


@st.cache_resource
def load_generator(api_key: str):
    return Generator(api_key=api_key)


@st.cache_resource
def load_detector(api_key: str):
    nli, relevance_scorer = _load_detector_models()
    return HallucinationDetector(api_key=api_key, nli=nli, relevance_scorer=relevance_scorer)


@st.cache_data
def load_corpus_metadata():
    path = DATA_DIR / "corpus_metadata.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


@st.cache_data
def load_ablation_results():
    retrieval_path = EVAL_DIR / "retrieval_ablation_results.json"
    hallucination_path = EVAL_DIR / "hallucination_ablation_results.json"
    retrieval = json.loads(retrieval_path.read_text()) if retrieval_path.exists() else None
    hallucination = json.loads(hallucination_path.read_text()) if hallucination_path.exists() else None
    return retrieval, hallucination


LABEL_CLASS = {"SUPPORTED": "support-ok", "UNCERTAIN": "support-uncertain", "FLAGGED": "support-flag"}
LABEL_ICON = {"SUPPORTED": "✅", "UNCERTAIN": "⚠️", "FLAGGED": "❌"}

SUGGESTED_QUESTIONS = [
    "What MET threshold is used in the mono-jet signal region?",
    "How is pileup corrected for jets?",
    "What triggers are used for MET-based searches?",
]

tab_ask, tab_corpus, tab_bench, tab_how = st.tabs(
    ["Ask", "Corpus Browser", "Benchmark & Ablation Results", "How It Works"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Ask
# ---------------------------------------------------------------------------
with tab_ask:
    if "query" not in st.session_state:
        st.session_state.query = ""

    st.markdown("**Ask anything about dark matter searches at the LHC**")
    col_input, col_button = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "query", value=st.session_state.query,
            placeholder='e.g. "What MET threshold is used in mono-jet event selection?"',
            label_visibility="collapsed",
        )
    with col_button:
        search_clicked = st.button("Search ▶", use_container_width=True)

    st.markdown("**Try:**")
    sq_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, sq in zip(sq_cols, SUGGESTED_QUESTIONS):
        if col.button(sq, use_container_width=True):
            st.session_state.query = sq
            st.rerun()

    run_query = query if (search_clicked and query) else None

    if run_query:
        st.session_state.query = run_query
        data_ready = (DATA_DIR / "chunks.jsonl").exists()
        active_api_key = get_active_api_key()
        if not data_ready:
            st.error(
                "No ingested corpus found in data/. Run `python src/ingest.py` first "
                "(see REPRODUCIBILITY.md), or download the pre-built index."
            )
        elif not active_api_key:
            st.warning(
                "This app needs an Anthropic API key to generate answers. "
                "Enter your own key in the sidebar (← left) to continue — "
                "get one at [console.anthropic.com](https://console.anthropic.com)."
            )
        else:
            with st.status("Searching corpus...", expanded=True) as status:
                st.write("Running hybrid retrieval (BM25 + SPECTER2)...")
                retriever = load_retriever()
                chunks = retriever.retrieve(run_query, k_final=5, mode="hybrid_rerank")

                st.write(f"Reranking {len(chunks)} candidates...")
                generator = load_generator(active_api_key)
                gen_result = generator.generate(run_query, chunks)

                detector = load_detector(active_api_key)

                if gen_result.corpus_insufficient:
                    status.update(label="Done — corpus insufficient", state="complete")
                else:
                    st.write("Checking for hallucinations (dual-signal)...")
                    detections = []
                    for s in gen_result.sentences:
                        # Resolve the exact cited chunk by index — deterministic, no ambiguity
                        # even when multiple retrieved chunks share the same (paper_id, section).
                        if s.chunk_index and 1 <= s.chunk_index <= len(chunks):
                            cited_text = chunks[s.chunk_index - 1].text
                        else:
                            cited_text = chunks[0].text if chunks else ""
                        detections.append(detector.detect(s.sentence, cited_text))
                    status.update(label="Done", state="complete")

            st.markdown("---")
            if gen_result.corpus_insufficient:
                st.warning(f"**Not answerable from corpus.** {gen_result.reason or ''}")
            else:
                col_answer, col_summary = st.columns([3, 1])
                with col_answer:
                    st.markdown("#### Answer")
                    for det in detections:
                        cls = LABEL_CLASS[det.final_label]
                        icon = LABEL_ICON[det.final_label]
                        st.markdown(
                            f'<span class="support-label {cls}">{icon} {det.final_label}</span>{det.sentence}',
                            unsafe_allow_html=True,
                        )
                        if det.final_label in ("FLAGGED", "UNCERTAIN"):
                            with st.expander("Why was this flagged?"):
                                st.write(f"**LLM judge:** {det.llm_label} — {det.llm_justification}")
                                st.write(f"**NLI score:** {det.nli_entailment_prob:.2f} (entailment probability)")
                                st.caption(f"NLI compared against: \"{det.nli_premise_span}\"")

                with col_summary:
                    st.markdown("#### Hallucination Summary")
                    n_sup = sum(1 for d in detections if d.final_label == "SUPPORTED")
                    n_unc = sum(1 for d in detections if d.final_label == "UNCERTAIN")
                    n_flag = sum(1 for d in detections if d.final_label == "FLAGGED")
                    st.markdown(f"✅ {n_sup} supported")
                    st.markdown(f"⚠️ {n_unc} uncertain")
                    st.markdown(f"❌ {n_flag} flagged")
                    export = json.dumps([d.to_dict() for d in detections], indent=2)
                    st.download_button("Export answer as JSON", export, file_name="answer.json")

                with st.expander(f"Show retrieved source chunks ({len(chunks)})"):
                    for c in chunks:
                        st.markdown(
                            f'<div class="chunk-card"><a class="arxiv-badge" '
                            f'href="https://arxiv.org/abs/{c.paper_id}" target="_blank">arXiv:{c.paper_id}</a> '
                            f'&nbsp; <b>{c.title}</b><br><i>{c.section}</i> · score={c.score:.3f}<br>'
                            f'{c.text[:200]}...</div>',
                            unsafe_allow_html=True,
                        )

                with st.expander("Compare retrieval methods for this query"):
                    with st.spinner("Running all 4 retrieval configs..."):
                        comparison_rows = []
                        for mode in ALL_MODES:
                            results = retriever.retrieve(run_query, k_final=5, mode=mode)
                            for r in results:
                                comparison_rows.append({
                                    "config": mode, "rank": r.rank,
                                    "paper_id": r.paper_id, "section": r.section, "score": round(r.score, 3),
                                })
                    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2 — Corpus Browser
# ---------------------------------------------------------------------------
with tab_corpus:
    papers = load_corpus_metadata()
    if not papers:
        st.info("No corpus ingested yet. Run `python src/ingest.py` to populate data/corpus_metadata.json.")
    else:
        st.markdown(f"**{len(papers)} papers indexed**")
        search_term = st.text_input("Search papers...", key="corpus_search")
        filtered = [p for p in papers if search_term.lower() in p["title"].lower()] if search_term else papers
        for p in filtered:
            st.markdown(
                f'<div class="chunk-card"><b>{p["title"]}</b><br>'
                f'{p["arxiv_id"]} &nbsp;·&nbsp; {", ".join(p["authors"][:3])}{" et al." if len(p["authors"]) > 3 else ""}<br>'
                f'<a class="arxiv-badge" href="https://arxiv.org/abs/{p["arxiv_id"]}" target="_blank">View on arXiv ↗</a></div>',
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# Tab 3 — Benchmark & Ablation Results
# ---------------------------------------------------------------------------
with tab_bench:
    retrieval_results, hallucination_results = load_ablation_results()
    st.info(
        "These results were computed on a 20-question manually labeled benchmark. "
        "All benchmark questions and labels are in eval/benchmark.json."
    )

    def _show_dataframe_with_gradient(df: pd.DataFrame, subset: list[str]):
        """Color-gradient styling needs matplotlib (pandas Styler dependency).
        Fall back to a plain dataframe rather than crashing the whole app if
        it's missing for any reason (e.g. a deployment environment where the
        pin didn't resolve)."""
        try:
            st.dataframe(df.style.background_gradient(cmap="RdYlGn", subset=subset),
                         use_container_width=True)
        except ImportError:
            st.dataframe(df, use_container_width=True)
            st.caption("(Color-gradient styling unavailable — matplotlib not installed.)")

    if not retrieval_results:
        st.warning("No ablation results yet. Run `python eval/run_ablation.py` after ingesting the corpus "
                    "and hand-labeling eval/benchmark.json.")
    else:
        st.markdown("#### Retrieval ablation")
        rdf = pd.DataFrame(retrieval_results).T.reset_index().rename(columns={"index": "config"})
        _show_dataframe_with_gradient(rdf, subset=["recall_at_5", "mrr"])
        fig = go.Figure(data=[go.Bar(x=rdf["config"], y=rdf["recall_at_5"])])
        fig.update_layout(title="Recall@5 by retrieval configuration", yaxis_title="Recall@5")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Hallucination detection ablation")
        hdf = pd.DataFrame(hallucination_results).T.reset_index().rename(columns={"index": "detector"})
        _show_dataframe_with_gradient(hdf, subset=["precision", "recall", "f1"])

# ---------------------------------------------------------------------------
# Tab 4 — How It Works
# ---------------------------------------------------------------------------
with tab_how:
    st.markdown("#### What is RAG, and why does naive LLM Q&A fail on scientific literature?")
    st.write(
        "Retrieval-Augmented Generation grounds an LLM's answer in retrieved source text "
        "instead of its parametric memory. Naive Q&A over physics papers fails because "
        "generic LLMs blend memorized facts with fluent-sounding fabrication, and dense "
        "scientific text (equations, cross-references, domain jargon) is exactly where "
        "that fabrication is hardest for a reader to catch."
    )
    st.code(
        "PDFs -> parse+chunk -> dual index (dense+sparse) -> hybrid retrieval + rerank\n"
        "     -> grounded generation (cited) -> dual-signal hallucination check -> answer",
        language="text",
    )

    st.markdown("#### Why SPECTER2 instead of a generic embedding model?")
    st.write(
        "SPECTER2 is trained on scientific paper citation graphs rather than general web "
        "text, so it places papers that cite or resemble each other closer together — a "
        "better match for retrieval over a corpus of scientific papers than a "
        "general-purpose sentence embedding model. See the SPECTER2 paper for details."
    )

    st.markdown("#### Why two hallucination signals instead of one?")
    st.write(
        "An LLM judging outputs from the same family of models tends to be sycophantic — "
        "rating fluent, on-topic text as supported even when a specific claim isn't in the "
        "source. A dedicated NLI cross-encoder has no shared training signal with the "
        "generator, so it fails independently; flagging a sentence if either signal objects "
        "catches more real errors than relying on either signal alone."
    )

    st.markdown("[Full technical report](../report/report.md) · [GitHub repo](.)")