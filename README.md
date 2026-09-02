# HEP-RAG — Citation-Grounded Physics Literature Q&A with Dual-Signal Hallucination Detection

**Live demo:** _add your Streamlit Community Cloud URL here after deploying (see REPRODUCIBILITY.md §6)_

A retrieval-augmented Q&A system over a corpus of high-energy physics papers
(arXiv `hep-ex`, dark matter / mono-jet / MET-based searches), built as a
rigorous, ablation-driven study rather than a demo. Every generated claim is
cited to a specific paper + section, and checked by two independent signals
before being labeled supported, uncertain, or flagged.

## Why this exists

Scientific RAG is a harder problem than general-purpose RAG: HEP papers are
entity-dense, cross-reference heavily (equations, figures, other papers),
use domain-specific phrasing generic embeddings weren't trained on, and the
cost of a confident-sounding hallucination is high in a scientific context.
This project treats retrieval quality and hallucination detection as
empirical questions to be measured, not assumed — see the ablation study
below.

## Architecture

```
PILLAR 1 — RETRIEVAL                      PILLAR 2 — GENERATION + DETECTION
─────────────────────                     ──────────────────────────────────
arXiv PDFs (hep-ex)                       Retrieved chunks (top-5, cited)
  → section-aware parse (PyMuPDF)           → grounded generation (Claude)
  → hierarchical chunking                      - cites [paper_id, section]
  → dual index:                                - refuses if context insufficient
      dense = SPECTER2 (ChromaDB)          → sentence segmentation
      sparse = BM25                        → SIGNAL 1: LLM-as-judge
  → hybrid retrieval (RRF fusion)             (SUPPORTED/PARTIALLY/UNSUPPORTED)
  → cross-encoder rerank → top-5          → SIGNAL 2: NLI cross-encoder
                                               (independent architecture)
                                           → fusion → SUPPORTED / UNCERTAIN / FLAGGED

PILLAR 3 — ABLATION & EVAL
───────────────────────────
20-question benchmark, 5 categories (single-chunk, multi-chunk synthesis,
partially answerable, unanswerable, adversarial false-premise)
  → retrieval ablation: dense-only vs sparse-only vs hybrid vs hybrid+rerank
      (Recall@5, MRR)
  → detector ablation: LLM-judge-only vs NLI-only vs dual-signal
      (precision/recall/F1 on flagging unsupported sentences)
```

## Why SPECTER2, not generic embeddings

SPECTER2 is trained on scientific citation graphs, so it places papers that
actually cite/resemble each other closer together — a materially better fit
for retrieval over a physics corpus than a general sentence embedding model
trained on web text.

## Why two independent hallucination signals

An LLM judging outputs from its own model family tends toward sycophancy —
rating fluent, on-topic text as supported even when a specific claim isn't
in the source. A dedicated NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`)
shares no training signal with the generator or the judge, so it fails
differently. A sentence is flagged if *either* signal independently objects,
so the fusion is a stricter filter than either alone.

## Ablation results

Run against a 20-question benchmark hand-labeled by the author directly
from the real ingested corpus (`eval/benchmark.json`). Full analysis,
including two honest methodological caveats, is in
`report/report.md`'s Evaluation section — read that before quoting these
numbers anywhere, especially the full pipeline's Recall@5.

**Retrieval ablation** (Recall@5 / MRR, n=15 of 20 questions scored —
5 questions have no gold chunk by design, see report):

| Config | Recall@5 | MRR |
|---|---|---|
| Dense (SPECTER2) only | 0.000 | 0.000 |
| Sparse (BM25) only | 0.400 | 0.322 |
| Hybrid RRF (no rerank) | 0.400 | 0.213 |
| Hybrid RRF + cross-encoder rerank (full) | 0.933 | 0.656 |

SPECTER2's exact 0.000 was investigated as a possible bug (wrong query-side
adapter) and fixed, but the number didn't change — diagnostic evidence
(`eval/retrieval_ranking_observations.md`) points to a genuine domain
mismatch instead: SPECTER2 is trained for document-level topical similarity,
not fine-grained passage retrieval against jargon-dense technical text.
Nearly all of the working pipeline's retrieval quality traces back to BM25 +
cross-encoder reranking, not to the dense/SPECTER2 component.

**Hallucination detection ablation** (P/R/F1, n=69 labeled sentences):

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| LLM judge only | 0.971 | 1.000 | 0.985 |
| NLI cross-encoder only | 0.569 | 1.000 | 0.725 |
| Dual-signal fusion (full) | 0.593 | 0.970 | 0.736 |

## Example output

A real, live-tested example of the dual-signal detector catching a genuine
error — not a clean success case, per the note above.

**Question:** "What MET threshold is used in the mono-jet signal region?"

The system generated (among other sentences): *"In the CMS monojet
analysis, the optimal MET cut for the signal region is determined to be
MET > 350 GeV, along with a leading jet pT > 110 GeV and |eta| < 2.4
requirement"* — citing paper `1409.2893v2`.

- **LLM judge:** SUPPORTED — "The source explicitly states the optimal MET
  cut is 350 GeV..." (**wrong** — see below)
- **NLI score:** 0.01 (near-zero entailment against the cited chunk)
- **Dual-signal fusion:** FLAGGED (correctly, since either signal objecting
  is sufficient)

**What actually happened:** every number in the sentence is real and
verbatim-correct — but from a *different* retrieved paper, `1206.0753v1`,
not the one cited. The LLM judge was fooled by the sentence's fluency and
topical correctness; the NLI model, with no stake in the generation,
correctly flagged that the *cited* chunk didn't support the claim. This is
exactly the sycophancy-vs-independent-signal failure mode the dual-signal
architecture exists to catch. Root cause and fix (switching from free-text
`[paper_id, section]` citations to a numbered-index format resolved
deterministically against the retrieved chunk list) are in
`src/generator.py` and documented in `report/report.md`'s case study.

## Setup

```bash
git clone <this-repo>
cd hep-rag
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
python src/ingest.py            # builds the corpus + indexes (~20-40 min)
streamlit run src/app.py
```

Full reproduction steps, including hand-labeling the eval benchmark and
running the ablation study, are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Repo structure

```
hep-rag/
├── README.md
├── REPRODUCIBILITY.md
├── RESUME_BULLET.txt
├── requirements.txt
├── download_index.py
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.template
├── data/                      ← generated by src/ingest.py
├── src/
│   ├── ingest.py
│   ├── retriever.py
│   ├── generator.py
│   ├── hallucination_detector.py
│   └── app.py
├── eval/
│   ├── benchmark.json
│   ├── run_ablation.py
│   ├── retrieval_ablation_results.json    ← generated
│   ├── hallucination_ablation_results.json ← generated
│   └── ablation_summary.md                ← generated
└── report/
    └── report.md
```

## Limitations

- **Gold-chunk labeling circularity**: the benchmark's gold chunks were
  selected from the full pipeline's own top-5 output, so its measured
  Recall@5 (0.933, above) is not a fully independent accuracy figure —
  see `report/report.md`'s Evaluation section.
- **SPECTER2 contributes ~0 standalone retrieval value in this domain** —
  a genuine domain/granularity mismatch, investigated and documented rather
  than left unexplained; see the ablation results above and
  `eval/retrieval_ranking_observations.md`.
- The NLI model may underperform on highly technical HEP claims that are
  semantically distant from its training data (news/entailment corpora),
  even when they're true — a known limitation of applying general-domain
  NLI to specialist scientific text, directly observed in the ablation
  above.
- The corpus covers one subfield (MET-based dark matter searches); results
  may not generalize to other HEP subfields without re-ingesting a
  different corpus.
- PDF parsing (PyMuPDF, text-layer extraction) does not preserve equation
  structure — equations are captured only as plain-text fragments plus an
  `equation_refs` pointer (e.g. "Eq. 3"), not as structured math.
- The section-header detector mislabels sections in papers with
  non-standard structure (most such content defaults to "Introduction").
  Cosmetic — doesn't affect retrieval, only citation display accuracy.
  Diagnosed, not yet fixed.
- Not yet deployed to Streamlit Community Cloud (local only).

## License

MIT — see the reproducibility doc for how to regenerate every result from
scratch on your own machine.