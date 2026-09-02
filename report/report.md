# HEP-RAG: Citation-Grounded Retrieval-Augmented Generation over High-Energy Physics Literature, with Dual-Signal Hallucination Detection

## Abstract

Retrieval-augmented generation over scientific literature faces two compounding
problems: retrieval must handle entity-dense, cross-referencing, jargon-heavy
text, and generation must be checked against sycophantic failure modes rather
than assumed correct because it "sounds grounded." This report describes
HEP-RAG, a system built over a corpus of 80 arXiv `hep-ex`/`hep-ph` papers on
MET-based dark matter searches, combining a hybrid dense/sparse retriever with
cross-encoder reranking and a dual-signal (LLM-judge + independent NLI)
hallucination detector. A 4-configuration retrieval ablation and a
3-configuration detector ablation, run against a 20-question hand-labeled
benchmark, show the full pipeline achieving 93.3% Recall@5 (vs. 0.0% for
dense-only retrieval — see the retrieval ablation's analysis for why, and a
caveat on how the full pipeline's figure was measured) and the dual-signal
detector achieving 0.593/0.970/0.736 precision/recall/F1 on flagging
unsupported claims.

## Introduction

Scientific RAG is harder than general-domain RAG along several axes.
High-energy physics papers are entity-dense — a single sentence may reference
a specific trigger threshold, a detector subsystem, a Monte Carlo generator,
and a systematic uncertainty source, each of which needs to be retrieved
correctly for the sentence to be answerable at all. Papers also lean heavily
on cross-reference structures (equations, figures, other papers) that plain
paragraph-level chunking discards. Domain-specific phrasing (e.g. "ABCD
method," "jet veto map," "hard MET") means generic embedding models, trained
on web text, have weak semantic resolution between HEP-specific concepts that
look superficially similar. Finally, the cost of a hallucination is higher in
a scientific context than in casual Q&A: a fabricated exclusion limit or a
misattributed systematic uncertainty is the kind of error a domain expert
would catch immediately but a general reader would not, and would actively
be misled by.

## Related Work

Frameworks like RAGAS and ARES evaluate RAG systems primarily via LLM-based
scoring of faithfulness and answer relevance — an approach that inherits the
same sycophancy risk this project is designed around, since an LLM judge
alone can rate its own model family's output as faithful even when a specific
claim isn't supported. HEP-RAG's dual-signal design instead pairs an LLM
judge with an architecturally independent NLI model, so the two signals fail
on different inputs rather than sharing a common blind spot.

## System Architecture

**Retrieval.** PDFs are parsed section-aware with PyMuPDF and split into
hierarchical chunks: full sections for context, 256-token/64-token-overlap
chunks as the retrieval unit, and sentence-level spans for the hallucination
checker. A dual index — SPECTER2 dense embeddings in ChromaDB, plus a BM25
sparse index — is fused via Reciprocal Rank Fusion and reranked with
`cross-encoder/ms-marco-MiniLM-L-6-v2` to a final top-5. SPECTER2 is used
instead of a generic sentence embedding model because it is trained on
scientific citation graphs (Cohan et al., SPECTER, 2020; Singh et al.,
SPECTER2, 2022) and places genuinely related scientific papers closer
together than a general-purpose embedding model trained on web text does.

**Generation.** The top-5 chunks, with full citation metadata, are passed to
Claude with a system prompt enforcing per-sentence citation in `[paper_id,
section]` format and an explicit `CORPUS_INSUFFICIENT` refusal path when the
retrieved context doesn't support an answer.

**Hallucination detection.** Each generated sentence is checked against its
cited chunk by two independent signals: an LLM-judge (a second, isolated
Claude call classifying SUPPORTED/PARTIALLY_SUPPORTED/UNSUPPORTED) and an NLI
cross-encoder (`cross-encoder/nli-deberta-v3-base`) producing an entailment
probability. A sentence is FLAGGED if either signal independently objects,
UNCERTAIN if only one signal is borderline, and SUPPORTED otherwise. This
two-signal design exists because an LLM judge sharing a model family with the
generator is prone to rating fluent, on-topic, citation-shaped text as
supported even when the specific claim isn't in the source — the NLI model
has no such incentive and fails on different inputs.

## Evaluation

Both ablations were run against a 20-question benchmark hand-labeled by the
author directly from the real ingested corpus (80 arXiv `hep-ex`/`hep-ph`
papers, 5,765 chunks) — not synthetic or assumed data. See
`eval/benchmark.json` and `REPRODUCIBILITY.md` for the labeling process.

**A methodological caveat that applies to the retrieval table below, stated
up front rather than left implicit:** gold chunks were selected by
inspecting each question's top-5 results from the `hybrid_rerank` config
specifically (via `eval/prepare_labeling.py`), not from an independent,
retriever-agnostic ground truth. This means the full pipeline's Recall@5 is
not a fully fair, apples-to-apples number against the other three configs —
by construction, every gold chunk was already inside `hybrid_rerank`'s own
shortlist, which mechanically inflates its measured recall relative to
configs that were never given a chance to nominate their own top-5 as
candidates. The comparison across configs is still informative (see
analysis below), but the full pipeline's 0.933 should not be read as an
independently-verified ground-truth accuracy figure.

### Retrieval ablation (Recall@5 / MRR, n=15 of 20 questions scored — see note below)

| Config | Recall@5 | MRR |
|---|---|---|
| Dense (SPECTER2) only | 0.000 | 0.000 |
| Sparse (BM25) only | 0.400 | 0.322 |
| Hybrid RRF (no rerank) | 0.400 | 0.213 |
| Hybrid RRF + cross-encoder rerank (full) | 0.933 | 0.656 |

5 of the 20 benchmark questions (Category D "unanswerable," plus E4) have no
gold chunk by design — nothing in the corpus should be cited for them — so
they're excluded from Recall@5/MRR scoring entirely rather than counted as
automatic misses.

**Analysis.** The most striking result is dense-only's exact 0.000 across
all 15 scored questions. This was initially suspected to be a bug: SPECTER2
is asymmetric, requiring a separate adapter for encoding queries
(`allenai/specter2_adhoc_query`) versus documents (`allenai/specter2`, the
"proximity" adapter) — an early version of this pipeline used the document
adapter symmetrically for both, which is a real, since-fixed defect. Fixing
it and re-running produced an unchanged 0.000, ruling that out as the (sole)
cause. Diagnostic inspection of raw dense-only output
(`debug_dense_retrieval.py`) instead showed near-uniform cosine scores
across the top-5 (e.g. 0.7694 to 0.7670 — separated by thousandths) and, for
one question, top results drawn from entirely unrelated physics subfields
(direct-detection detector calibration, gamma-ray indirect detection) rather
than merely imprecise mono-jet-adjacent results. Both signatures point to a
genuine domain/granularity mismatch rather than a remaining implementation
bug: SPECTER2 is trained on citation-graph-based document-level topical
similarity (does paper A relate to paper B, typically via title+abstract),
which is a different task from discriminating which specific, jargon-dense,
256-token body-text passage answers a specific technical question. Full
diagnostic detail is in `eval/retrieval_ranking_observations.md`.

A second, related finding: hybrid RRF *without* reranking does not clearly
beat sparse alone — tied on Recall@5 (0.400) but notably worse on MRR (0.213
vs. 0.322). This suggests that fusing in a near-uninformative dense signal
via RRF actively degrades ranking quality even when it doesn't change
whether the right chunk appears in the top 5 at all — consistent with
dense's near-random scores diluting rather than complementing BM25's
signal. The full pipeline's recovery to 0.933/0.656 appears to be driven
almost entirely by the cross-encoder reranking step correctly reordering a
candidate pool that BM25, not SPECTER2, populated with the actually-relevant
chunks. In other words: in this pipeline, on this corpus, essentially all of
the real retrieval quality traces back to BM25 + reranking, with SPECTER2
contributing close to nothing standalone — a more specific and more useful
finding than a flat "hybrid beats sparse" result would have been.

A third, independent finding, unrelated to the SPECTER2 issue: across three
separate benchmark questions (A3, B2, B4), the single chunk that most
precisely and numerically answered the question consistently ranked *last*
among the 5 chunks retrieved by the full pipeline, while vaguer, more
narrative chunks ranked higher. This pattern held even before the SPECTER2
diagnosis and is more likely attributable to the reranker's own training
distribution (`ms-marco-MiniLM-L-6-v2`, trained on web-search-style
query-passage relevance) favoring fluent, question-echoing prose over
terse tabular/enumerated technical content. See
`eval/retrieval_ranking_observations.md` for the full case-by-case detail.

### Hallucination detection ablation (P/R/F1, n=69 labeled sentences across all 20 questions)

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| LLM judge only | 0.971 | 1.000 | 0.985 |
| NLI cross-encoder only | 0.569 | 1.000 | 0.725 |
| Dual-signal fusion (full) | 0.593 | 0.970 | 0.736 |

**Analysis.** The LLM judge's near-perfect score should not be read as "the
LLM judge doesn't fail" — it failed in exactly this way once already, on a
real, live example (see the citation-misattribution case study below), and
that failure was subtle: a fluent, plausible, topically-correct sentence
whose citation pointed at the wrong paper. What this benchmark's `1.000`
recall / `0.971` precision actually reflects is that the LLM judge reliably
catches *obvious* mismatches (unrelated topics, explicit numeric
contradictions) — which is most of what the hand-constructed
`labeled_sentences` negative examples are, since they were built as clear
teachable cases rather than maximally adversarial ones. The real
misattribution case the pipeline caught live was subtler than anything in
this benchmark, and is arguably the more informative evidence of the LLM
judge's actual failure mode.

The dual-signal fusion's precision (0.593) sits much closer to the NLI
signal's own precision (0.569) than to the LLM judge's (0.971). This is a
direct, explainable consequence of the fusion rule (`fuse()` in
`hallucination_detector.py`): a sentence is FLAGGED if *either* signal
objects. Since the LLM judge contributes almost no false positives on this
benchmark, nearly all of dual-signal's false positives are inherited from
NLI's continued over-flagging of genuinely-supported sentences — meaning
even after the NLI premise-narrowing fix (feeding the model a relevant
sentence span rather than a full 256-token chunk; see below), NLI still
meaningfully over-flags true positives as unsupported. This is a real,
current limitation of the dual-signal design as implemented, not
disqualifying (the union-based fusion is what let it catch the real
misattribution case NLI alone would have also caught, and LLM-alone would
have missed), but a legitimate target for future threshold recalibration
(`FLAG_THRESHOLD`/`UNCERTAIN_THRESHOLD` in `hallucination_detector.py`).

### Case study: a real caught hallucination

While testing the live app (not this benchmark), a generated answer
correctly stated real, verbatim-accurate numbers (`MET > 350 GeV`, leading
jet `pT > 110 GeV`, `|eta| < 2.4`) but initially cited them to the wrong
paper — the numbers were real and present in the retrieved context, just in
a different chunk (`1206.0753v1`) than the one the citation pointed to
(`1409.2893v2`). The NLI signal correctly scored near-zero entailment
against the (wrong) cited chunk (0.01), while the LLM judge incorrectly
rated the sentence SUPPORTED, apparently fooled by the sentence's fluency
and topical correctness despite the citation being wrong. The dual-signal
fusion correctly flagged the sentence regardless, since FLAGGED requires
only one signal to object. Root cause: the generator's citation prompt
originally asked the model to reproduce `[paper_id, section]` as free text,
a harder, more error-prone generation task than necessary; switching to a
numbered-index citation format (`[N]`, resolved deterministically against
the exact retrieved chunk list rather than trusted as free text) fixed the
app-side half of this bug outright and reduced, though does not guarantee
against, the model-side citation error. Full detail, including the exact
retrieved chunk text, is preserved in project chat history and available on
request.


## Limitations

- **Gold-chunk labeling circularity.** The 20-question benchmark's gold
  chunks were selected from `hybrid_rerank`'s own top-5 output (see
  Evaluation, above), meaning the full pipeline's measured Recall@5 is not
  an independent, retriever-agnostic accuracy figure. A more rigorous
  future benchmark would derive gold chunks from a broader, retriever-
  agnostic candidate pool (e.g. top-20-per-config or full-text search)
  before any single config's output is inspected.
- **SPECTER2 contributes negligible standalone retrieval value in this
  domain** (see Evaluation, above) — a genuine domain/granularity mismatch
  between citation-graph-trained document embeddings and fine-grained,
  jargon-dense passage retrieval, not a remaining implementation defect.
  Nearly all of the working pipeline's retrieval quality is attributable to
  BM25 + cross-encoder reranking.
- The NLI model (`cross-encoder/nli-deberta-v3-base`) is trained on
  general-domain entailment data and may underperform on HEP claims that are
  true but phrased in ways that look semantically distant from its training
  distribution — a known domain-transfer gap for off-the-shelf NLI models
  applied to specialist scientific text. This was directly observed (see
  the case study above and `eval/hallucination_ablation_results.json`'s
  precision figures) even after mitigating the worst of it by narrowing the
  NLI premise to a relevant sentence span rather than a full chunk.
- The corpus covers a single HEP subfield (MET-based dark matter searches);
  conclusions about retrieval/detection performance may not transfer to
  other subfields without re-running the ablation on a different corpus. In
  practice, the arXiv query used also pulled in a small number of
  topically-adjacent-but-distinct papers (a dark-photon haloscope search,
  an indirect-detection review, a direct-detection calibration white
  paper) that surfaced during benchmark labeling as useful negative-example
  material but are not, strictly, mono-jet collider searches.
- PDF text-layer extraction does not preserve equation structure; equations
  are captured as plain-text fragments with an `equation_refs` pointer
  rather than structured math, which likely degrades retrieval precision on
  equation-heavy questions specifically.
- The section-header detector (`SECTION_HEADERS`/`SECTION_RE` in
  `src/ingest.py`) is tuned to typical ATLAS/CMS experimental-analysis
  paper structure and mislabels sections in papers with non-standard
  structure (Roman-numeral headings, lettered subsections, thesis-style
  chapters) — most such content gets labeled "Introduction" by default.
  This does not affect retrieval (embeddings are computed from chunk text,
  not the section label), only the accuracy of citation display in the app
  and report. Diagnosed during benchmark labeling; not yet fixed.
- Not yet deployed to Streamlit Community Cloud — verified working via
  `streamlit run src/app.py` locally only.

## Conclusion

HEP-RAG demonstrates that scientific RAG quality and hallucination-detection
reliability are measurable, not assumable: a 4-configuration retrieval
ablation and a 3-configuration detector ablation, run against a hand-labeled
20-question benchmark spanning single-chunk, multi-chunk, partially
answerable, unanswerable, and adversarial question types, give a concrete,
reproducible account of what this system gets right and where it still
fails — the latter being, in an evaluation-honest project, at least as
informative as the former.