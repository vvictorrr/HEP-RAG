# Reproducibility

Every number in the README and `report/report.md` is produced by the commands
below, run in this order, from a clean clone. No step uses synthetic or
hand-typed placeholder data as a stand-in for a real result.

## 0. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # required for generation + LLM-judge signal
```

Requires: Python 3.10+, ~4GB free disk for the corpus + models, an Anthropic
API key with access to `claude-sonnet-5`. First run downloads SPECTER2's
base model + proximity adapter, the cross-encoder reranker, and the NLI
model from HuggingFace (~2-3GB total) — this needs real internet access;
it will not work in a fully offline sandbox.

Note on SPECTER2: it loads as a base model (`allenai/specter2_base`) plus
a separately-attached adapter (`allenai/specter2`) via the `adapters`
library — see `src/specter2_encoder.py`. It is not a single
`SentenceTransformer(...)`-loadable model name.

## 1. Corpus acquisition + indexing

```bash
python src/ingest.py \
    --query "cat:hep-ex AND all:missing transverse energy dark matter" \
    --max-results 80 \
    --out-dir data/
```

This downloads PDFs from arXiv (some may 404 — the script logs and skips
these; a run of ~80 requested papers typically yields somewhere in the
60-80 range of successful downloads, logged to stdout), parses them
section-aware with PyMuPDF, chunks them (256 tokens / 64 overlap), embeds
with SPECTER2 into a persistent ChromaDB collection at `data/chroma_db/`,
and builds a BM25 index at `data/bm25_index.pkl`. Expect 20-40 minutes on a
laptop CPU (embedding is the bottleneck; a GPU speeds this up substantially).

Outputs: `data/corpus_metadata.json`, `data/chunks.jsonl`,
`data/bm25_index.pkl`, `data/chroma_db/`.

## 2. Hand-label the benchmark against the real corpus

`eval/benchmark.json` ships with 20 questions (5 categories × 4) and empty
`gold_chunk_ids` / `ground_truth_answer` / `labeled_sentences` fields. Before
running the ablation:

1. For each question, query the retriever manually (or via the Streamlit
   app's Ask tab) to find the actual chunk(s) that answer it, and record
   their `chunk_id`s in `gold_chunk_ids`.
2. Write `ground_truth_answer` as a set of cited sentences using only those
   real chunks — the same constraint the generator itself is under.
3. Populate `labeled_sentences` as a list of `[sentence, cited_chunk_text,
   label]` triples (`label` ∈ `{SUPPORTED, UNSUPPORTED}`), covering both
   sentences from your reference answer and, ideally, some sentences
   deliberately drawn from the *wrong* chunk to give the detector something
   real to catch.

Budget ~2-3 hours. This is the step that makes the eval real rather than
decorative — do not skip or approximate it.

## 3. Run the ablation study

```bash
python eval/run_ablation.py
```

Computes Recall@5/MRR for all 4 retrieval configs and precision/recall/F1
for all 3 hallucination-detector variants, against the hand-labeled
benchmark from step 2. Refuses to run (raises `RuntimeError`) if any
question is still missing `gold_chunk_ids`, so a partially-labeled
benchmark can't silently produce misleading numbers.

Outputs: `eval/retrieval_ablation_results.json`,
`eval/hallucination_ablation_results.json`, `eval/ablation_summary.md`.

## 4. Run the app locally

```bash
streamlit run src/app.py
```

## 5. Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (data/chroma_db/ committed if <100MB, otherwise
   fill in `download_index.py`'s `RELEASE_URL` and upload `data.zip` as a
   release asset).
2. On [share.streamlit.io](https://share.streamlit.io), point a new app at
   `src/app.py`.
3. In the app's Settings → Secrets, paste the contents of
   `.streamlit/secrets.toml.template` with your real `ANTHROPIC_API_KEY`.
4. Add the resulting live URL to the top of `README.md`.