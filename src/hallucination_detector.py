"""
hallucination_detector.py — The core contribution of this project.

Two INDEPENDENT signals judge whether a generated sentence is actually
supported by its cited source chunk:

  Signal 1 (LLM-as-judge):    a second, isolated Claude call classifies
                               SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED
                               with a justification.
  Signal 2 (NLI cross-encoder): cross-encoder/nli-deberta-v3-base — a
                               dedicated natural-language-inference model,
                               architecturally unrelated to the generator —
                               scores entailment / neutral / contradiction.

Why two signals instead of one: an LLM judging its own (or a sibling
model's) output is prone to sycophancy — it tends to rate fluent, on-topic
text as supported even when the specific claim isn't in the source. The
NLI model has no stake in the generation and no shared training signal
with the generator, so it fails differently. Fusing them catches more
than either alone would, at the cost of being a stricter (not looser)
filter — see eval/run_ablation.py's hallucination-detection ablation for
the actual precision/recall/F1 numbers.

IMPORTANT — NLI premise length: cross-encoder/nli-deberta-v3-base is
trained on short, single-sentence premise/hypothesis pairs (SNLI/MultiNLI-
style). Empirically (see debug_nli.py), feeding it a full 256-token chunk
as the premise collapses entailment scores toward ~0 even for claims that
are verbatim in the chunk — the model can't reliably find the matching
content buried in surrounding text. To fix this, _select_relevant_span()
uses the existing ms-marco-MiniLM reranker to narrow the chunk down to its
most relevant sentence(s) before scoring. The LLM judge still sees the
full chunk (it doesn't have this same failure mode).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import anthropic
from sentence_transformers import CrossEncoder
from transformers import pipeline

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker. You will be given a SENTENCE and the \
SOURCE CHUNK it was cited against. Decide whether the source chunk actually \
supports the sentence's claim.

Respond with ONLY a JSON object, no other text:
{{"label": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED", "justification": "<one sentence>"}}

Rules:
- SUPPORTED: every part of the claim is directly stated or a trivial paraphrase of the source.
- PARTIALLY_SUPPORTED: some part of the claim is supported, but it adds specifics \
(numbers, conditions, comparisons) the source does not state.
- UNSUPPORTED: the source does not support the claim, or contradicts it, or is unrelated.
Be strict: if in doubt between SUPPORTED and PARTIALLY_SUPPORTED, choose PARTIALLY_SUPPORTED.

SENTENCE: {sentence}

SOURCE CHUNK: {chunk}
"""

# NLI thresholds (see fuse() below): tuned as defaults against the SHORT-premise
# regime (see debug_nli.py) where a genuine match scores ~0.99 entailment.
# Re-derive from the ablation study in eval/hallucination_ablation_results.json
# once the benchmark is hand-labeled, rather than treating these as final.
FLAG_THRESHOLD = 0.40
UNCERTAIN_THRESHOLD = 0.65

# Physics prose has abbreviations (e.g., et al., Ref., Fig., Eq.) that a naive
# splitter mis-handles, and this regex-based splitter is not immune to that —
# it's a lightweight heuristic to narrow a chunk to a relevant span, not a
# precise sentence tokenizer. Good enough for this purpose; a dedicated
# sentence tokenizer (nltk/spacy) would be more robust if this becomes a
# bottleneck.
_SENTENCE_SPLIT_RE = re.compile(r"(?<!\d)(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


@dataclass
class DetectionResult:
    sentence: str
    llm_label: str            # SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / ERROR
    llm_justification: str
    nli_entailment_prob: float
    nli_label: str             # entailment / neutral / contradiction
    nli_premise_span: str      # the (usually short) span of the chunk actually scored against
    final_label: str           # SUPPORTED / UNCERTAIN / FLAGGED

    def to_dict(self):
        return asdict(self)


def fuse(llm_label: str, nli_entailment_prob: float) -> str:
    """Fusion logic: a sentence is FLAGGED if EITHER signal independently
    flags it. It is UNCERTAIN if only one signal is borderline. Otherwise
    SUPPORTED. This makes FLAGGED the union of two independent failure
    modes, rather than requiring both signals to agree — a stricter,
    higher-recall filter than either alone."""
    if llm_label == "UNSUPPORTED" or nli_entailment_prob < FLAG_THRESHOLD:
        return "FLAGGED"
    if llm_label == "PARTIALLY_SUPPORTED" or nli_entailment_prob < UNCERTAIN_THRESHOLD:
        return "UNCERTAIN"
    return "SUPPORTED"


class HallucinationDetector:
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-5",
                 nli=None, relevance_scorer=None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        # nli/relevance_scorer can be pre-loaded and injected (see src/app.py's
        # _load_detector_models()) so that a multi-user deployment sharing one
        # server process doesn't reload these expensive, API-key-independent
        # models every time a different user supplies a different key — only
        # `self.client` above actually varies per key, and constructing an
        # anthropic.Anthropic client is cheap. Standalone scripts (e.g.
        # eval/run_ablation.py) that just do HallucinationDetector() with no
        # injection still work exactly as before — they load both here.
        if nli is None:
            print("Loading cross-encoder/nli-deberta-v3-base ...")
            nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base",
                            top_k=None)
        self.nli = nli
        if relevance_scorer is None:
            print("Loading sentence-relevance cross-encoder (for NLI premise extraction) ...")
            relevance_scorer = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.relevance_scorer = relevance_scorer

    # ---- Signal 1 -----------------------------------------------------

    def _llm_judge(self, sentence: str, chunk_text: str) -> tuple[str, str]:
        prompt = JUDGE_SYSTEM_PROMPT.format(sentence=sentence, chunk=chunk_text)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            text = text.strip("`").removeprefix("json").strip()
            parsed = json.loads(text)
            return parsed["label"], parsed["justification"]
        except Exception as e:
            return "ERROR", f"LLM judge call failed: {e}"

    # ---- Signal 2 -------------------------------------------------------

    def _select_relevant_span(self, hypothesis: str, chunk_text: str, top_n: int = 3) -> str:
        """Narrow a chunk down to its ~top_n most claim-relevant sentence(s),
        using the existing reranker cross-encoder as a relevance scorer.
        Falls back to the full chunk if it's already short enough to split
        into <= top_n sentences."""
        sentences = _split_sentences(chunk_text)
        if len(sentences) <= top_n:
            return chunk_text
        pairs = [[hypothesis, s] for s in sentences]
        scores = self.relevance_scorer.predict(pairs)
        top_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:top_n]
        top_idx.sort()  # preserve original sentence order for readability
        return " ".join(sentences[i] for i in top_idx)

    def _nli_score(self, sentence: str, chunk_text: str) -> tuple[float, str, str]:
        # cross-encoder/nli-deberta-v3-base expects premise=chunk (source),
        # hypothesis=sentence (claim) — does the source entail the claim?
        # Premise is narrowed to a relevant span first — see module docstring.
        premise = self._select_relevant_span(sentence, chunk_text)
        results = self.nli({"text": premise, "text_pair": sentence})
        scores = {r["label"].lower(): r["score"] for r in results[0]} if isinstance(results[0], list) else \
                 {r["label"].lower(): r["score"] for r in results}
        entailment_prob = scores.get("entailment", 0.0)
        top_label = max(scores, key=scores.get)
        return entailment_prob, top_label, premise

    # ---- Public interface ----------------------------------------------

    def detect(self, sentence: str, cited_chunk_text: str) -> DetectionResult:
        llm_label, justification = self._llm_judge(sentence, cited_chunk_text)
        nli_prob, nli_label, nli_premise_span = self._nli_score(sentence, cited_chunk_text)
        final = fuse(llm_label, nli_prob) if llm_label != "ERROR" else "UNCERTAIN"
        return DetectionResult(
            sentence=sentence, llm_label=llm_label, llm_justification=justification,
            nli_entailment_prob=nli_prob, nli_label=nli_label, nli_premise_span=nli_premise_span,
            final_label=final,
        )

    def detect_all(self, sentences: list[str], cited_chunk_texts: list[str]) -> list[DetectionResult]:
        return [self.detect(s, c) for s, c in zip(sentences, cited_chunk_texts)]

    # ---- Ablation-mode variants (used by eval/run_ablation.py) ----------

    def detect_llm_only(self, sentence: str, cited_chunk_text: str) -> str:
        label, _ = self._llm_judge(sentence, cited_chunk_text)
        return "FLAGGED" if label == "UNSUPPORTED" else (
            "UNCERTAIN" if label == "PARTIALLY_SUPPORTED" else "SUPPORTED")

    def detect_nli_only(self, sentence: str, cited_chunk_text: str) -> str:
        prob, _, _ = self._nli_score(sentence, cited_chunk_text)
        if prob < FLAG_THRESHOLD:
            return "FLAGGED"
        if prob < UNCERTAIN_THRESHOLD:
            return "UNCERTAIN"
        return "SUPPORTED"