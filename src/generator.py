"""
generator.py — Grounded generation over retrieved chunks, with citation
enforcement and structured parsing into per-sentence {sentence, citation}
objects for the downstream hallucination detector.

Citation format: the model cites [N], the numbered position of a context
chunk (as shown in _format_context), NOT the paper_id/section as free text.
This is a deliberate fix for a real observed failure mode: when asked to
reproduce [paper_id, section] verbatim, the model sometimes attached a
correct, verbatim-from-the-corpus fact to the wrong paper's citation
(source misattribution) — especially when multiple retrieved chunks share
similar topical language. A small integer is a much narrower, lower-entropy
token for the model to get right than free-text metadata, and — critically —
once emitted, [N] resolves to an exact chunk_id deterministically on our
side, with no string-matching ambiguity. This does not make source
misattribution impossible (the model can still write the wrong N), only
less likely and more cheaply checkable; the dual-signal hallucination
detector remains the safety net for whatever slips through.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import anthropic

DEFAULT_MODEL = os.environ.get("GENERATION_MODEL", "claude-sonnet-5")

SYSTEM_PROMPT = """You are a physics literature assistant. You answer questions ONLY using \
the numbered context chunks provided below. Follow these rules exactly:

1. Answer in complete sentences, one sentence per line.
2. Every sentence containing a factual claim must end with a citation in the \
exact format [N], where N is the number of the context chunk (e.g. "[1]", "[2]") \
whose text actually contains the fact you are stating. Cite the chunk NUMBER only —
never the paper_id or section name as text.
3. Before citing a chunk number, re-check that chunk's actual text contains the \
specific fact (the number, name, or claim) you are about to state. If two chunks \
are from the same paper or cover similar ground, do not assume a fact belongs to \
whichever one you saw first or most recently — check the specific chunk's text.
4. Do not use any knowledge beyond the provided context, even if you know the \
answer from general physics knowledge. This corpus may be incomplete or use \
different conventions than what you recall — defer to the context.
5. If the provided context does not contain enough information to answer the \
question, output exactly one line: CORPUS_INSUFFICIENT: [brief reason]
6. Do not add a preamble, summary, or caveats outside the cited sentences.

CONTEXT CHUNKS:
{context_block}
"""

CITATION_RE = re.compile(r"\[(\d+)\]\s*\.?\s*$")


@dataclass
class GeneratedSentence:
    sentence: str
    chunk_index: int | None   # 1-indexed position among the chunks passed to generate(); None if uncited/invalid
    paper_id: str | None      # resolved from chunk_index, for display only
    section: str | None       # resolved from chunk_index, for display only
    raw_citation: str | None


@dataclass
class GenerationResult:
    corpus_insufficient: bool
    reason: str | None
    sentences: list[GeneratedSentence]
    raw_text: str


def _format_context(chunks) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] paper_id={c.paper_id} | section={c.section} | title=\"{c.title}\"\n{c.text}"
        )
    return "\n\n".join(blocks)


def _parse_response(text: str, chunks) -> GenerationResult:
    text = text.strip()
    if text.startswith("CORPUS_INSUFFICIENT"):
        reason = text.split(":", 1)[1].strip() if ":" in text else None
        return GenerationResult(corpus_insufficient=True, reason=reason, sentences=[], raw_text=text)

    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = CITATION_RE.search(line)
        if m:
            idx = int(m.group(1))
            sentence_text = line[:m.start()].strip()
            if 1 <= idx <= len(chunks):
                cited = chunks[idx - 1]
                sentences.append(GeneratedSentence(sentence_text, idx, cited.paper_id, cited.section, m.group(0)))
            else:
                # model cited a chunk number outside the range it was actually given — treat as uncited
                sentences.append(GeneratedSentence(sentence_text, None, None, None, m.group(0)))
        else:
            sentences.append(GeneratedSentence(line, None, None, None, None))
    return GenerationResult(corpus_insufficient=False, reason=None, sentences=sentences, raw_text=text)


class Generator:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, question: str, chunks) -> GenerationResult:
        system = SYSTEM_PROMPT.format(context_block=_format_context(chunks))
        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            system=system,
            messages=[{"role": "user", "content": question}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _parse_response(text, chunks)