"""
specter2_encoder.py — Correct SPECTER2 loading.

SPECTER2 is NOT a single loadable sentence-transformers model. It is
distributed as a base transformer (`allenai/specter2_base`) plus a
separately-loaded adapter, combined via AllenAI's `adapters` library.
`SentenceTransformer("allenai/specter2")` 404s because "allenai/specter2"
on its own is only adapter weights, not a full model — that was the bug
in an earlier version of this file.

IMPORTANT — SPECTER2 is asymmetric, and an earlier version of this file
got that wrong too: AllenAI ships TWO adapters for two different roles,
not one adapter used symmetrically for everything.

  - `allenai/specter2` (the "proximity" adapter): for encoding DOCUMENTS
    (papers), as "title [SEP] abstract-or-passage". Correct for building
    the corpus index in ingest.py.
  - `allenai/specter2_adhoc_query` (the "adhoc query" adapter): for
    encoding QUERIES/QUESTIONS in ad-hoc query-to-document retrieval — a
    bare question string, no title/[SEP] formatting needed.

These two adapters are meant to be mixed: query embeddings from the
adhoc_query adapter are compared against document embeddings from the
proximity adapter. A previous version of this file loaded only the
proximity adapter and used it for both documents AND bare queries at
retrieval time — mechanically valid (it runs, it returns vectors) but
semantically wrong, since the proximity adapter was never trained to
represent a bare natural-language question. That mismatch is the likely
cause of dense-only retrieval scoring an exact 0.000 Recall@5 across the
full benchmark (see eval/retrieval_ranking_observations.md) — not just
"SPECTER2 is weak here," but query and document embeddings plausibly
living in different representational regimes entirely.

Fix: ingest.py continues to use adapter_name="allenai/specter2" (the
default) for documents — no re-ingestion needed, those embeddings were
already correct. retriever.py now explicitly requests
adapter_name="allenai/specter2_adhoc_query" for its query-side encoder.

Reference: https://github.com/allenai/SPECTER2
"""

from __future__ import annotations

import torch
from adapters import AutoAdapterModel
from transformers import AutoTokenizer

PROXIMITY_ADAPTER = "allenai/specter2"           # documents
ADHOC_QUERY_ADAPTER = "allenai/specter2_adhoc_query"  # queries


class Specter2Encoder:
    def __init__(self, adapter_name: str = PROXIMITY_ADAPTER, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_name = adapter_name
        print(f"Loading allenai/specter2_base + adapter '{adapter_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(adapter_name, source="hf", load_as="specter2_active", set_active=True)
        self.model.to(self.device)
        self.model.eval()

    @property
    def sep_token(self) -> str:
        return self.tokenizer.sep_token

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 16) -> list[list[float]]:
        """Encode texts (either 'title [SEP] passage' documents when loaded
        with the proximity adapter, or bare query strings when loaded with
        the adhoc_query adapter). Returns the [CLS]-token embedding from the
        last hidden state — SPECTER's standard representation (not mean
        pooling) — as plain Python lists, ready for ChromaDB."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            outputs = self.model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.extend(cls_embeddings.cpu().tolist())
        return all_embeddings