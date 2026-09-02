"""
retriever.py — Hybrid retrieval (dense SPECTER2 + sparse BM25) with
Reciprocal Rank Fusion and cross-encoder reranking.

Exposes one class, HybridRetriever, whose `retrieve(query, k_final, mode)`
method supports four modes with an identical return signature, so the
eval harness (eval/run_ablation.py) can compare them apples-to-apples:

    mode="dense"           -> SPECTER2 only
    mode="sparse"          -> BM25 only
    mode="hybrid"          -> RRF fusion of dense + sparse, no rerank
    mode="hybrid_rerank"   -> RRF fusion + cross-encoder rerank (default, full pipeline)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import chromadb
from sentence_transformers import CrossEncoder

from specter2_encoder import Specter2Encoder, ADHOC_QUERY_ADAPTER

RRF_K = 60  # standard RRF damping constant


@dataclass
class RetrievedChunk:
    chunk_id: str
    paper_id: str
    title: str
    section: str
    text: str
    score: float           # method-specific score (cosine sim / BM25 score / RRF score / rerank score)
    rank: int               # 1-indexed rank within this retrieval call


class HybridRetriever:
    def __init__(self, data_dir: str = "data"):
        data_dir = Path(data_dir)

        self.dense_model = Specter2Encoder(adapter_name=ADHOC_QUERY_ADAPTER)

        print("Loading cross-encoder reranker ...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.chroma_client = chromadb.PersistentClient(path=str(data_dir / "chroma_db"))
        self.collection = self.chroma_client.get_collection("hep_chunks_specter2")

        with open(data_dir / "bm25_index.pkl", "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_chunk_ids = bm25_data["chunk_ids"]

        # chunk_id -> full chunk record, for hydrating results from either index
        self.chunk_by_id = {}
        with open(data_dir / "chunks.jsonl") as f:
            for line in f:
                c = json.loads(line)
                self.chunk_by_id[c["chunk_id"]] = c

    # ---- individual retrieval methods -----------------------------------

    def _dense_search(self, query: str, k: int) -> list[RetrievedChunk]:
        q_emb = self.dense_model.encode([query])
        result = self.collection.query(query_embeddings=q_emb, n_results=k)
        out = []
        for rank, (cid, dist) in enumerate(zip(result["ids"][0], result["distances"][0]), start=1):
            c = self.chunk_by_id[cid]
            out.append(RetrievedChunk(
                chunk_id=cid, paper_id=c["paper_id"], title=c["title"], section=c["section"],
                text=c["text"], score=1 - dist, rank=rank,  # cosine distance -> similarity
            ))
        return out

    def _sparse_search(self, query: str, k: int) -> list[RetrievedChunk]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out = []
        for rank, idx in enumerate(ranked_idx, start=1):
            cid = self.bm25_chunk_ids[idx]
            c = self.chunk_by_id[cid]
            out.append(RetrievedChunk(
                chunk_id=cid, paper_id=c["paper_id"], title=c["title"], section=c["section"],
                text=c["text"], score=float(scores[idx]), rank=rank,
            ))
        return out

    @staticmethod
    def _rrf_fuse(*ranked_lists: list[RetrievedChunk], k: int = RRF_K) -> list[RetrievedChunk]:
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, RetrievedChunk] = {}
        for ranked_list in ranked_lists:
            for item in ranked_list:
                rrf_scores[item.chunk_id] = rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (k + item.rank)
                chunk_lookup[item.chunk_id] = item
        fused_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=cid, paper_id=chunk_lookup[cid].paper_id, title=chunk_lookup[cid].title,
                section=chunk_lookup[cid].section, text=chunk_lookup[cid].text,
                score=rrf_scores[cid], rank=rank,
            )
            for rank, cid in enumerate(fused_ids, start=1)
        ]

    def _cross_encoder_rerank(self, query: str, candidates: list[RetrievedChunk], k_final: int) -> list[RetrievedChunk]:
        pairs = [[query, c.text] for c in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k_final]
        return [
            RetrievedChunk(
                chunk_id=c.chunk_id, paper_id=c.paper_id, title=c.title, section=c.section,
                text=c.text, score=float(score), rank=rank,
            )
            for rank, (c, score) in enumerate(reranked, start=1)
        ]

    # ---- public interface -------------------------------------------------

    def retrieve(self, query: str, k_final: int = 5, mode: str = "hybrid_rerank") -> list[RetrievedChunk]:
        if mode == "dense":
            return self._dense_search(query, k_final)

        if mode == "sparse":
            return self._sparse_search(query, k_final)

        dense = self._dense_search(query, k=20)
        sparse = self._sparse_search(query, k=20)
        fused = self._rrf_fuse(dense, sparse)

        if mode == "hybrid":
            return fused[:k_final]

        if mode == "hybrid_rerank":
            # dedup + cap at ~30 candidates before the (more expensive) rerank pass
            candidates = fused[:30]
            return self._cross_encoder_rerank(query, candidates, k_final)

        raise ValueError(f"Unknown retrieval mode: {mode}")


ALL_MODES = ["dense", "sparse", "hybrid", "hybrid_rerank"]