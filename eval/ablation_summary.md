# Ablation Summary

## Retrieval ablation (Recall@5 / MRR)

| Config | Recall@5 | MRR |
|---|---|---|
| Dense (SPECTER2) only | 0.000 | 0.000 |
| Sparse (BM25) only | 0.400 | 0.322 |
| Hybrid RRF (no rerank) | 0.400 | 0.213 |
| Hybrid RRF + cross-encoder rerank (full) | 0.933 | 0.656 |

## Hallucination detection ablation (P/R/F1)

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| LLM judge only | 1.000 | 1.000 | 1.000 |
| NLI cross-encoder only | 0.600 | 1.000 | 0.750 |
| Dual-signal fusion (full) | 0.615 | 0.970 | 0.753 |
