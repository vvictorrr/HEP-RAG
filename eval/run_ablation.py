"""
run_ablation.py — Runs the full retrieval ablation (4 configs) and
hallucination-detection ablation (3 detector variants) over the 20-question
benchmark, and writes real results to:

  eval/retrieval_ablation_results.json
  eval/hallucination_ablation_results.json
  eval/ablation_summary.md

IMPORTANT: this script computes retrieval Recall@5/MRR against a
`gold_chunk_ids` field per question, and hallucination P/R/F1 against
`labeled_sentences` per question — both of which must be hand-labeled in
eval/benchmark.json against the REAL ingested corpus first (see the
`_README` note in that file and REPRODUCIBILITY.md).

Some questions (Category D "unanswerable", and any Category E question
where no retrieved chunk answers it, e.g. E4) have EMPTY gold_chunk_ids
BY DESIGN — nothing in the corpus should be cited, so there is no gold
chunk to compute Recall@5/MRR against. These are excluded from the
retrieval-ablation average (not scored as automatic misses), but their
labeled_sentences still contribute to the hallucination-detection
ablation, which is unaffected by whether a gold retrieval chunk exists.

This script still refuses to run if a question has an empty
labeled_sentences list AND a non-"CORPUS_INSUFFICIENT"-style
ground_truth_answer, since that combination usually means labeling was
simply never finished for that question rather than intentionally left
empty — see load_benchmark() below.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from retriever import HybridRetriever, ALL_MODES  # noqa: E402
from hallucination_detector import HallucinationDetector  # noqa: E402

BENCHMARK_PATH = Path(__file__).parent / "benchmark.json"

# Categories where an empty gold_chunk_ids list is expected/intentional
# (the question is designed to test refusal, not retrieval recall).
NO_GOLD_EXPECTED_CATEGORIES = {"D"}


def load_benchmark():
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    questions = data["questions"]

    unfinished = [
        q["id"] for q in questions
        if not q.get("labeled_sentences") and not q.get("ground_truth_answer")
    ]
    if unfinished:
        raise RuntimeError(
            f"{len(unfinished)} benchmark questions have neither labeled_sentences nor "
            f"a ground_truth_answer: {unfinished}. These look genuinely unlabeled rather "
            f"than intentionally empty — hand-label them against the real ingested corpus "
            f"before running the ablation (see benchmark.json's _README field). Refusing "
            f"to compute metrics against placeholder data."
        )
    return questions


def recall_at_k(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    return 1.0 if any(rid in gold_ids for rid in retrieved_ids) else 0.0


def mrr(retrieved_ids: list[str], gold_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in gold_ids:
            return 1.0 / rank
    return 0.0


def run_retrieval_ablation(questions, retriever) -> dict:
    scored_questions = [q for q in questions if q.get("gold_chunk_ids")]
    skipped = [q["id"] for q in questions if not q.get("gold_chunk_ids")]
    if skipped:
        print(f"Skipping retrieval-recall scoring for {len(skipped)} questions with no "
              f"gold chunk by design (unanswerable/refusal cases): {skipped}")

    results = {mode: {"recall_at_5": [], "mrr": []} for mode in ALL_MODES}
    for q in scored_questions:
        gold = q["gold_chunk_ids"]
        for mode in ALL_MODES:
            retrieved = retriever.retrieve(q["question"], k_final=5, mode=mode)
            ids = [r.chunk_id for r in retrieved]
            results[mode]["recall_at_5"].append(recall_at_k(ids, gold))
            results[mode]["mrr"].append(mrr(ids, gold))

    summary = {}
    for mode in ALL_MODES:
        n = len(results[mode]["recall_at_5"])
        summary[mode] = {
            "recall_at_5": sum(results[mode]["recall_at_5"]) / n,
            "mrr": sum(results[mode]["mrr"]) / n,
            "n_questions": n,
        }
    return summary


def prf1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_hallucination_ablation(questions, detector: HallucinationDetector) -> dict:
    """
    Ground truth: sentence_labels[i] = "SUPPORTED" | "UNSUPPORTED" for the
    i-th sentence of the hand-written reference answer. We treat
    "flag unsupported sentences" as the positive class.
    """
    counts = {variant: {"tp": 0, "fp": 0, "fn": 0} for variant in ["llm_only", "nli_only", "dual_signal"]}

    for q in questions:
        for sentence, cited_chunk_text, label in q.get("labeled_sentences", []):
            gold_positive = (label == "UNSUPPORTED")

            llm_pred = detector.detect_llm_only(sentence, cited_chunk_text) == "FLAGGED"
            nli_pred = detector.detect_nli_only(sentence, cited_chunk_text) == "FLAGGED"
            dual_pred = detector.detect(sentence, cited_chunk_text).final_label == "FLAGGED"

            for variant, pred in [("llm_only", llm_pred), ("nli_only", nli_pred), ("dual_signal", dual_pred)]:
                if pred and gold_positive:
                    counts[variant]["tp"] += 1
                elif pred and not gold_positive:
                    counts[variant]["fp"] += 1
                elif not pred and gold_positive:
                    counts[variant]["fn"] += 1

    return {variant: prf1(**c) for variant, c in counts.items()}


def write_summary_md(retrieval_results: dict, hallucination_results: dict, out_path: Path):
    lines = ["# Ablation Summary\n", "## Retrieval ablation (Recall@5 / MRR)\n",
             "| Config | Recall@5 | MRR |", "|---|---|---|"]
    labels = {"dense": "Dense (SPECTER2) only", "sparse": "Sparse (BM25) only",
              "hybrid": "Hybrid RRF (no rerank)", "hybrid_rerank": "Hybrid RRF + cross-encoder rerank (full)"}
    for mode in ALL_MODES:
        r = retrieval_results[mode]
        lines.append(f"| {labels[mode]} | {r['recall_at_5']:.3f} | {r['mrr']:.3f} |")

    lines += ["\n## Hallucination detection ablation (P/R/F1)\n",
              "| Detector | Precision | Recall | F1 |", "|---|---|---|---|"]
    d_labels = {"llm_only": "LLM judge only", "nli_only": "NLI cross-encoder only",
                "dual_signal": "Dual-signal fusion (full)"}
    for variant in ["llm_only", "nli_only", "dual_signal"]:
        r = hallucination_results[variant]
        lines.append(f"| {d_labels[variant]} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |")

    out_path.write_text("\n".join(lines) + "\n")


def main():
    questions = load_benchmark()
    retriever = HybridRetriever(data_dir="data")
    detector = HallucinationDetector()

    retrieval_results = run_retrieval_ablation(questions, retriever)
    with open(Path(__file__).parent / "retrieval_ablation_results.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)

    hallucination_results = run_hallucination_ablation(questions, detector)
    with open(Path(__file__).parent / "hallucination_ablation_results.json", "w") as f:
        json.dump(hallucination_results, f, indent=2)

    write_summary_md(retrieval_results, hallucination_results, Path(__file__).parent / "ablation_summary.md")
    print("Ablation complete. See eval/ablation_summary.md for the formatted tables.")


if __name__ == "__main__":
    main()