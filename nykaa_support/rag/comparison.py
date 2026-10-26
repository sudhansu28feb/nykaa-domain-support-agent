"""
Comparative Evaluation of Chunking Strategies (Task 5).

Calculates document-level Precision and Recall across the 5 in-scope queries from Task 4
for both chunking strategies (Fixed-Size with Overlap vs. Sentence-Based).
Preserves parent document mapping and deduplicates parent document IDs before scoring.
"""

from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from nykaa_support.rag.store import (
    COLLECTION_FIXED_OVERLAP,
    COLLECTION_SENTENCE_BASED,
    RAGStore,
)

# 5 In-Scope Benchmark Queries from Task 4 with Ground Truth Parent Document IDs
BENCHMARK_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "query_id": "Q1",
        "query": "What is the return window for cosmetics and beauty products?",
        "ground_truth_doc_ids": ["01_return_window_by_category"],
    },
    {
        "query_id": "Q2",
        "query": "How long does COD refund take via NEFT or Nykaa Wallet?",
        "ground_truth_doc_ids": ["02_cod_refund_timelines"],
    },
    {
        "query_id": "Q3",
        "query": "What are the delivery SLAs for Tier 1 metro cities?",
        "ground_truth_doc_ids": ["03_delivery_slas"],
    },
    {
        "query_id": "Q4",
        "query": "Am I eligible for free reverse pickup on returns?",
        "ground_truth_doc_ids": ["04_reverse_pickup_eligibility"],
    },
    {
        "query_id": "Q5",
        "query": "What is the warranty coverage for electronic styling tools?",
        "ground_truth_doc_ids": ["05_warranty_terms_by_category"],
    },
]


class QueryEvaluationResult(BaseModel):
    query_id: str
    query: str
    strategy: str
    top_k: int
    raw_retrieved_doc_ids: List[str]
    deduped_doc_ids: List[str]
    ground_truth_doc_ids: List[str]
    relevant_retrieved: List[str]
    precision: float
    recall: float
    arithmetic_str: str


class ComparisonReport(BaseModel):
    fixed_overlap_results: List[QueryEvaluationResult]
    sentence_based_results: List[QueryEvaluationResult]
    fixed_mean_precision: float
    fixed_mean_recall: float
    sentence_mean_precision: float
    sentence_mean_recall: float
    recommendation: str


def evaluate_strategy_on_queries(
    store: RAGStore,
    collection_name: str,
    top_k: int = 3
) -> List[QueryEvaluationResult]:
    """
    Compute document-level precision and recall for a collection on benchmark queries.
    Deduplicates parent-document IDs before calculating metrics.
    """
    results: List[QueryEvaluationResult] = []

    for item in BENCHMARK_GROUND_TRUTH:
        q_id = item["query_id"]
        q_text = item["query"]
        gt_docs: Set[str] = set(item["ground_truth_doc_ids"])

        # Retrieve top-k chunks
        retrieved_chunks = store.query(collection_name, q_text, top_k=top_k)
        raw_doc_ids = [c["doc_id"] for c in retrieved_chunks]

        # Deduplicate retrieved parent document IDs while preserving order
        deduped_doc_ids: List[str] = []
        for d in raw_doc_ids:
            if d and d not in deduped_doc_ids:
                deduped_doc_ids.append(d)

        retrieved_set = set(deduped_doc_ids)
        relevant_retrieved = list(retrieved_set & gt_docs)

        # Precision = |Relevant Retrieved| / |Total Retrieved Unique Docs|
        # Recall    = |Relevant Retrieved| / |Total Ground Truth Docs|
        num_rel_ret = len(relevant_retrieved)
        num_total_ret = len(deduped_doc_ids)
        num_gt = len(gt_docs)

        precision = round(num_rel_ret / max(1, num_total_ret), 4)
        recall = round(num_rel_ret / max(1, num_gt), 4)

        arithmetic_str = (
            f"Precision = {num_rel_ret}/{num_total_ret} = {precision:.4f} | "
            f"Recall = {num_rel_ret}/{num_gt} = {recall:.4f}"
        )

        results.append(
            QueryEvaluationResult(
                query_id=q_id,
                query=q_text,
                strategy=collection_name,
                top_k=top_k,
                raw_retrieved_doc_ids=raw_doc_ids,
                deduped_doc_ids=deduped_doc_ids,
                ground_truth_doc_ids=list(gt_docs),
                relevant_retrieved=relevant_retrieved,
                precision=precision,
                recall=recall,
                arithmetic_str=arithmetic_str
            )
        )

    return results


def run_comparative_evaluation(store: Optional[RAGStore] = None, top_k: int = 3) -> ComparisonReport:
    """
    Run full comparison across both collections and generate recommendation report.
    """
    rag_store = store or RAGStore(local_files_only=True)
    rag_store.ingest_documents()

    fixed_res = evaluate_strategy_on_queries(rag_store, COLLECTION_FIXED_OVERLAP, top_k=top_k)
    sent_res = evaluate_strategy_on_queries(rag_store, COLLECTION_SENTENCE_BASED, top_k=top_k)

    fixed_prec = round(sum(r.precision for r in fixed_res) / len(fixed_res), 4)
    fixed_rec = round(sum(r.recall for r in fixed_res) / len(fixed_res), 4)

    sent_prec = round(sum(r.precision for r in sent_res) / len(sent_res), 4)
    sent_rec = round(sum(r.recall for r in sent_res) / len(sent_res), 4)

    # 2-3 sentence recommendation citing numbers
    if sent_prec >= fixed_prec and sent_rec >= fixed_rec:
        recommendation = (
            f"We recommend deploying the Sentence-Based chunking strategy ({COLLECTION_SENTENCE_BASED}), "
            f"which achieved a Mean Document Precision of {sent_prec:.2%} and Mean Document Recall of {sent_rec:.2%}, "
            f"outperforming the Fixed-Size Overlap strategy ({fixed_prec:.2%} precision, {fixed_rec:.2%} recall). "
            f"Sentence-level chunking isolates concise, atomic policy rules without cross-topic bleeding from adjacent markdown headings."
        )
    else:
        recommendation = (
            f"We recommend deploying the Fixed-Size Overlap chunking strategy ({COLLECTION_FIXED_OVERLAP}), "
            f"which achieved a Mean Document Precision of {fixed_prec:.2%} and Mean Document Recall of {fixed_rec:.2%}, "
            f"compared to Sentence-Based chunking ({sent_prec:.2%} precision, {sent_rec:.2%} recall). "
            f"The fixed window with overlap preserves transitional context between adjacent sentences that improves retrieval density."
        )

    return ComparisonReport(
        fixed_overlap_results=fixed_res,
        sentence_based_results=sent_res,
        fixed_mean_precision=fixed_prec,
        fixed_mean_recall=fixed_rec,
        sentence_mean_precision=sent_prec,
        sentence_mean_recall=sent_rec,
        recommendation=recommendation
    )


if __name__ == "__main__":
    report = run_comparative_evaluation()
    print("=" * 80)
    print("TASK 5: COMPARATIVE EVALUATION OF CHUNKING STRATEGIES (DOCUMENT-LEVEL)")
    print("=" * 80)
    print("\n--- 1. FIXED-SIZE WITH OVERLAP (COLLECTION: nykaa_policies_fixed_overlap) ---")
    for r in report.fixed_overlap_results:
        print(f"[{r.query_id}] \"{r.query}\"")
        print(f"     Raw Retrieved Docs     : {r.raw_retrieved_doc_ids}")
        print(f"     Deduped Docs           : {r.deduped_doc_ids}")
        print(f"     Ground Truth Docs      : {r.ground_truth_doc_ids}")
        print(f"     Per-Query Arithmetic   : {r.arithmetic_str}")

    print(f"\nFixed-Size Summary: Mean Precision = {report.fixed_mean_precision:.4f} | Mean Recall = {report.fixed_mean_recall:.4f}")

    print("\n--- 2. SENTENCE-BASED (COLLECTION: nykaa_policies_sentence_based) ---")
    for r in report.sentence_based_results:
        print(f"[{r.query_id}] \"{r.query}\"")
        print(f"     Raw Retrieved Docs     : {r.raw_retrieved_doc_ids}")
        print(f"     Deduped Docs           : {r.deduped_doc_ids}")
        print(f"     Ground Truth Docs      : {r.ground_truth_doc_ids}")
        print(f"     Per-Query Arithmetic   : {r.arithmetic_str}")

    print(f"\nSentence-Based Summary: Mean Precision = {report.sentence_mean_precision:.4f} | Mean Recall = {report.sentence_mean_recall:.4f}")

    print("\n--- 3. NUMBERS-CITED RECOMMENDATION ---")
    print(report.recommendation)
    print("=" * 80)
