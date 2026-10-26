"""
Empirical Calibration Script for RAG Grounded Retrieval (Task 4).
Measures top-1 cosine similarities for in-scope and out-of-scope queries
across both chunking strategies to determine empirical threshold.
"""

from nykaa_support.rag.store import (
    COLLECTION_FIXED_OVERLAP,
    COLLECTION_SENTENCE_BASED,
    RAGStore,
)

IN_SCOPE_QUERIES = [
    "What is the return window for cosmetics and beauty products?",
    "How long does COD refund take via NEFT or Nykaa Wallet?",
    "What are the delivery SLAs for Tier 1 metro cities?",
    "Am I eligible for free reverse pickup on returns?",
    "What is the warranty coverage for electronic styling tools?",
]

OUT_OF_SCOPE_QUERIES = [
    "How do I bake a chocolate sourdough bread at home?",
    "What is the current stock market price of Tesla or Bitcoin?",
    "Who won the 2022 FIFA World Cup soccer championship in Qatar?",
]


def run_calibration():
    store = RAGStore(local_files_only=True)
    store.ingest_documents()

    print("=" * 80)
    print("TASK 4: EMPIRICAL CALIBRATION OF 'I DON'T KNOW' THRESHOLD")
    print("=" * 80)

    in_scope_fixed = []
    in_scope_sentence = []

    print("\n--- IN-SCOPE QUERIES (Expected High Semantic Match) ---")
    for q in IN_SCOPE_QUERIES:
        res_f = store.query(COLLECTION_FIXED_OVERLAP, q, top_k=1)[0]
        res_s = store.query(COLLECTION_SENTENCE_BASED, q, top_k=1)[0]
        in_scope_fixed.append(res_f["similarity"])
        in_scope_sentence.append(res_s["similarity"])
        print(f"Query: \"{q}\"")
        print(f"  Fixed-Size Overlap  : top-1 sim = {res_f['similarity']:.4f} | doc = {res_f['doc_id']}")
        print(f"  Sentence-Based      : top-1 sim = {res_s['similarity']:.4f} | doc = {res_s['doc_id']}")

    out_scope_fixed = []
    out_scope_sentence = []

    print("\n--- OUT-OF-SCOPE QUERIES (Expected Low Semantic Match) ---")
    for q in OUT_OF_SCOPE_QUERIES:
        res_f = store.query(COLLECTION_FIXED_OVERLAP, q, top_k=1)[0]
        res_s = store.query(COLLECTION_SENTENCE_BASED, q, top_k=1)[0]
        out_scope_fixed.append(res_f["similarity"])
        out_scope_sentence.append(res_s["similarity"])
        print(f"Query: \"{q}\"")
        print(f"  Fixed-Size Overlap  : top-1 sim = {res_f['similarity']:.4f} | doc = {res_f['doc_id']}")
        print(f"  Sentence-Based      : top-1 sim = {res_s['similarity']:.4f} | doc = {res_s['doc_id']}")

    print("\n" + "=" * 80)
    print("EMPIRICAL CLUSTER SUMMARY")
    print("=" * 80)
    print(f"Fixed-Size In-Scope Range   : [{min(in_scope_fixed):.4f}, {max(in_scope_fixed):.4f}] (mean: {sum(in_scope_fixed)/len(in_scope_fixed):.4f})")
    print(f"Fixed-Size Out-Scope Range  : [{min(out_scope_fixed):.4f}, {max(out_scope_fixed):.4f}] (mean: {sum(out_scope_fixed)/len(out_scope_fixed):.4f})")
    print(f"Sentence-Based In-Scope Range : [{min(in_scope_sentence):.4f}, {max(in_scope_sentence):.4f}] (mean: {sum(in_scope_sentence)/len(in_scope_sentence):.4f})")
    print(f"Sentence-Based Out-Scope Range: [{min(out_scope_sentence):.4f}, {max(out_scope_sentence):.4f}] (mean: {sum(out_scope_sentence)/len(out_scope_sentence):.4f})")


if __name__ == "__main__":
    run_calibration()
