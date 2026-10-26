"""
RAG Subsystem for Nykaa Domain Support Agent.
"""

from nykaa_support.rag.chunking import (
    DocumentChunk,
    chunk_fixed_size_with_overlap,
    chunk_sentence_based,
    load_policy_documents,
)
from nykaa_support.rag.store import (
    COLLECTION_FIXED_OVERLAP,
    COLLECTION_SENTENCE_BASED,
    RAGStore,
)
from nykaa_support.rag.generator import (
    EMPIRICAL_SIMILARITY_THRESHOLD,
    FALLBACK_RESPONSE_TEXT,
    GroundedAnswer,
    GroundedGenerator,
)
from nykaa_support.rag.comparison import (
    BENCHMARK_GROUND_TRUTH,
    ComparisonReport,
    QueryEvaluationResult,
    evaluate_strategy_on_queries,
    run_comparative_evaluation,
)

__all__ = [
    "DocumentChunk",
    "chunk_fixed_size_with_overlap",
    "chunk_sentence_based",
    "load_policy_documents",
    "COLLECTION_FIXED_OVERLAP",
    "COLLECTION_SENTENCE_BASED",
    "RAGStore",
    "EMPIRICAL_SIMILARITY_THRESHOLD",
    "FALLBACK_RESPONSE_TEXT",
    "GroundedAnswer",
    "GroundedGenerator",
    "BENCHMARK_GROUND_TRUTH",
    "ComparisonReport",
    "QueryEvaluationResult",
    "evaluate_strategy_on_queries",
    "run_comparative_evaluation",
]
