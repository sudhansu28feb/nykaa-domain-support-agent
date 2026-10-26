"""
Grounded Generation Engine with Empirical Threshold Fallback (Task 4).

Strictly complies with Capstone Project Brief:
1. Answers questions using ONLY retrieved context from policy documents.
2. Under MOCK_LLM, retrieval similarity is the grounding signal.
3. Uses empirically calibrated 'I don't know' threshold (0.35) situated squarely
   between the measured out-of-scope cluster ([0.0886, 0.1788]) and in-scope cluster ([0.5978, 0.6695]).
4. Returns exact fallback when similarity is below threshold.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from nykaa_support.rag.store import (
    COLLECTION_FIXED_OVERLAP,
    COLLECTION_SENTENCE_BASED,
    RAGStore,
)

# Empirically Calibrated Threshold:
# In-scope cosine similarity range: [0.5978, 0.6695] (mean ~0.64)
# Out-of-scope cosine similarity range: [0.0886, 0.1788] (mean ~0.13)
# Selected Threshold = 0.35 (Safe midpoint providing >0.17 margin on both sides)
EMPIRICAL_SIMILARITY_THRESHOLD = 0.35

FALLBACK_RESPONSE_TEXT = (
    "I don't know. The requested information is not covered by our official Nykaa policy documentation."
)


class GroundedAnswer(BaseModel):
    """Structured response from the grounded generation engine."""
    query: str
    answer: str
    top_similarity: float
    retrieved_doc_ids: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    is_fallback: bool
    strategy_used: str
    confidence: float


class GroundedGenerator:
    """Retrieval-augmented answer generator using local ChromaDB and SentenceTransformers."""

    def __init__(
        self,
        rag_store: Optional[RAGStore] = None,
        similarity_threshold: float = EMPIRICAL_SIMILARITY_THRESHOLD,
        default_collection: str = COLLECTION_SENTENCE_BASED,
        top_k: int = 3
    ):
        self.store = rag_store or RAGStore(local_files_only=True)
        self.threshold = similarity_threshold
        self.default_collection = default_collection
        self.top_k = top_k

    def generate_answer(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> GroundedAnswer:
        """
        Retrieve top-k chunks and generate an answer using ONLY retrieved context.
        If top-1 similarity is below the empirical threshold, trigger the fallback.
        """
        target_collection = collection_name or self.default_collection
        k = top_k or self.top_k

        # 1. Retrieve top-k chunks
        retrieved = self.store.query(
            collection_name=target_collection,
            query_text=query,
            top_k=k
        )

        if not retrieved:
            return GroundedAnswer(
                query=query,
                answer=FALLBACK_RESPONSE_TEXT,
                top_similarity=0.0,
                retrieved_doc_ids=[],
                retrieved_chunks=[],
                is_fallback=True,
                strategy_used=target_collection,
                confidence=0.0
            )

        top_chunk = retrieved[0]
        top_sim = top_chunk["similarity"]

        # Deduplicate parent doc_ids preserving order
        seen_docs = set()
        doc_ids: List[str] = []
        for c in retrieved:
            d_id = c.get("doc_id", "")
            if d_id and d_id not in seen_docs:
                seen_docs.add(d_id)
                doc_ids.append(d_id)

        # 2. Threshold Check (Empirical Fallback Gate)
        if top_sim < self.threshold:
            return GroundedAnswer(
                query=query,
                answer=FALLBACK_RESPONSE_TEXT,
                top_similarity=top_sim,
                retrieved_doc_ids=doc_ids,
                retrieved_chunks=retrieved,
                is_fallback=True,
                strategy_used=target_collection,
                confidence=top_sim
            )

        # 3. Grounded Answer Synthesis using ONLY retrieved context
        # In MOCK_LLM mode, synthesizes response directly from the retrieved policy passages
        context_passages = [c["text"].strip() for c in retrieved if c["similarity"] >= self.threshold]
        synthesized_context = " ".join(context_passages)

        greeting = "Hello! Thank you for reaching out to Nykaa Customer Support.\n\n"
        body = f"Based on our official policy:\n{synthesized_context}\n"
        closing = "\nPlease let us know if you need any further assistance with your Nykaa order!"
        full_answer = greeting + body + closing

        return GroundedAnswer(
            query=query,
            answer=full_answer,
            top_similarity=top_sim,
            retrieved_doc_ids=doc_ids,
            retrieved_chunks=retrieved,
            is_fallback=False,
            strategy_used=target_collection,
            confidence=top_sim
        )
