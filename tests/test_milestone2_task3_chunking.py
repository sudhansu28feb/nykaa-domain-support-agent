"""
Comprehensive Test Suite for Task 3:
Dual Chunking Strategies, Metadata Preservation, and Separate ChromaDB Collections.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from nykaa_support.rag.chunking import (
    chunk_fixed_size_with_overlap,
    chunk_sentence_based,
    load_policy_documents,
)
from nykaa_support.rag.store import (
    COLLECTION_FIXED_OVERLAP,
    COLLECTION_SENTENCE_BASED,
    RAGStore,
)


def test_load_all_policy_documents():
    """Verify loading all 12 distinct policy documents with titles and bodies."""
    docs = load_policy_documents()
    assert len(docs) >= 12, f"Expected at least 12 documents, found {len(docs)}"

    for doc in docs:
        assert doc["doc_id"].startswith("0") or doc["doc_id"].startswith("1")
        assert len(doc["title"]) > 5
        assert len(doc["body"]) > 100
        assert doc["filename"].endswith(".md")

    print(f"test_load_all_policy_documents: PASSED ({len(docs)} documents loaded successfully)")


def test_fixed_size_chunking_with_overlap():
    """Verify fixed-size chunking with character overlap and parent metadata preservation."""
    sample_text = (
        "Nykaa maintains category-specific return windows to balance customer flexibility with product hygiene standards. "
        "Beauty and cosmetics items can be returned within 15 calendar days of delivery only if seals remain unopened. "
        "Apparel and footwear items are eligible for returns within 15 calendar days provided original tags are intact. "
        "Home decor products must be returned within 10 calendar days in original condition."
    )
    doc_id = "01_return_window_by_category"
    title = "Return Window by Product Category"

    chunks = chunk_fixed_size_with_overlap(
        text=sample_text,
        doc_id=doc_id,
        topic_title=title,
        chunk_size=180,
        chunk_overlap=40
    )

    assert len(chunks) >= 2, f"Expected multiple chunks for 400+ chars, got {len(chunks)}"

    for idx, c in enumerate(chunks):
        assert c.doc_id == doc_id
        assert c.topic_title == title
        assert c.strategy == "fixed_size_overlap"
        assert c.chunk_index == idx
        assert c.char_length == len(c.text)
        assert c.char_length <= 180 + 10  # Allow small margin for whitespace boundaries

    # Verify overlap exists between chunk 0 and chunk 1
    # Check that a substring from end of chunk 0 appears in beginning of chunk 1
    chunk0_words = c_words = chunks[0].text.split()
    overlap_tail = " ".join(chunk0_words[-3:])
    assert overlap_tail in chunks[1].text, "Expected word overlap between consecutive chunks"

    print(f"test_fixed_size_chunking_with_overlap: PASSED ({len(chunks)} chunks, overlap verified)")


def test_sentence_based_chunking():
    """Verify sentence-based chunking produces atomic policy statements with full metadata."""
    sample_text = (
        "Standard domestic orders are fulfilled across India with reliable courier partners. "
        "Shipments to Tier 1 metro regions are delivered within 2 to 3 business days from warehouse dispatch. "
        "Deliveries to Tier 2 and Tier 3 cities require 4 to 6 business days. "
        "Nykaa provides free standard shipping on all orders valued at ₹499 and above."
    )
    doc_id = "03_delivery_slas"
    title = "Delivery Service Level Agreements (SLAs)"

    chunks = chunk_sentence_based(
        text=sample_text,
        doc_id=doc_id,
        topic_title=title
    )

    assert len(chunks) == 4, f"Expected 4 sentence chunks, got {len(chunks)}"

    for idx, c in enumerate(chunks):
        assert c.doc_id == doc_id
        assert c.topic_title == title
        assert c.strategy == "sentence_based"
        assert c.chunk_index == idx
        assert c.text.endswith(".")
        assert c.char_length == len(c.text)

    print(f"test_sentence_based_chunking: PASSED ({len(chunks)} sentence chunks verified)")


def test_chromadb_collection_separation_and_upsert():
    """Verify separate ChromaDB collections, local SentenceTransformer embedding, and collection.upsert()."""
    # Create in-memory RAGStore
    rag_store = RAGStore(local_files_only=True)

    # Ingest all policy documents
    stats = rag_store.ingest_documents()
    print(f"\n[RAG Ingestion Statistics]: {stats}")

    assert stats["documents_loaded"] >= 12
    assert stats["fixed_overlap_chunks"] > 20
    assert stats["sentence_based_chunks"] >= 48

    # Verify collections exist separately in ChromaDB
    col_names = [c.name for c in rag_store.client.list_collections()]
    assert COLLECTION_FIXED_OVERLAP in col_names, f"Missing {COLLECTION_FIXED_OVERLAP}"
    assert COLLECTION_SENTENCE_BASED in col_names, f"Missing {COLLECTION_SENTENCE_BASED}"

    # Verify item count in each collection
    fixed_count = rag_store.col_fixed.count()
    sentence_count = rag_store.col_sentence.count()
    assert fixed_count == stats["fixed_overlap_chunks"]
    assert sentence_count == stats["sentence_based_chunks"]

    # Query both collections with a realistic sample policy query
    test_query = "What is the return window for cosmetics and beauty products?"

    res_fixed = rag_store.query(COLLECTION_FIXED_OVERLAP, test_query, top_k=2)
    res_sentence = rag_store.query(COLLECTION_SENTENCE_BASED, test_query, top_k=2)

    assert len(res_fixed) == 2
    assert len(res_sentence) == 2

    # Top result in both should relate to return window
    assert res_fixed[0]["doc_id"] == "01_return_window_by_category"
    assert res_sentence[0]["doc_id"] == "01_return_window_by_category"

    # Both similarities should be high (> 0.50)
    assert res_fixed[0]["similarity"] > 0.50
    assert res_sentence[0]["similarity"] > 0.50

    print("test_chromadb_collection_separation_and_upsert: PASSED (Both collections queried with sensible retrieval)")
    print(f"  - Fixed chunk top similarity: {res_fixed[0]['similarity']:.4f} ({res_fixed[0]['doc_id']})")
    print(f"  - Sentence chunk top similarity: {res_sentence[0]['similarity']:.4f} ({res_sentence[0]['doc_id']})")


def test_repeatable_idempotent_ingestion():
    """Verify collection.upsert() is idempotent and does not create duplicate entries on repeated runs."""
    rag_store = RAGStore(local_files_only=True)

    # First ingestion
    stats_first = rag_store.ingest_documents()
    count_fixed_first = rag_store.col_fixed.count()
    count_sent_first = rag_store.col_sentence.count()

    # Second ingestion (upsert should update rather than duplicate)
    stats_second = rag_store.ingest_documents()
    count_fixed_second = rag_store.col_fixed.count()
    count_sent_second = rag_store.col_sentence.count()

    assert count_fixed_first == count_fixed_second, "Fixed collection count changed on repeated upsert"
    assert count_sent_first == count_sent_second, "Sentence collection count changed on repeated upsert"

    print("test_repeatable_idempotent_ingestion: PASSED (collection.upsert() is 100% idempotent)")


if __name__ == "__main__":
    test_load_all_policy_documents()
    test_fixed_size_chunking_with_overlap()
    test_sentence_based_chunking()
    test_chromadb_collection_separation_and_upsert()
    test_repeatable_idempotent_ingestion()
    print("\nAll Task 3 Dual Chunking and Local Vector Store tests PASSED!")
