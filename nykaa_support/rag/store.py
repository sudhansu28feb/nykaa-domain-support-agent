"""
ChromaDB Vector Store Management with Local SentenceTransformers (Task 3).

Strictly complies with Capstone Project Brief:
1. Embed every chunk with local SentenceTransformers ('all-MiniLM-L6-v2', local_files_only=True).
2. Index each chunking strategy in its OWN separate ChromaDB collection.
3. Uses collection.upsert() for deterministic and idempotent ingestion.
4. Preserves doc_id and parent metadata for precision/recall evaluation.
"""

import os
from typing import Any, Dict, List, Optional
import chromadb
from sentence_transformers import SentenceTransformer

from nykaa_support.rag.chunking import (
    DocumentChunk,
    chunk_fixed_size_with_overlap,
    chunk_sentence_based,
    load_policy_documents,
)

# Collection Names as required by dual chunking architecture
COLLECTION_FIXED_OVERLAP = "nykaa_policies_fixed_overlap"
COLLECTION_SENTENCE_BASED = "nykaa_policies_sentence_based"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGStore:
    """Manages dual local ChromaDB collections and SentenceTransformers embeddings."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        local_files_only: bool = True
    ):
        self.model_name = model_name
        # Load local weights without network lookup
        self.encoder = SentenceTransformer(model_name, local_files_only=local_files_only)

        # Initialize ChromaDB client (Persistent if directory provided, else Ephemeral in-memory)
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Create/Get separate collections with cosine distance metric
        self.col_fixed = self.client.get_or_create_collection(
            name=COLLECTION_FIXED_OVERLAP,
            metadata={"hnsw:space": "cosine", "strategy": "fixed_size_overlap"}
        )
        self.col_sentence = self.client.get_or_create_collection(
            name=COLLECTION_SENTENCE_BASED,
            metadata={"hnsw:space": "cosine", "strategy": "sentence_based"}
        )

    def ingest_documents(self, policies_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Load policy documents, compute chunks for both strategies, and index into separate collections
        using collection.upsert() as strictly required by Task 3.
        """
        raw_docs = load_policy_documents(policies_dir)
        all_fixed_chunks: List[DocumentChunk] = []
        all_sentence_chunks: List[DocumentChunk] = []

        for doc in raw_docs:
            doc_id = doc["doc_id"]
            title = doc["title"]
            body = doc["body"]

            # Strategy 1: Fixed-size with overlap
            fixed_chunks = chunk_fixed_size_with_overlap(
                text=body,
                doc_id=doc_id,
                topic_title=title,
                chunk_size=300,
                chunk_overlap=60
            )
            all_fixed_chunks.extend(fixed_chunks)

            # Strategy 2: Sentence-based
            sentence_chunks = chunk_sentence_based(
                text=body,
                doc_id=doc_id,
                topic_title=title
            )
            all_sentence_chunks.extend(sentence_chunks)

        # Upsert Strategy 1 into COLLECTION_FIXED_OVERLAP
        if all_fixed_chunks:
            fixed_ids = [c.chunk_id for c in all_fixed_chunks]
            fixed_texts = [c.text for c in all_fixed_chunks]
            fixed_metas = [
                {
                    "doc_id": c.doc_id,
                    "topic_title": c.topic_title,
                    "strategy": c.strategy,
                    "chunk_index": c.chunk_index,
                    "char_length": c.char_length
                }
                for c in all_fixed_chunks
            ]
            fixed_embeddings = self.encoder.encode(fixed_texts, show_progress_bar=False).tolist()
            self.col_fixed.upsert(
                ids=fixed_ids,
                embeddings=fixed_embeddings,
                documents=fixed_texts,
                metadatas=fixed_metas
            )

        # Upsert Strategy 2 into COLLECTION_SENTENCE_BASED
        if all_sentence_chunks:
            sent_ids = [c.chunk_id for c in all_sentence_chunks]
            sent_texts = [c.text for c in all_sentence_chunks]
            sent_metas = [
                {
                    "doc_id": c.doc_id,
                    "topic_title": c.topic_title,
                    "strategy": c.strategy,
                    "chunk_index": c.chunk_index,
                    "char_length": c.char_length
                }
                for c in all_sentence_chunks
            ]
            sent_embeddings = self.encoder.encode(sent_texts, show_progress_bar=False).tolist()
            self.col_sentence.upsert(
                ids=sent_ids,
                embeddings=sent_embeddings,
                documents=sent_texts,
                metadatas=sent_metas
            )

        return {
            "documents_loaded": len(raw_docs),
            "fixed_overlap_chunks": len(all_fixed_chunks),
            "sentence_based_chunks": len(all_sentence_chunks)
        }

    def query(
        self,
        collection_name: str,
        query_text: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Query a specific collection and return retrieved chunks with cosine similarities in [0, 1].
        """
        if collection_name == COLLECTION_FIXED_OVERLAP:
            col = self.col_fixed
        elif collection_name == COLLECTION_SENTENCE_BASED:
            col = self.col_sentence
        else:
            col = self.client.get_collection(collection_name)

        q_emb = self.encoder.encode([query_text], show_progress_bar=False).tolist()
        results = col.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved: List[Dict[str, Any]] = []
        if not results["ids"] or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for idx, chunk_id in enumerate(ids):
            dist = distances[idx] if idx < len(distances) else 1.0
            # Convert cosine distance d in [0, 2] to cosine similarity in [-1, 1], clamped to [0, 1]
            similarity = round(max(0.0, min(1.0, 1.0 - dist)), 4)
            meta = metas[idx] if idx < len(metas) else {}
            text = docs[idx] if idx < len(docs) else ""

            retrieved.append({
                "chunk_id": chunk_id,
                "doc_id": meta.get("doc_id", ""),
                "topic_title": meta.get("topic_title", ""),
                "strategy": meta.get("strategy", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity": similarity,
                "distance": round(dist, 4),
                "text": text,
                "rank": idx + 1
            })

        return retrieved

    def add_document_content(
        self,
        filename: str,
        content: str,
        topic_title: Optional[str] = None
    ) -> int:
        """
        Dynamically ingest a new policy document into the sentence-based vector store (Task 11).
        Returns the number of chunks added.
        """
        doc_id = os.path.splitext(filename)[0]
        title = topic_title or doc_id.replace("_", " ").title()

        chunks = chunk_sentence_based(
            text=content,
            doc_id=doc_id,
            topic_title=title
        )
        if not chunks:
            return 0

        chunk_ids = [c.chunk_id for c in chunks]
        chunk_texts = [c.text for c in chunks]
        chunk_metas = [
            {
                "doc_id": c.doc_id,
                "topic_title": c.topic_title,
                "strategy": c.strategy,
                "chunk_index": c.chunk_index,
                "char_length": c.char_length
            }
            for c in chunks
        ]
        embeddings = self.encoder.encode(chunk_texts, show_progress_bar=False).tolist()

        self.col_sentence.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=chunk_metas
        )
        return len(chunks)
