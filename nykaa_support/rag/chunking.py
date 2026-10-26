"""
Chunking Strategies for Nykaa Domain Policy Documents (Task 3).

Implements both required chunking strategies:
1. Fixed-size chunking with overlap (sliding window).
2. Sentence-based chunking.

Each chunk preserves parent document metadata and doc_id for precision/recall evaluation.
"""

import os
import re
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Pydantic schema representing an indexed document chunk with parent lineage."""
    chunk_id: str = Field(description="Unique chunk identifier")
    doc_id: str = Field(description="Parent document identifier (e.g. 01_return_window_by_category)")
    topic_title: str = Field(description="Title extracted from policy document header")
    text: str = Field(description="Clean chunk text content")
    strategy: str = Field(description="Chunking strategy: 'fixed_size_overlap' or 'sentence_based'")
    chunk_index: int = Field(description="Zero-based sequence index of chunk within the parent document")
    char_length: int = Field(description="Character length of chunk text")


def load_policy_documents(policies_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Read all markdown documents from data/policies/.
    Returns list of dictionaries containing doc_id, filename, title, and body text.
    """
    if policies_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        policies_dir = os.path.join(base_dir, "data", "policies")

    if not os.path.exists(policies_dir):
        raise FileNotFoundError(f"Policies directory not found: {policies_dir}")

    docs: List[Dict[str, str]] = []
    filenames = sorted([f for f in os.listdir(policies_dir) if f.endswith(".md")])

    for fname in filenames:
        doc_id = os.path.splitext(fname)[0]
        filepath = os.path.join(policies_dir, fname)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        title = "Nykaa Support Policy"
        body_lines: List[str] = []

        for line in lines:
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
            elif not line.startswith("#"):
                body_lines.append(line)

        body_text = " ".join(body_lines)
        docs.append({
            "doc_id": doc_id,
            "filename": fname,
            "title": title,
            "body": body_text,
            "raw": raw_text
        })

    return docs


def chunk_fixed_size_with_overlap(
    text: str,
    doc_id: str,
    topic_title: str,
    chunk_size: int = 300,
    chunk_overlap: int = 60
) -> List[DocumentChunk]:
    """
    Strategy 1: Fixed-size character window with overlap.
    Breaks text on whitespace boundaries when possible to preserve words.
    """
    chunks: List[DocumentChunk] = []
    clean_text = " ".join(text.split())
    text_len = len(clean_text)

    if text_len == 0:
        return chunks

    if text_len <= chunk_size:
        return [
            DocumentChunk(
                chunk_id=f"{doc_id}_fix_0",
                doc_id=doc_id,
                topic_title=topic_title,
                text=clean_text,
                strategy="fixed_size_overlap",
                chunk_index=0,
                char_length=text_len
            )
        ]

    step = max(1, chunk_size - chunk_overlap)
    start = 0
    idx = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_slice = clean_text[start:end].strip()

        # If not at end of text, avoid slicing in the middle of a word if feasible
        if end < text_len and " " in chunk_slice:
            last_space = chunk_slice.rfind(" ")
            if last_space > chunk_size * 0.6:  # Only adjust if space is reasonably far in
                end = start + last_space
                chunk_slice = clean_text[start:end].strip()

        if chunk_slice:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}_fix_{idx}",
                    doc_id=doc_id,
                    topic_title=topic_title,
                    text=chunk_slice,
                    strategy="fixed_size_overlap",
                    chunk_index=idx,
                    char_length=len(chunk_slice)
                )
            )
            idx += 1

        if end >= text_len:
            break
        start += step

    return chunks


def chunk_sentence_based(
    text: str,
    doc_id: str,
    topic_title: str
) -> List[DocumentChunk]:
    """
    Strategy 2: Sentence-based chunking.
    Splits text on terminal punctuation boundaries (.!?), preserving atomic policy facts.
    """
    clean_text = " ".join(text.split())
    # Split by sentence end boundaries followed by space or end of string
    raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    chunks: List[DocumentChunk] = []
    for idx, sentence in enumerate(sentences):
        chunks.append(
            DocumentChunk(
                chunk_id=f"{doc_id}_sent_{idx}",
                doc_id=doc_id,
                topic_title=topic_title,
                text=sentence,
                strategy="sentence_based",
                chunk_index=idx,
                char_length=len(sentence)
            )
        )

    return chunks
