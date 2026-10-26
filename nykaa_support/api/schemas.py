"""
Pydantic Schemas for Nykaa Support FastAPI Endpoints (Task 11).
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Customer query request schema for POST /ask."""
    query: str = Field(..., min_length=1, description="Customer question or order inquiry")
    session_id: Optional[str] = Field(default=None, description="Optional conversational session identifier")


class AskResponse(BaseModel):
    """Response schema for POST /ask."""
    query: str = Field(description="Sanitized customer query")
    final_answer: str = Field(description="Final customer resolution after compliance review")
    query_type: str = Field(description="Query categorization ('policy', 'order', 'combined', 'injection_blocked', etc.)")
    order_id: Optional[str] = Field(default=None, description="Extracted order identifier if applicable")
    order_data: Optional[Dict[str, Any]] = Field(default=None, description="Order details if looked up")
    retrieved_policy_docs: List[str] = Field(default_factory=list, description="Policy documents retrieved")
    escalation_score: float = Field(default=0.0, description="Escalation score in [0.0, 1.0]")
    escalation_recommended: bool = Field(default=False, description="Whether escalation to human tier is recommended")
    review_verdict: Optional[Dict[str, Any]] = Field(default=None, description="Structured AutoGen review verdict")
    session_id: str = Field(description="Session identifier tracking conversation context")
    trace_id: str = Field(description="Unique request tracing UUID")


class AddDocumentRequest(BaseModel):
    """Dynamic document ingestion request schema for POST /add-document."""
    filename: str = Field(..., min_length=3, description="Policy document filename (e.g., '13_loyalty_tier_policy.md')")
    content: str = Field(..., min_length=10, description="Policy document body text")
    category: Optional[str] = Field(default="policy", description="Knowledge base topic category")


class AddDocumentResponse(BaseModel):
    """Response schema for POST /add-document."""
    status: str = Field(description="Ingestion status ('success' or 'error')")
    filename: str = Field(description="Name of the indexed policy document")
    chunks_added: int = Field(description="Number of sentence chunks added to vector store")
    collection_name: str = Field(description="ChromaDB collection updated")
    trace_id: str = Field(description="Unique request tracing UUID")


class HealthResponse(BaseModel):
    """Service health response schema for GET /health."""
    status: str = "healthy"
    service: str = "nykaa-domain-support-agent"
    offline_mode: bool = True
    risk_level: str = "Medium"
    rag_collection: str = "nykaa_policies_sentence_based"
