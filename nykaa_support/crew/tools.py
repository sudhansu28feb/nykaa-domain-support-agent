"""
CrewAI Custom Tools for Nykaa Support Crew (Task 6 & Task 7).

Strictly complies with Capstone Project Brief:
1. check_order_status tool: wrapped as OrderStatusLookupTool (takes record_id: str).
2. rag_lookup tool: wrapped as RAGPolicyTool (takes query: str).
   Tests pitfall 2: named 'rag_lookup', but uses query argument schema.
3. Enforces least autonomy: only the Lookup Agent is wired with check_order_status.
"""

import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from dataset import check_order_status
from nykaa_support.rag.generator import GroundedGenerator
from nykaa_support.rag.store import COLLECTION_SENTENCE_BASED, RAGStore


class OrderStatusInput(BaseModel):
    """Declared argument schema for order status lookup."""
    record_id: str = Field(description="Order record identifier (e.g., NYK-10001)")


class OrderStatusLookupTool(BaseTool):
    """
    Task 6 compliant CrewAI tool:
    Looks up order status, order_value_inr, and calculated escalation_score.
    """
    name: str = "check_order_status"
    description: str = (
        "Lookup an order's status, order_value_inr, days_since_created, and designed escalation_score in [0, 1]. "
        "Input must be a valid record_id string (e.g., 'NYK-10001')."
    )
    args_schema: Type[BaseModel] = OrderStatusInput

    def _run(self, record_id: str) -> str:
        """Execute deterministic order lookup."""
        clean_id = record_id.strip().strip("'\"")
        res = check_order_status(clean_id)
        if "error" in res:
            return json.dumps({
                "status": "Not Found",
                "record_id": clean_id,
                "message": f"Order {clean_id} was not found in our records.",
                "escalation_score": 0.0,
                "escalate_recommended": False
            })

        status = res["status"]
        val = res["order_value_inr"]
        score = res["escalation_score"]
        is_delayed = res["delayed_shipment"]
        esc_rec = res["escalate_recommended"]
        days = res["days_since_created"]

        delay_str = "DELAYED past delivery SLA" if is_delayed else "on schedule"
        esc_str = "HUMAN ESCALATION RECOMMENDED" if esc_rec else "standard processing"

        summary = (
            f"Order {clean_id}: Status is '{status}', order value ₹{val}, placed {days} days ago ({delay_str}). "
            f"Calculated escalation score: {score:.4f} ({esc_str})."
        )
        return summary

    def execute_as(self, calling_agent_role: str, record_id: str) -> str:
        """
        Execute order lookup with least-autonomy role enforcement.
        Raises PermissionDeniedError if the calling agent is unauthorized.
        """
        from nykaa_support.governance.least_autonomy import authorize_tool_execution
        authorize_tool_execution(calling_agent_role, self.name)
        return self._run(record_id)


class RAGPolicyInput(BaseModel):
    """Declared argument schema for RAG policy retrieval."""
    query: str = Field(description="Natural language policy question to search in Nykaa policy documents")


class RAGPolicyTool(BaseTool):
    """
    Task 7 compliant RAG tool:
    Uses the recommended sentence-based collection (Task 5 winner) to retrieve policy context.
    Named 'rag_lookup' to explicitly test schema-based dispatch (Pitfall 2).
    """
    name: str = "rag_lookup"
    description: str = (
        "Retrieve official Nykaa customer support policy documents for return windows, "
        "refund timelines, delivery SLAs, warranties, cancellations, and claims. Input is a policy query string."
    )
    args_schema: Type[BaseModel] = RAGPolicyInput
    generator: Optional[GroundedGenerator] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.generator is None:
            store = RAGStore(local_files_only=True)
            store.ingest_documents()
            self.generator = GroundedGenerator(
                rag_store=store,
                default_collection=COLLECTION_SENTENCE_BASED
            )

    def _run(self, query: str) -> str:
        """Execute grounded policy retrieval and return context."""
        clean_q = query.strip().strip("'\"")
        answer_obj = self.generator.generate_answer(clean_q)
        if answer_obj.is_fallback:
            return f"Policy Search: No relevant policy found for '{clean_q}'. Threshold fallback triggered."

        context_passages = [c["text"] for c in answer_obj.retrieved_chunks if c["similarity"] >= 0.35]
        return "Relevant Nykaa Policy Context: " + " ".join(context_passages)

    def execute_as(self, calling_agent_role: str, query: str) -> str:
        """
        Execute RAG retrieval with least-autonomy role enforcement.
        Raises PermissionDeniedError if the calling agent is unauthorized.
        """
        from nykaa_support.governance.least_autonomy import authorize_tool_execution
        authorize_tool_execution(calling_agent_role, self.name)
        return self._run(query)
