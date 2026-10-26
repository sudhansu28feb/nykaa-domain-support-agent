"""
Pydantic Structured Output Schema for Nykaa Support Crew (Task 9).

Strictly complies with the Capstone Project Brief:
1. Defines a Pydantic BaseModel that every crew response must conform to.
2. Validates types, required fields, and score bounds [0.0, 1.0].
3. Explicit error handling for malformed outputs without silent acceptance.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


QueryType = Literal["order_status", "policy_inquiry", "combined", "out_of_scope", "general"]


class OrderDetails(BaseModel):
    """Structured order data extracted or looked up."""
    record_id: str = Field(description="Order record identifier (e.g., NYK-10001)")
    status: str = Field(description="Order status (Placed, Shipped, Delivered, Returned, Refunded, or Unknown)")
    order_value_inr: int = Field(ge=0, description="Order value in Indian Rupees")
    days_since_created: int = Field(ge=0, le=365, description="Days since order was placed")
    delayed_shipment: bool = Field(description="Whether order is delayed past SLA")
    escalation_score: float = Field(ge=0.0, le=1.0, description="Calculated escalation score in [0.0, 1.0]")
    escalate_recommended: bool = Field(description="Whether escalation threshold >= 0.72 is breached")


class CrewSupportResponse(BaseModel):
    """
    Mandatory Pydantic model for all Nykaa Support Crew outputs (Task 9).
    """
    query: str = Field(min_length=1, description="Original customer query")
    final_answer: str = Field(min_length=1, description="Synthesized customer-facing response text")
    query_type: QueryType = Field(description="Categorical classification of the inquiry")
    order_id: Optional[str] = Field(default=None, description="Extracted or referenced order ID")
    order_data: Optional[OrderDetails] = Field(default=None, description="Structured order fulfillment data if applicable")
    retrieved_policy_docs: List[str] = Field(default_factory=list, description="Referenced policy document titles")
    escalation_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Escalation score in [0.0, 1.0]")
    escalation_recommended: bool = Field(default=False, description="Flag indicating human escalation recommendation")

    @field_validator("escalation_score")
    @classmethod
    def validate_escalation_score_bounds(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError(f"escalation_score must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("query")
    @classmethod
    def validate_non_empty_query(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace only")
        return v.strip()

    @field_validator("final_answer")
    @classmethod
    def validate_non_empty_answer(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("final_answer cannot be empty or whitespace only")
        return v.strip()


class StructuredOutputError(Exception):
    """Raised when crew response validation explicitly fails."""
    def __init__(self, message: str, errors: Optional[List[Dict[str, Any]]] = None):
        super().__init__(message)
        self.errors = errors or []


def validate_crew_response(raw_input: Any) -> CrewSupportResponse:
    """
    Strict validation of raw data against the CrewSupportResponse model.
    Raises StructuredOutputError on any schema or value violation.
    """
    if isinstance(raw_input, CrewSupportResponse):
        return raw_input

    try:
        if isinstance(raw_input, dict):
            return CrewSupportResponse.model_validate(raw_input)
        elif isinstance(raw_input, str):
            # Attempt JSON parse first
            try:
                data = json.loads(raw_input)
                if isinstance(data, dict):
                    return CrewSupportResponse.model_validate(data)
            except json.JSONDecodeError:
                pass
            raise StructuredOutputError(
                "Raw string input could not be validated as structured CrewSupportResponse JSON.",
                errors=[{"input": raw_input, "reason": "Not valid JSON dictionary"}]
            )
        else:
            raise StructuredOutputError(
                f"Unsupported payload type for CrewSupportResponse: {type(raw_input)}",
                errors=[{"type": str(type(raw_input))}]
            )
    except ValidationError as ve:
        raise StructuredOutputError(
            f"Validation failed against CrewSupportResponse schema: {ve.error_count()} error(s)",
            errors=ve.errors()
        ) from ve


def format_crew_response(
    query: str,
    final_answer: str,
    order_info: Optional[Dict[str, Any]] = None,
    policy_docs: Optional[List[str]] = None
) -> CrewSupportResponse:
    """
    Factory helper to reliably build a validated CrewSupportResponse.
    """
    order_details = None
    esc_score = None
    esc_rec = False
    order_id = None

    if order_info and "error" not in order_info and order_info.get("status") != "Unknown":
        order_id = order_info.get("record_id")
        esc_score = order_info.get("escalation_score")
        esc_rec = order_info.get("escalate_recommended", False)
        order_details = OrderDetails(
            record_id=order_info.get("record_id", "NYK-UNKNOWN"),
            status=order_info.get("status", "Unknown"),
            order_value_inr=order_info.get("order_value_inr", 0),
            days_since_created=order_info.get("days_since_created", 0),
            delayed_shipment=order_info.get("delayed_shipment", False),
            escalation_score=order_info.get("escalation_score", 0.0),
            escalate_recommended=esc_rec
        )

    # Determine query type
    has_policy = bool(policy_docs)
    has_order = bool(order_id)
    if has_policy and has_order:
        q_type: QueryType = "combined"
    elif has_order:
        q_type = "order_status"
    elif has_policy:
        q_type = "policy_inquiry"
    else:
        q_type = "general"

    resp = CrewSupportResponse(
        query=query,
        final_answer=final_answer,
        query_type=q_type,
        order_id=order_id,
        order_data=order_details,
        retrieved_policy_docs=policy_docs or [],
        escalation_score=esc_score,
        escalation_recommended=esc_rec
    )
    return validate_crew_response(resp)
