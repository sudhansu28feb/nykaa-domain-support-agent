"""
Four-Layer AI Governance Models & Risk Classification for Nykaa Support Agent (Task 15).

Strictly complies with Capstone Project Brief (Pages 8-9):
1. Application Layer: Principle of Least Autonomy & Risk Classification (Low/Medium/High).
2. Runtime Layer: Per-request token and cost budget caps.
3. Model Layer: Grounded generation and AutoGen review stage.
4. Infrastructure Layer: Local models, in-process socket connection interceptor, and zero telemetry.
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SystemRiskClassification(BaseModel):
    """
    Formal Risk Classification complying with the Capstone Project Brief taxonomy:
    - Low: Summarization, transcription
    - Medium: Code generation, customer support tickets
    - High: Medical data, hiring decisions, financial underwriting
    """
    system_name: str = "Nykaa Domain Support Agent"
    risk_level: RiskLevel = RiskLevel.MEDIUM
    category: str = "Customer support tickets & order operations"
    framework: str = "Four-Layer AI Governance Model"
    rationale: str = (
        "The Nykaa Domain Support Agent operates at the Medium Risk tier under the four-layer "
        "AI governance framework. Under the brief's taxonomy, Low Risk encompasses read-only summarization "
        "and transcription, Medium Risk governs customer support tickets and code generation, while High Risk "
        "is reserved for irreversible medical decisions, automated hiring screening, and banking/credit data. "
        "Because this system autonomously handles customer support inquiries concerning live order fulfillment, "
        "monetary return and refund policies, delayed shipment escalations, and product claims, an erroneous or "
        "hallucinated commitment could produce customer friction, financial reconciliation discrepancies, or "
        "unauthorized return approvals. However, it does not perform irreversible financial transactions or "
        "process high-stakes sensitive health data. Medium Risk is therefore the precise classification, "
        "mandating application-layer least autonomy, input PII masking, continuous escalation scoring, "
        "runtime budget caps, and multi-agent AutoGen compliance review."
    )
    governance_controls: Dict[str, str] = Field(
        default_factory=lambda: {
            "Application Layer": "Strict least-autonomy tool gating (Lookup Agent only for order status), PII masking.",
            "Model Layer": "Grounded generation with 0.35 similarity cutoff, 2-agent AutoGen compliance review stage.",
            "Runtime Layer": "Pre-execution per-request token/cost budget cap enforcement, prompt injection defense.",
            "Infrastructure Layer": "Local SentenceTransformers and ChromaDB vector stores, in-process socket interceptor, zero telemetry."
        }
    )


class BudgetConfig(BaseModel):
    """Configuration for runtime per-request budget cap."""
    max_tokens_per_request: int = Field(default=1500, description="Maximum estimated tokens permitted per request")
    max_cost_usd_per_request: float = Field(default=0.03, description="Maximum estimated cost in USD per request")
    cost_per_1k_tokens_usd: float = Field(default=0.002, description="Reference baseline cost per 1,000 tokens")


class GovernanceCheckResult(BaseModel):
    """Structured result of governance evaluation."""
    allowed: bool
    layer: str
    reason: str
    estimated_tokens: int
    estimated_cost_usd: float


class GovernanceError(Exception):
    """Base exception for AI governance violations."""
    pass


class PermissionDeniedError(GovernanceError):
    """Raised when an agent attempts an unauthorized tool execution (Application Layer)."""
    pass


class BudgetExceededError(GovernanceError):
    """Raised when a request exceeds the per-request token/cost budget (Runtime Layer)."""
    pass
