"""
Runtime-Layer AI Governance: Per-Request Token & Cost Budget Cap (Task 15).

Strictly complies with Capstone Project Brief (Pages 8-9):
1. Enforces a per-request token/cost budget cap.
2. Rejects deliberately oversized requests before performing expensive operations.
3. Operates 100% offline under MOCK_LLM without requiring external API counters.
"""

from typing import Optional
from nykaa_support.governance.models import (
    BudgetConfig,
    BudgetExceededError,
    GovernanceCheckResult,
)


def estimate_request_tokens(text: str) -> int:
    """
    Deterministic offline token estimator compatible with MOCK_LLM mode.
    Combines word count and character-level heuristics (1 token ~ 4 chars / 0.75 words).
    """
    if not text or not text.strip():
        return 0

    words = text.strip().split()
    word_estimate = int(len(words) * 1.33)
    char_estimate = int(len(text.strip()) / 3.8)

    return max(1, max(word_estimate, char_estimate))


def estimate_request_cost_usd(tokens: int, cost_per_1k: float = 0.002) -> float:
    """Calculate estimated cost in USD based on reference per-1k token rate."""
    return round((tokens / 1000.0) * cost_per_1k, 6)


class RuntimeBudgetGuard:
    """
    Runtime-layer guard that inspects incoming queries and rejects oversized payloads
    before invoking vector search, embedding models, or agent LLMs.
    """

    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()

    def evaluate_request(self, query: str) -> GovernanceCheckResult:
        """
        Evaluate query against budget caps without raising an exception.
        Returns GovernanceCheckResult.
        """
        tokens = estimate_request_tokens(query)
        cost = estimate_request_cost_usd(tokens, self.config.cost_per_1k_tokens_usd)

        if tokens > self.config.max_tokens_per_request:
            return GovernanceCheckResult(
                allowed=False,
                layer="Runtime Layer",
                reason=(
                    f"Token Budget Exceeded: Request estimated at {tokens} tokens exceeds the "
                    f"configured limit of {self.config.max_tokens_per_request} tokens per request."
                ),
                estimated_tokens=tokens,
                estimated_cost_usd=cost
            )

        if cost > self.config.max_cost_usd_per_request:
            return GovernanceCheckResult(
                allowed=False,
                layer="Runtime Layer",
                reason=(
                    f"Cost Budget Exceeded: Estimated cost ${cost:.5f} exceeds the "
                    f"maximum allowed budget of ${self.config.max_cost_usd_per_request:.5f} per request."
                ),
                estimated_tokens=tokens,
                estimated_cost_usd=cost
            )

        return GovernanceCheckResult(
            allowed=True,
            layer="Runtime Layer",
            reason="Request conforms to runtime token and cost budget limits.",
            estimated_tokens=tokens,
            estimated_cost_usd=cost
        )

    def enforce_request_budget(self, query: str) -> GovernanceCheckResult:
        """
        Enforce budget cap. Raises BudgetExceededError if the request is oversized.
        Must be called prior to running retrieval or model pipelines.
        """
        result = self.evaluate_request(query)
        if not result.allowed:
            raise BudgetExceededError(
                f"Runtime Governance Violation: {result.reason} "
                f"Request was blocked prior to executing retrieval or agent operations."
            )
        return result

    # Standard alias
    validate_budget = enforce_request_budget
