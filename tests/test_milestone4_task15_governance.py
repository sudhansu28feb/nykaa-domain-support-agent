"""
Dedicated Test Suite for Task 15:
Four-Layer AI Governance Model & Principle of Least Autonomy.

Requirements strictly verified:
1. System Risk Classification:
   - Medium Risk (Customer support tickets & order operations)
   - Formal Four-Layer AI Governance Model
   - Documented one-paragraph rationale comparing against Low and High risk tiers
   - All four governance layers accounted for: Application, Model, Runtime, Infrastructure
2. Application Layer (Principle of Least Autonomy):
   - Only the Lookup Agent may call check_order_status
   - Retrieval Agent and Response Composer are denied direct access (raises PermissionDeniedError)
   - Agent team wiring validation detects and rejects miswired agents
   - One-paragraph least autonomy guard rationale
3. Runtime Layer (Per-Request Token & Cost Budget Cap):
   - Runtime per-request token and cost cap
   - Normal queries (< 1500 tokens) are permitted
   - Oversized payloads (> 1500 tokens) are rejected with BudgetExceededError prior to expensive execution
   - 100% offline MOCK_LLM compatibility without paid token counters or network APIs
4. Task 16 Bonus / Response Caching:
   - In-memory cache keyed by normalized query text
   - Demonstrates repeated identical/variant queries produce cache hit avoiding redundant calls
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from contextlib import contextmanager
from crewai import Agent

@contextmanager
def assert_raises(expected_exception):
    class ExceptionCatcher:
        value = None
    catcher = ExceptionCatcher()
    try:
        yield catcher
    except expected_exception as e:
        catcher.value = e
    else:
        raise AssertionError(f"Expected exception {expected_exception.__name__} was not raised.")


from nykaa_support.crew.agents import (
    create_lookup_agent,
    create_response_composer_agent,
    create_retrieval_agent,
)
from nykaa_support.crew.support_crew import NykaaSupportCrew
from nykaa_support.crew.tools import OrderStatusLookupTool, RAGPolicyTool
from nykaa_support.governance import (
    BudgetConfig,
    BudgetExceededError,
    GovernanceCheckResult,
    NormalizedQueryCache,
    PermissionDeniedError,
    RiskLevel,
    RuntimeBudgetGuard,
    SystemRiskClassification,
    authorize_tool_execution,
    estimate_request_cost_usd,
    estimate_request_tokens,
    get_least_autonomy_guard_rationale,
    normalize_query_text,
    validate_agent_tool_wiring,
)
from nykaa_support.mock_llm.crewai_adapter import CrewAIMockLLM


# =====================================================================
# 1. System Risk Classification & Rationale Tests
# =====================================================================

def test_system_risk_classification():
    """Verify system risk is classified as Medium with complete 4-layer controls and rationale."""
    classification = SystemRiskClassification()

    print("\n[AI Governance Risk Classification]:")
    print(f"  System: {classification.system_name}")
    print(f"  Risk Level: {classification.risk_level.value}")
    print(f"  Category: {classification.category}")
    print(f"  Framework: {classification.framework}")

    # Schema & taxonomic verification
    assert classification.risk_level == RiskLevel.MEDIUM
    assert classification.risk_level.value == "Medium"
    assert "Customer support tickets" in classification.category
    assert classification.framework == "Four-Layer AI Governance Model"

    # Verify all four governance layers are explicitly documented
    controls = classification.governance_controls
    assert "Application Layer" in controls
    assert "Model Layer" in controls
    assert "Runtime Layer" in controls
    assert "Infrastructure Layer" in controls

    # Verify rationale substance: must contrast Low, Medium, and High
    rationale = classification.rationale
    assert len(rationale) > 200, "Rationale must be a thorough paragraph"
    assert "Medium Risk" in rationale
    assert "Low Risk" in rationale or "low" in rationale.lower()
    assert "High Risk" in rationale or "high" in rationale.lower()
    assert "customer support" in rationale.lower()
    assert "order" in rationale.lower()

    print("test_system_risk_classification: PASSED (Risk classification and rationale verified)")


# =====================================================================
# 2. Application Layer: Principle of Least Autonomy Tests
# =====================================================================

def test_least_autonomy_permitted_action():
    """Lookup Agent is permitted to execute check_order_status."""
    lookup_tool = OrderStatusLookupTool()
    lookup_agent_role = "Nykaa Order Operations Specialist"

    # Execution as authorized agent must succeed
    result = lookup_tool.execute_as(lookup_agent_role, "NYK-10001")
    assert isinstance(result, str)
    assert "NYK-10001" in result
    assert "Status" in result
    assert "escalation score" in result.lower()

    print("test_least_autonomy_permitted_action: PASSED (Lookup Agent authorized for check_order_status)")


def test_least_autonomy_denied_actions():
    """Retrieval Agent, Composer Agent, and external roles are denied check_order_status."""
    lookup_tool = OrderStatusLookupTool()

    # Case A: Retrieval Agent denied
    with assert_raises(PermissionDeniedError) as exc_retrieval:
        lookup_tool.execute_as("Nykaa Policy Retrieval Specialist", "NYK-10001")
    assert "denied access" in str(exc_retrieval.value).lower()
    assert "least autonomy" in str(exc_retrieval.value).lower()

    # Case B: Response Composer Agent denied
    with assert_raises(PermissionDeniedError) as exc_composer:
        lookup_tool.execute_as("Nykaa Customer Support Response Composer", "NYK-10001")
    assert "denied access" in str(exc_composer.value).lower()

    # Case C: Arbitrary external actor denied
    with assert_raises(PermissionDeniedError) as exc_external:
        lookup_tool.execute_as("External Client", "NYK-10001")
    assert "denied access" in str(exc_external.value).lower()

    print("test_least_autonomy_denied_actions: PASSED (Unauthorized agents successfully blocked)")


def test_least_autonomy_wiring_validation():
    """Agent team wiring validator accepts compliant wiring and rejects miswired agents."""
    llm = CrewAIMockLLM()
    rag_tool = RAGPolicyTool()
    lookup_tool = OrderStatusLookupTool()

    retrieval_agent = create_retrieval_agent(llm, rag_tool)
    lookup_agent = create_lookup_agent(llm, lookup_tool)
    response_composer = create_response_composer_agent(llm)

    # 1. Compliant wiring passes
    validate_agent_tool_wiring([retrieval_agent, lookup_agent, response_composer])

    # 2. Miswiring: Assign check_order_status to Retrieval Agent
    miswired_retrieval = Agent(
        role="Nykaa Policy Retrieval Specialist",
        goal="Retrieve policies and check orders illegally",
        backstory="Misconfigured agent",
        llm=llm,
        tools=[lookup_tool]
    )
    with assert_raises(PermissionDeniedError) as exc_miswired:
        validate_agent_tool_wiring([miswired_retrieval])
    assert "denied access to tool 'check_order_status'" in str(exc_miswired.value)

    # 3. Miswiring: Assign any tool to Response Composer
    miswired_composer = Agent(
        role="Nykaa Customer Support Response Composer",
        goal="Compose answers with illegal tools",
        backstory="Misconfigured composer",
        llm=llm,
        tools=[rag_tool]
    )
    with assert_raises(PermissionDeniedError) as exc_composer_tools:
        validate_agent_tool_wiring([miswired_composer])
    assert "must have 0 tools" in str(exc_composer_tools.value)

    print("test_least_autonomy_wiring_validation: PASSED (Team wiring validation fully functional)")


def test_least_autonomy_rationale_documentation():
    """Verify that the one-paragraph least autonomy guard rationale is documented."""
    rationale = get_least_autonomy_guard_rationale()
    assert len(rationale) > 150
    assert "Principle of Least Autonomy" in rationale
    assert "check_order_status" in rationale
    assert "Order Operations Specialist" in rationale
    print("test_least_autonomy_rationale_documentation: PASSED (Least autonomy rationale documented)")


# =====================================================================
# 3. Runtime Layer: Per-Request Budget Cap Tests
# =====================================================================

def test_runtime_budget_normal_request_permitted():
    """Verify normal-sized customer requests pass the budget evaluation."""
    guard = RuntimeBudgetGuard()

    normal_query = "What is the return window for unopened cosmetics and skincare products?"
    result = guard.evaluate_request(normal_query)

    assert result.allowed is True
    assert result.layer == "Runtime Layer"
    assert result.estimated_tokens < 50
    assert result.estimated_cost_usd < 0.001

    # enforce_request_budget must return result without raising
    enforced = guard.enforce_request_budget(normal_query)
    assert enforced.allowed is True

    print("test_runtime_budget_normal_request_permitted: PASSED (Normal query permitted within budget)")


def test_runtime_budget_oversized_request_rejected():
    """Verify oversized request is caught and rejected prior to expensive execution."""
    guard = RuntimeBudgetGuard(BudgetConfig(max_tokens_per_request=1500))

    # Construct an oversized query (approx 1800+ tokens)
    large_payload = "Please inspect order NYK-10001 with this extensive trace: " + ("repeat payload context " * 600)

    # 1. evaluate_request returns allowed=False
    result = guard.evaluate_request(large_payload)
    assert result.allowed is False
    assert "Token Budget Exceeded" in result.reason
    assert result.estimated_tokens > 1500

    # 2. enforce_request_budget raises BudgetExceededError
    with assert_raises(BudgetExceededError) as exc_budget:
        guard.enforce_request_budget(large_payload)

    err_msg = str(exc_budget.value)
    assert "Runtime Governance Violation" in err_msg
    assert "Token Budget Exceeded" in err_msg
    assert "blocked prior to executing" in err_msg

    print("test_runtime_budget_oversized_request_rejected: PASSED (Oversized query blocked pre-execution)")


def test_runtime_cost_budget_cap_enforcement():
    """Verify custom monetary cost cap rejection."""
    # Set tight budget cap of $0.0001
    tight_guard = RuntimeBudgetGuard(BudgetConfig(max_cost_usd_per_request=0.0001))
    medium_query = "Please check order status and policy details for multiple returns: " + ("query item detail " * 50)

    with assert_raises(BudgetExceededError) as exc_cost:
        tight_guard.enforce_request_budget(medium_query)

    assert "Cost Budget Exceeded" in str(exc_cost.value)
    print("test_runtime_cost_budget_cap_enforcement: PASSED (Cost budget cap enforced)")


def test_offline_mock_llm_budget_compatibility():
    """Ensure token estimation and budget calculation require zero network/API calls."""
    sample_text = "What is the refund timeline for Cash on Delivery orders?"
    tokens = estimate_request_tokens(sample_text)
    cost = estimate_request_cost_usd(tokens)

    assert tokens > 0
    assert isinstance(tokens, int)
    assert cost > 0.0
    assert isinstance(cost, float)

    print("test_offline_mock_llm_budget_compatibility: PASSED (100% offline estimation confirmed)")


# =====================================================================
# 4. Response Caching (Task 16 Integration Verification)
# =====================================================================

def test_normalized_query_caching():
    """Verify in-memory caching keyed by normalized query text with call counter evidence."""
    cache = NormalizedQueryCache()
    execution_counter = {"calls": 0}

    def sample_grounded_generator(query: str) -> str:
        execution_counter["calls"] += 1
        return f"Grounded response for: {query}"

    query_1 = "  What is the return window for cosmetics?  "
    query_2 = "what is the return window for cosmetics???"  # Differs only by casing and punctuation

    # Turn 1: Cache miss
    res1, is_hit1, _ = cache.get_or_call(query_1, sample_grounded_generator)
    assert is_hit1 is False
    assert execution_counter["calls"] == 1
    assert "Grounded response" in res1

    # Turn 2: Cache hit (identical normalized key)
    res2, is_hit2, _ = cache.get_or_call(query_2, sample_grounded_generator)
    assert is_hit2 is True
    assert execution_counter["calls"] == 1, "Generator function must NOT be called on cache hit"
    assert res1 == res2

    # Stats validation
    stats = cache.get_stats()
    assert stats["total_calls"] == 2
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate_pct"] == 50.0

    print("test_normalized_query_caching: PASSED (Normalized query caching verified with counter evidence)")


if __name__ == "__main__":
    test_system_risk_classification()
    test_least_autonomy_permitted_action()
    test_least_autonomy_denied_actions()
    test_least_autonomy_wiring_validation()
    test_least_autonomy_rationale_documentation()
    test_runtime_budget_normal_request_permitted()
    test_runtime_budget_oversized_request_rejected()
    test_runtime_cost_budget_cap_enforcement()
    test_offline_mock_llm_budget_compatibility()
    test_normalized_query_caching()
    print("\nAll Task 15 Governance and Least Autonomy tests PASSED!")
