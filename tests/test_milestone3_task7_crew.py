"""
Dedicated Test Suite for Task 7:
CrewAI Multi-Agent Workflow, Tool Schema Dispatch, and Least Autonomy.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TESTING"] = "true"

from nykaa_support.crew import (
    NykaaSupportCrew,
    OrderStatusLookupTool,
    RAGPolicyTool,
    create_lookup_agent,
    create_response_composer_agent,
    create_retrieval_agent,
)
from nykaa_support.mock_llm.crewai_adapter import CrewAIMockLLM


LOOKUP_EXECUTION_CALLS = []
RAG_EXECUTION_CALLS = []


class TrackingLookupTool(OrderStatusLookupTool):
    """Subclass to track actual execution calls."""

    def _run(self, record_id: str) -> str:
        LOOKUP_EXECUTION_CALLS.append(record_id)
        return super()._run(record_id)


class TrackingRAGTool(RAGPolicyTool):
    """Subclass to track actual execution calls."""

    def _run(self, query: str) -> str:
        RAG_EXECUTION_CALLS.append(query)
        return super()._run(query)


def test_least_autonomy_enforcement():
    """Verify strict least-privilege tool allocation across all 3 agents."""
    llm = CrewAIMockLLM()
    rag_tool = TrackingRAGTool()
    lookup_tool = TrackingLookupTool()

    crew_obj = NykaaSupportCrew(llm=llm, rag_tool=rag_tool, lookup_tool=lookup_tool)

    # 1. Retrieval Agent must have ONLY rag_lookup
    assert len(crew_obj.retrieval_agent.tools) == 1
    assert crew_obj.retrieval_agent.tools[0].name == "rag_lookup"
    assert crew_obj.retrieval_agent.tools[0].args_schema.__name__ == "RAGPolicyInput"
    assert "query" in crew_obj.retrieval_agent.tools[0].args_schema.model_fields

    # 2. Lookup Agent must have ONLY check_order_status
    assert len(crew_obj.lookup_agent.tools) == 1
    assert crew_obj.lookup_agent.tools[0].name == "check_order_status"
    assert crew_obj.lookup_agent.tools[0].args_schema.__name__ == "OrderStatusInput"
    assert "record_id" in crew_obj.lookup_agent.tools[0].args_schema.model_fields

    # 3. Response Composer Agent must have NO tools (least autonomy)
    assert len(crew_obj.response_composer.tools) == 0

    print("test_least_autonomy_enforcement: PASSED (Strict least privilege verified)")


def test_crew_invokes_rag_tool_on_policy_query():
    """Demonstrate RAG tool actually invoked on a policy inquiry via kickoff."""
    RAG_EXECUTION_CALLS.clear()
    LOOKUP_EXECUTION_CALLS.clear()

    llm = CrewAIMockLLM()
    rag_tool = TrackingRAGTool()
    lookup_tool = TrackingLookupTool()
    crew_obj = NykaaSupportCrew(llm=llm, rag_tool=rag_tool, lookup_tool=lookup_tool)

    query = "What is the return window for electronics and cosmetics?"
    result = crew_obj.kickoff_policy_inquiry(query)
    result_str = str(result)

    print("\n[Policy Inquiry Kickoff Result]:\n", result_str)

    # Verification: RAG tool was executed, Lookup tool was NOT executed
    assert len(RAG_EXECUTION_CALLS) >= 1, "RAG tool must be executed"
    assert len(LOOKUP_EXECUTION_CALLS) == 0, "Lookup tool must NOT be executed on policy query"
    assert "return" in RAG_EXECUTION_CALLS[0].lower() or "electronics" in RAG_EXECUTION_CALLS[0].lower()
    assert "Nykaa" in result_str or "Policy" in result_str or "return" in result_str

    print("test_crew_invokes_rag_tool_on_policy_query: PASSED (RAG tool invoked)")


def test_crew_invokes_lookup_tool_on_order_query():
    """Demonstrate order status lookup tool actually invoked on order inquiry via kickoff."""
    RAG_EXECUTION_CALLS.clear()
    LOOKUP_EXECUTION_CALLS.clear()

    llm = CrewAIMockLLM()
    rag_tool = TrackingRAGTool()
    lookup_tool = TrackingLookupTool()
    crew_obj = NykaaSupportCrew(llm=llm, rag_tool=rag_tool, lookup_tool=lookup_tool)

    order_id = "NYK-10001"
    result = crew_obj.kickoff_order_inquiry(order_id)
    result_str = str(result)

    print("\n[Order Inquiry Kickoff Result]:\n", result_str)

    # Verification: Lookup tool was executed with exact order_id, RAG tool was NOT executed
    assert len(LOOKUP_EXECUTION_CALLS) >= 1, "Lookup tool must be executed"
    assert len(RAG_EXECUTION_CALLS) == 0, "RAG tool must NOT be executed on pure order query"
    assert LOOKUP_EXECUTION_CALLS[0] == "NYK-10001"
    assert "NYK-10001" in result_str
    assert "escalation score" in result_str.lower() or "status" in result_str.lower()

    print("test_crew_invokes_lookup_tool_on_order_query: PASSED (Lookup tool invoked)")


def test_crew_combined_workflow():
    """Demonstrate all 3 agents cooperating on a combined inquiry via kickoff."""
    RAG_EXECUTION_CALLS.clear()
    LOOKUP_EXECUTION_CALLS.clear()

    llm = CrewAIMockLLM()
    rag_tool = TrackingRAGTool()
    lookup_tool = TrackingLookupTool()
    crew_obj = NykaaSupportCrew(llm=llm, rag_tool=rag_tool, lookup_tool=lookup_tool)

    order_id = "NYK-10002"
    policy_q = f"Can I return order {order_id} and what is the return window?"
    result = crew_obj.kickoff_combined_inquiry(order_id, policy_q)
    result_str = str(result)

    print("\n[Combined Inquiry Kickoff Result]:\n", result_str)

    # Verification: Both tools were executed
    assert len(RAG_EXECUTION_CALLS) >= 1, "RAG tool must be executed"
    assert len(LOOKUP_EXECUTION_CALLS) >= 1, "Lookup tool must be executed"
    assert LOOKUP_EXECUTION_CALLS[0] == "NYK-10002"
    assert "NYK-10002" in result_str

    print("test_crew_combined_workflow: PASSED (Both tools invoked and combined)")


if __name__ == "__main__":
    test_least_autonomy_enforcement()
    test_crew_invokes_rag_tool_on_policy_query()
    test_crew_invokes_lookup_tool_on_order_query()
    test_crew_combined_workflow()
    print("\nAll Task 7 CrewAI Multi-Agent Workflow tests PASSED!")
