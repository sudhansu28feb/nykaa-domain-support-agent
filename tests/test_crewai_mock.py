"""
Tests for CrewAI Agent with Mock LLM and Tool Calling.
Verifies telemetry opt-out and deterministic ReAct tool execution.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure UTF-8 for console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure CrewAI telemetry is disabled
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TESTING"] = "true"

from crewai import Agent, Task, Crew
from crewai.tools import tool
from nykaa_support.mock_llm.crewai_adapter import CrewAIMockLLM

tool_execution_log = []


@tool
def lookup_order_status(order_id: str) -> str:
    """Lookup order status and shipping tracking in the Nykaa fulfillment system."""
    tool_execution_log.append(order_id)
    return f"Order {order_id} is OUT FOR DELIVERY via BlueDart (Tracking #BD889922). Expected delivery: Today by 6 PM."


def test_crewai_agent_with_tool():
    tool_execution_log.clear()

    # Verify telemetry environment variable
    assert os.getenv("OTEL_SDK_DISABLED") == "true", "OTEL_SDK_DISABLED must be set to true"

    llm = CrewAIMockLLM()
    agent = Agent(
        role="Nykaa Logistics Specialist",
        goal="Track customer packages and provide live dispatch updates",
        backstory="Dedicated order support agent for Nykaa e-commerce platform",
        llm=llm,
        tools=[lookup_order_status],
        verbose=False
    )

    task = Task(
        description="Customer wants to know the live status of order NYK-998822.",
        expected_output="Detailed shipping and delivery status for order NYK-998822.",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()
    result_str = str(result)
    print("\n[CrewAI Execution Result]:\n", result_str)

    # Verifications
    assert len(tool_execution_log) == 1, f"Expected 1 tool execution, got {len(tool_execution_log)}"
    assert "NYK-998822" in tool_execution_log[0], f"Expected order ID in tool log, got {tool_execution_log}"
    assert "OUT FOR DELIVERY" in result_str or "BlueDart" in result_str
    print("test_crewai_agent_with_tool PASSED")


def test_crewai_agent_direct_response():
    llm = CrewAIMockLLM()
    agent = Agent(
        role="Nykaa Policy Guide",
        goal="Answer general inquiries about Nykaa store policies",
        backstory="Knowledgeable customer support specialist",
        llm=llm,
        tools=[],  # No tools provided -> direct response
        verbose=False
    )

    task = Task(
        description="What is the return policy for cosmetics?",
        expected_output="Explanation of return policy",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()
    result_str = str(result)
    print("\n[CrewAI Direct Response]:\n", result_str)

    assert "15 days" in result_str or "Nykaa" in result_str
    print("test_crewai_agent_direct_response PASSED")


if __name__ == "__main__":
    test_crewai_agent_with_tool()
    test_crewai_agent_direct_response()
    print("\nAll CrewAI mock tests completed successfully!")
