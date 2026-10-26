"""
CrewAI Agents for Nykaa Support Workflow (Task 7).

Strictly adheres to the Principle of Least Autonomy / Least Privilege:
1. Retrieval Agent: Equipped ONLY with RAGPolicyTool (rag_lookup).
2. Lookup Agent: Equipped ONLY with OrderStatusLookupTool (check_order_status).
3. Response Composer Agent: Equipped with NO tools (combines findings into final draft).
"""

from typing import Optional
from crewai import Agent

from nykaa_support.crew.tools import OrderStatusLookupTool, RAGPolicyTool
from nykaa_support.mock_llm.crewai_adapter import CrewAIMockLLM


def create_retrieval_agent(
    llm: Optional[CrewAIMockLLM] = None,
    rag_tool: Optional[RAGPolicyTool] = None
) -> Agent:
    """Creates the Retrieval Agent equipped only with the RAG policy retrieval tool."""
    mock_llm = llm or CrewAIMockLLM()
    tool = rag_tool or RAGPolicyTool()

    return Agent(
        role="Nykaa Policy Retrieval Specialist",
        goal="Retrieve authoritative policy documentation from the Nykaa knowledge base for customer inquiries.",
        backstory=(
            "Expert in Nykaa company policies including returns, refunds, delivery SLAs, "
            "damaged item claims, warranties, and loyalty points."
        ),
        llm=mock_llm,
        tools=[tool],
        verbose=False,
        allow_delegation=False
    )


def create_lookup_agent(
    llm: Optional[CrewAIMockLLM] = None,
    lookup_tool: Optional[OrderStatusLookupTool] = None
) -> Agent:
    """Creates the Lookup Agent equipped only with the order status lookup tool."""
    mock_llm = llm or CrewAIMockLLM()
    tool = lookup_tool or OrderStatusLookupTool()

    return Agent(
        role="Nykaa Order Operations Specialist",
        goal="Lookup order fulfillment records, delivery delay status, and calculate escalation scores.",
        backstory=(
            "Operations analyst who checks order records in Nykaa fulfillment database and "
            "evaluates delay risks and human escalation requirements."
        ),
        llm=mock_llm,
        tools=[tool],
        verbose=False,
        allow_delegation=False
    )


def create_response_composer_agent(
    llm: Optional[CrewAIMockLLM] = None
) -> Agent:
    """Creates the Response Composer Agent with zero tools (least autonomy)."""
    mock_llm = llm or CrewAIMockLLM()

    return Agent(
        role="Nykaa Customer Support Response Composer",
        goal="Synthesize retrieved policy details and order fulfillment findings into a complete, empathetic response.",
        backstory=(
            "Lead customer support specialist at Nykaa dedicated to clear, polite, and "
            "thorough customer communications."
        ),
        llm=mock_llm,
        tools=[],  # Strictly NO tools: least privilege
        verbose=False,
        allow_delegation=False
    )
