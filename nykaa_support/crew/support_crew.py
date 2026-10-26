"""
CrewAI Multi-Agent Workflow for Nykaa Domain Support (Task 7).

Orchestrates a 3-agent crew:
1. Retrieval Agent (with rag_lookup tool)
2. Lookup Agent (with check_order_status tool)
3. Response Composer (zero tools, synthesizes outputs)

Features:
- Executes strictly via Crew.kickoff()
- Enforces Least Autonomy / Least Privilege
- Telemetry disabled
"""

import os
import re
from typing import Any, Dict, List, Optional
from crewai import Agent, Crew, Process, Task

from nykaa_support.crew.agents import (
    create_lookup_agent,
    create_response_composer_agent,
    create_retrieval_agent,
)
from nykaa_support.crew.tools import OrderStatusLookupTool, RAGPolicyTool
from nykaa_support.mock_llm.crewai_adapter import CrewAIMockLLM
from nykaa_support.mock_llm.engine import extract_order_id


class NykaaSupportCrew:
    """
    CrewAI Orchestrator for Nykaa Customer Support multi-agent system.
    """

    def __init__(
        self,
        llm: Optional[CrewAIMockLLM] = None,
        rag_tool: Optional[RAGPolicyTool] = None,
        lookup_tool: Optional[OrderStatusLookupTool] = None
    ):
        # Enforce telemetry opt-out
        os.environ["OTEL_SDK_DISABLED"] = "true"
        os.environ["CREWAI_TESTING"] = "true"

        self.llm = llm or CrewAIMockLLM()
        self.rag_tool = rag_tool or RAGPolicyTool()
        self.lookup_tool = lookup_tool or OrderStatusLookupTool()

        # Instantiate 3 specialized agents
        self.retrieval_agent = create_retrieval_agent(self.llm, self.rag_tool)
        self.lookup_agent = create_lookup_agent(self.llm, self.lookup_tool)
        self.response_composer = create_response_composer_agent(self.llm)

        # Enforce and verify least autonomy on instantiation
        self.verify_least_autonomy()

    def verify_least_autonomy(self) -> None:
        """
        Verify that agent permissions are strictly least-privilege:
        - Retrieval Agent has ONLY rag_lookup
        - Lookup Agent has ONLY check_order_status
        - Response Composer has NO tools
        """
        assert len(self.retrieval_agent.tools) == 1, "Retrieval Agent must have exactly 1 tool"
        assert self.retrieval_agent.tools[0].name == "rag_lookup", "Retrieval Agent must only have rag_lookup"

        assert len(self.lookup_agent.tools) == 1, "Lookup Agent must have exactly 1 tool"
        assert self.lookup_agent.tools[0].name == "check_order_status", "Lookup Agent must only have check_order_status"

        assert len(self.response_composer.tools) == 0, "Response Composer Agent must have 0 tools (least privilege)"

        # Enforce governance layer wiring validation
        from nykaa_support.governance.least_autonomy import validate_agent_tool_wiring
        validate_agent_tool_wiring([self.retrieval_agent, self.lookup_agent, self.response_composer])

    def kickoff_policy_inquiry(self, query: str) -> str:
        """
        Execute CrewAI workflow for a policy inquiry:
        - Retrieval Agent retrieves grounded policy via rag_lookup
        - Response Composer synthesizes a customer-facing response
        """
        task_retrieve = Task(
            description=f"Retrieve official Nykaa policy context for customer question: '{query}'.",
            expected_output="Grounded policy clauses and context relevant to the customer inquiry.",
            agent=self.retrieval_agent
        )

        task_compose = Task(
            description=(
                f"Synthesize the retrieved policy documentation into a clear, empathetic, and complete "
                f"customer response addressing: '{query}'."
            ),
            expected_output="Complete customer support response with policy guidance.",
            agent=self.response_composer,
            context=[task_retrieve]
        )

        crew = Crew(
            agents=[self.retrieval_agent, self.response_composer],
            tasks=[task_retrieve, task_compose],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    def kickoff_order_inquiry(self, order_id: str, query: Optional[str] = None) -> str:
        """
        Execute CrewAI workflow for an order fulfillment inquiry:
        - Lookup Agent looks up status, delays, and escalation score via check_order_status
        - Response Composer synthesizes a customer-facing status update
        """
        inquiry_text = query or f"What is the status of my order {order_id}?"

        task_lookup = Task(
            description=f"Check the live order fulfillment records and escalation score for order {order_id}.",
            expected_output="Live status, order value, delay status, and calculated escalation score.",
            agent=self.lookup_agent
        )

        task_compose = Task(
            description=(
                f"Synthesize the order lookup findings into an empathetic and helpful customer update "
                f"answering: '{inquiry_text}'."
            ),
            expected_output="Polite and thorough order status response.",
            agent=self.response_composer,
            context=[task_lookup]
        )

        crew = Crew(
            agents=[self.lookup_agent, self.response_composer],
            tasks=[task_lookup, task_compose],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    def kickoff_combined_inquiry(self, order_id: str, policy_query: str) -> str:
        """
        Execute full 3-agent CrewAI workflow:
        - Retrieval Agent checks policy (e.g. return window or refund timeline)
        - Lookup Agent checks order status and escalation score
        - Response Composer synthesizes unified answer
        """
        task_retrieve = Task(
            description=f"Retrieve official Nykaa policy for: '{policy_query}'.",
            expected_output="Relevant policy rules and requirements.",
            agent=self.retrieval_agent
        )

        task_lookup = Task(
            description=f"Lookup order status and escalation score for order {order_id}.",
            expected_output="Fulfillment status and escalation details.",
            agent=self.lookup_agent
        )

        task_compose = Task(
            description=(
                f"Synthesize both the retrieved policy context and the live order status for order {order_id} "
                f"to answer: '{policy_query}'."
            ),
            expected_output="Comprehensive response covering order status and policy requirements.",
            agent=self.response_composer,
            context=[task_retrieve, task_lookup]
        )

        crew = Crew(
            agents=[self.retrieval_agent, self.lookup_agent, self.response_composer],
            tasks=[task_retrieve, task_lookup, task_compose],
            process=Process.sequential,
            verbose=False
        )

        result = crew.kickoff()
        return str(result)

    def kickoff_for_query(self, query: str) -> str:
        """
        Automatically routes and executes the query through the CrewAI kickoff mechanism.
        """
        order_id = extract_order_id(query)
        has_policy_keywords = any(kw in query.lower() for kw in [
            "return", "refund", "exchange", "sla", "warranty", "policy", "cancel", "damaged", "points"
        ])

        if order_id and has_policy_keywords:
            return self.kickoff_combined_inquiry(order_id, query)
        elif order_id:
            return self.kickoff_order_inquiry(order_id, query)
        else:
            return self.kickoff_policy_inquiry(query)
