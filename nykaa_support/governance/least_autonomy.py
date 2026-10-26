"""
Application-Layer AI Governance: Principle of Least Autonomy (Task 15).

Enforces that:
1. ONLY the Lookup Agent may call check_order_status.
2. The Retrieval Agent is restricted to rag_lookup.
3. The Response Composer is assigned zero tools (synthesis only).
4. Unauthorized tool invocations or wiring attempts raise PermissionDeniedError.
"""

from typing import Any, List, Optional
from crewai import Agent

from nykaa_support.governance.models import PermissionDeniedError


# Authorized mappings: tool_name -> authorized agent role substring
AUTHORIZED_TOOL_ROLES = {
    "check_order_status": "Order Operations Specialist",
    "rag_lookup": "Policy Retrieval Specialist",
}


def authorize_tool_execution(agent_role: str, tool_name: str) -> None:
    """
    Application-layer permission guard.
    Raises PermissionDeniedError if the calling agent is unauthorized for the tool.
    """
    clean_role = agent_role.strip()
    clean_tool = tool_name.strip()

    if clean_tool in AUTHORIZED_TOOL_ROLES:
        required_role_substr = AUTHORIZED_TOOL_ROLES[clean_tool]
        if required_role_substr.lower() not in clean_role.lower():
            raise PermissionDeniedError(
                f"Application Governance Guard: Agent '{clean_role}' is denied access to tool '{clean_tool}'. "
                f"By the Principle of Least Autonomy, tool '{clean_tool}' is restricted exclusively to the "
                f"'{required_role_substr}'."
            )


def validate_agent_tool_wiring(agents: List[Agent]) -> None:
    """
    Validates that a CrewAI agent team adheres to the Principle of Least Autonomy.
    Raises PermissionDeniedError if any agent is miswired.
    """
    for agent in agents:
        role = getattr(agent, "role", "")
        tools = getattr(agent, "tools", []) or []

        # Enforce that Response Composer never possesses any tools (Least Autonomy)
        if "Response Composer" in role and len(tools) > 0:
            raise PermissionDeniedError(
                f"Application Governance Guard: Response Composer Agent must have 0 tools (Least Autonomy), "
                f"but was wired with {len(tools)} tool(s)."
            )

        for t in tools:
            tool_name = getattr(t, "name", str(t))
            # Verify authorization
            authorize_tool_execution(role, tool_name)


def get_least_autonomy_guard_rationale() -> str:
    """
    One-paragraph rationale of the least-autonomy guard as required by the brief.
    """
    return (
        "Under the Application layer of the Four-Layer AI Governance framework, the Principle of Least Autonomy "
        "restricts each agent's executable surface strictly to its designated operational domain. "
        "The 'check_order_status' tool connects to live Nykaa order logistics and customer database records. "
        "Allowing the Retrieval Agent or Response Composer Agent direct access to order records would create "
        "privilege creep, expand the attack surface for prompt-injection manipulation, and introduce unauthorized "
        "state lookups during generic policy inquiries. Enforcing that only the Order Operations Specialist "
        "(Lookup Agent) possesses and executes 'check_order_status' prevents accidental data leakage, eliminates "
        "unintended tool invocation during policy queries, and establishes a clear audit trail for operational lookups."
    )


# Standard alias
enforce_least_autonomy = authorize_tool_execution
