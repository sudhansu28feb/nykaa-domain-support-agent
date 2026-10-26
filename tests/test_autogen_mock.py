"""
Tests for AutoGen 0.7.5 Review Team with Structured Pydantic Output (Task 14).

Verifies:
1. Structured output via ReviewVerdict (approved: bool, final_answer: str, reason: str).
2. RoundRobinGroupChat team constructed with custom_message_types=[StructuredMessage[ReviewVerdict]].
3. Bounded with max_turns=2 (or MaxMessageTermination(3)).
4. Demonstration of at least one case approving unchanged and at least one revising ungrounded claims.
"""

import asyncio
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

from nykaa_support.mock_llm.autogen_adapter import AutoGenMockChatClient
from nykaa_support.mock_llm.engine import ReviewVerdict


async def test_autogen_agent_structured_output():
    """Verify single agent structured output matching Task 14 ReviewVerdict schema."""
    client = AutoGenMockChatClient()
    agent = AssistantAgent(
        name="FinalEditor",
        model_client=client,
        system_message="You are a strict compliance editor reviewing customer support replies for Nykaa.",
        output_content_type=ReviewVerdict
    )

    draft = (
        "Hello! Thank you for contacting Nykaa Support. You can return unused makeup items within 15 days "
        "of delivery as long as all seals and tags are intact. Please let us know if you need help initiating the return!"
    )

    run_result = await agent.run(task=draft)
    final_message = run_result.messages[-1]
    content = final_message.content

    print("\n[AutoGen Structured Message Content]:\n", repr(content))

    assert isinstance(content, ReviewVerdict), f"Expected ReviewVerdict, got {type(content)}"
    assert content.approved is True
    assert "15 days" in content.final_answer
    assert len(content.reason) > 5
    print("test_autogen_agent_structured_output PASSED (ReviewVerdict verified)")


async def test_autogen_multiagent_review_team_approval_case():
    """
    Task 14 Case 1: Team approves a factually grounded draft unchanged.
    Uses custom_message_types=[StructuredMessage[ReviewVerdict]] and max_turns=2.
    """
    client = AutoGenMockChatClient()

    reviewer = AssistantAgent(
        name="PolicyComplianceReviewer",
        model_client=client,
        system_message="You are a policy compliance reviewer checking Nykaa support drafts."
    )

    final_editor = AssistantAgent(
        name="FinalEditor",
        model_client=client,
        system_message="You are the final editor producing a structured ReviewVerdict.",
        output_content_type=ReviewVerdict
    )

    # Use max_turns=2 as documented in the capstone brief
    team = RoundRobinGroupChat(
        participants=[reviewer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]]
    )

    valid_draft = (
        "Hello! Thank you for reaching out to Nykaa Support. Your return request for the unopened cosmetics "
        "has been authorized within our 15-day policy window. Our courier partner will complete reverse pickup "
        "within 24 to 48 hours."
    )

    print("\n[AutoGen Team Case 1 - Approved Unchanged]:")
    messages = []
    async for msg in team.run_stream(task=valid_draft):
        if hasattr(msg, "content"):
            messages.append(msg)
            print(f"  [{getattr(msg, 'source', 'Team')}]: {str(msg.content)[:120]}...")

    assert len(messages) >= 2, f"Expected at least 2 messages in team run, got {len(messages)}"
    final_content = messages[-1].content
    assert isinstance(final_content, ReviewVerdict)
    assert final_content.approved is True
    assert valid_draft.strip() == final_content.final_answer.strip()
    print("test_autogen_multiagent_review_team_approval_case PASSED")


async def test_autogen_multiagent_review_team_revision_case():
    """
    Task 14 Case 2: Team catches ungrounded claim injected into draft and revises it.
    """
    client = AutoGenMockChatClient()

    reviewer = AssistantAgent(
        name="PolicyComplianceReviewer",
        model_client=client,
        system_message="You are a policy compliance reviewer checking Nykaa support drafts."
    )

    final_editor = AssistantAgent(
        name="FinalEditor",
        model_client=client,
        system_message="You are the final editor producing a structured ReviewVerdict.",
        output_content_type=ReviewVerdict
    )

    team = RoundRobinGroupChat(
        participants=[reviewer, final_editor],
        max_turns=2,
        custom_message_types=[StructuredMessage[ReviewVerdict]]
    )

    # Deliberately injected ungrounded claim (90-day window and cash payout)
    hallucinated_draft = (
        "You can return it within 90 days for a full cash payout delivered to your doorstep tomorrow."
    )

    print("\n[AutoGen Team Case 2 - Revised Ungrounded Claim]:")
    messages = []
    async for msg in team.run_stream(task=hallucinated_draft):
        if hasattr(msg, "content"):
            messages.append(msg)
            print(f"  [{getattr(msg, 'source', 'Team')}]: {str(msg.content)[:120]}...")

    assert len(messages) >= 2, f"Expected at least 2 messages in team run, got {len(messages)}"
    final_content = messages[-1].content
    assert isinstance(final_content, ReviewVerdict)
    assert final_content.approved is False
    assert "15 calendar days" in final_content.final_answer
    assert "ungrounded" in final_content.reason.lower() or "revised" in final_content.reason.lower()
    print("test_autogen_multiagent_review_team_revision_case PASSED")


async def main():
    await test_autogen_agent_structured_output()
    await test_autogen_multiagent_review_team_approval_case()
    await test_autogen_multiagent_review_team_revision_case()
    print("\nAll AutoGen Task 14 structured review tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
