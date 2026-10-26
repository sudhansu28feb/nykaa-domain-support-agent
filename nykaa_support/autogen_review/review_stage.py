"""
AutoGen Review Stage for Nykaa Support Workflow (Task 14).

Implements a 2-agent Round Robin Group Chat team:
1. PolicyComplianceReviewer (checks draft against official Nykaa policies)
2. FinalEditor (produces structured Pydantic ReviewVerdict)

Strictly complies with Capstone Project Brief (Pages 7-8):
- RoundRobinGroupChat bounded with max_turns=2
- FinalEditor assigned output_content_type=ReviewVerdict
- Team constructed with custom_message_types=[StructuredMessage[ReviewVerdict]]
- Takes CrewAI draft answer + retrieved context as input
- Evaluates and either approves unchanged or revises ungrounded claims
"""

import asyncio
from typing import Optional
from pydantic import BaseModel

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat

from nykaa_support.mock_llm.autogen_adapter import AutoGenMockChatClient
from nykaa_support.mock_llm.engine import ReviewVerdict


class NykaaReviewStage:
    """
    Orchestrates the AutoGen 2-agent compliance and editorial review stage.
    """

    def __init__(self, model_client: Optional[AutoGenMockChatClient] = None):
        self.client = model_client or AutoGenMockChatClient()
        # Initial team instance for inspection & test assertions
        self.team = self._build_team()
        self.reviewer = self.team._participants[0]
        self.final_editor = self.team._participants[1]

    def _build_team(self) -> RoundRobinGroupChat:
        """Construct the 2-agent RoundRobinGroupChat team bound to the calling event loop."""
        reviewer = AssistantAgent(
            name="PolicyComplianceReviewer",
            model_client=self.client,
            system_message=(
                "You are an expert policy compliance reviewer for Nykaa Customer Support. "
                "Audit draft answers against official Nykaa return windows, refund timelines, "
                "and delivery SLAs. Identify any ungrounded, exaggerated, or non-compliant claims."
            )
        )
        final_editor = AssistantAgent(
            name="FinalEditor",
            model_client=self.client,
            system_message=(
                "You are the final editor for Nykaa Customer Support. Synthesize the compliance findings "
                "and output a structured ReviewVerdict. If the draft is compliant, approve it unchanged; "
                "if ungrounded, revise it to strictly match official Nykaa policies."
            ),
            output_content_type=ReviewVerdict
        )
        return RoundRobinGroupChat(
            participants=[reviewer, final_editor],
            max_turns=2,
            custom_message_types=[StructuredMessage[ReviewVerdict]]
        )

    async def review_draft(
        self,
        draft_answer: str,
        context: Optional[str] = None
    ) -> ReviewVerdict:
        """
        Run the AutoGen review team asynchronously on the draft answer and supporting context.
        Returns the final structured ReviewVerdict.
        """
        task_prompt = f"Draft Response to review:\n{draft_answer.strip()}"
        if context and context.strip():
            task_prompt += f"\n\nSupporting Retrieved Policy Context:\n{context.strip()}"

        active_team = self._build_team()
        messages = []
        async for msg in active_team.run_stream(task=task_prompt):
            if hasattr(msg, "content"):
                messages.append(msg)

        if not messages:
            raise RuntimeError("AutoGen review team produced zero messages.")

        final_msg = messages[-1]
        content = getattr(final_msg, "content", None)

        if isinstance(content, ReviewVerdict):
            return content
        elif isinstance(content, str):
            # Attempt parsing JSON into ReviewVerdict
            return ReviewVerdict.model_validate_json(content)
        elif isinstance(content, dict):
            return ReviewVerdict.model_validate(content)
        else:
            raise ValueError(f"Unexpected AutoGen verdict message content type: {type(content)}")

    def review_draft_sync(
        self,
        draft_answer: str,
        context: Optional[str] = None
    ) -> ReviewVerdict:
        """Synchronous wrapper for review_draft."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an active event loop, run in separate thread or create task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.review_draft(draft_answer, context))
                return future.result()
        else:
            return asyncio.run(self.review_draft(draft_answer, context))
