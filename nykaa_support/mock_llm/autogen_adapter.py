"""
AutoGen 0.7.5 ChatCompletionClient Adapter for Deterministic Mock LLM.
Supports single agents, multi-agent review teams, and structured Pydantic output.
"""

import json
import re
from typing import Any, AsyncGenerator, Dict, Mapping, Optional, Sequence, Union
from pydantic import BaseModel

from autogen_core import CancellationToken
from autogen_core.models import (
    ChatCompletionClient,
    CreateResult,
    LLMMessage,
    ModelCapabilities,
    ModelFamily,
    ModelInfo,
    RequestUsage,
)
from autogen_core.tools import Tool, ToolSchema

from nykaa_support.mock_llm.engine import (
    MockReviewDecision,
    ReviewVerdict,
    review_response_deterministically,
)


class AutoGenMockChatClient(ChatCompletionClient):
    """Deterministic ChatCompletionClient compatible with AutoGen 0.4+ / 0.7.5."""

    def __init__(self, model_name: str = "mock-autogen-chat"):
        self.model_name = model_name
        self._usage = RequestUsage(prompt_tokens=20, completion_tokens=30)

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | str = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        # Extract original draft content and last turn content
        draft_content = ""
        for m in messages:
            c = getattr(m, "content", "")
            if not isinstance(c, str):
                c = str(c)
            # User task message contains the initial draft to be audited
            if hasattr(m, "source") and getattr(m, "source") == "user":
                draft_content = c
                break
        if not draft_content and messages:
            first_c = messages[0].content
            draft_content = first_c if isinstance(first_c, str) else str(first_c)

        # Unwrap clean_draft if wrapped with 'Draft Response to review:'
        clean_draft = draft_content
        match = re.search(r'Draft Response to review:\s*(.*?)(?:\n\nSupporting Retrieved Policy Context:|\Z)', draft_content, re.DOTALL)
        if match:
            clean_draft = match.group(1).strip()

        last_content = messages[-1].content if messages else ""
        if not isinstance(last_content, str):
            last_content = str(last_content)

        # Handle structured Pydantic output
        if json_output is not None and isinstance(json_output, type) and issubclass(json_output, BaseModel):
            # Check if requested model is ReviewVerdict, MockReviewDecision or arbitrary BaseModel
            if issubclass(json_output, (ReviewVerdict, MockReviewDecision)):
                # Evaluate the actual clean draft being reviewed by the team
                verdict = review_response_deterministically(clean_draft)
                content = verdict.model_dump_json()
            else:
                # Dynamically construct compliant instance for requested Pydantic schema
                sample_data = self._build_sample_pydantic_dict(json_output, clean_draft)
                instance = json_output.model_validate(sample_data)
                content = instance.model_dump_json()
        elif json_output is True:
            verdict = review_response_deterministically(clean_draft)
            content = verdict.model_dump_json()
        else:
            # Conversational text response for group chat
            verdict = review_response_deterministically(clean_draft)
            content = f"Review Summary: {verdict.reason} (Approved: {verdict.approved})"


        return CreateResult(
            finish_reason="stop",
            content=content,
            usage=self._usage,
            cached=False
        )


    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        tool_choice: Tool | str = "auto",
        json_output: Optional[bool | type[BaseModel]] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        res = await self.create(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            json_output=json_output,
            extra_create_args=extra_create_args,
            cancellation_token=cancellation_token,
        )
        yield str(res.content)
        yield res

    async def close(self) -> None:
        pass

    def actual_usage(self) -> RequestUsage:
        return self._usage

    def total_usage(self) -> RequestUsage:
        return self._usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 35

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 4096

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            vision=False,
            function_calling=True,
            json_output=True
        )

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family=ModelFamily.UNKNOWN
        )

    def _build_sample_pydantic_dict(self, model_cls: type[BaseModel], context_text: str) -> Dict[str, Any]:
        """Construct valid dummy dictionary satisfying the Pydantic schema."""
        data: Dict[str, Any] = {}
        fields = model_cls.model_fields
        for field_name, field_info in fields.items():
            ann = field_info.annotation
            ann_str = str(ann).lower()
            if "bool" in ann_str:
                data[field_name] = True
            elif "float" in ann_str:
                data[field_name] = 0.95
            elif "int" in ann_str:
                data[field_name] = 1
            elif "list" in ann_str:
                data[field_name] = ["Adheres to Nykaa support guidelines"]
            elif "dict" in ann_str:
                data[field_name] = {"policy_verified": True}
            else:
                data[field_name] = f"Verified: Adheres to Nykaa support policy."
        return data
