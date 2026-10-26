"""
Standalone Grounded Generation and LangChain BaseChatModel Adapter for Mock LLM.
"""

from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from nykaa_support.mock_llm.engine import generate_grounded_response


class GroundedGenerator:
    """Direct deterministic grounded response generator for Nykaa support."""

    def __init__(self, brand_name: str = "Nykaa"):
        self.brand_name = brand_name

    def generate(self, query: str, context: Optional[str] = None) -> str:
        """Generate response deterministically from query and context."""
        return generate_grounded_response(query, context)


class MockLangChainChatModel(BaseChatModel):
    """Deterministic LangChain BaseChatModel for offline and test workflows."""

    model_name: str = Field(default="mock-nykaa-chat-model")

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_query = ""
        context = ""

        # Extract user query and optional system/context messages
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if msg.type in ("system", "tool", "function"):
                context += f"\n{content}"
            elif msg.type == "user":
                user_query = content

        if not user_query and messages:
            last = messages[-1]
            user_query = last.content if isinstance(last.content, str) else str(last.content)

        answer = generate_grounded_response(query=user_query, context=context.strip() if context else None)
        message = AIMessage(content=answer)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "mock-nykaa-langchain"
