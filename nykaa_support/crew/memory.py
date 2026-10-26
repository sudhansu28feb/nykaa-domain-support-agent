"""
LangChain Session Memory for Nykaa Support Agent (Task 8).

Implements multi-turn conversation memory using:
- InMemoryChatMessageHistory
- RunnableWithMessageHistory

Strictly complies with the Capstone Project Brief:
1. Maintains in-process conversation history across turns within one process run.
2. Context (e.g. order ID, topic) persists across multi-turn exchanges within a session.
3. Fresh sessions start with zero prior context.
4. Deterministic offline execution.
"""

import re
from typing import Any, Dict, List, Optional
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from nykaa_support.crew.support_crew import NykaaSupportCrew
from nykaa_support.mock_llm.engine import extract_order_id, generate_grounded_response


class NykaaSessionMemoryManager:
    """
    Manages session-based conversation histories for customer support chats.
    """

    def __init__(self, crew: Optional[NykaaSupportCrew] = None):
        self.crew = crew or NykaaSupportCrew()
        self.session_store: Dict[str, InMemoryChatMessageHistory] = {}

        # Construct RunnableWithMessageHistory pipeline
        self._runnable = RunnableLambda(self._process_turn)
        self.chain = RunnableWithMessageHistory(
            self._runnable,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Fetch or create in-memory chat message history for the given session ID."""
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        return self.session_store[session_id]

    def _process_turn(self, payload: Dict[str, Any]) -> str:
        """
        Process a single turn, leveraging conversation history for resolution.
        """
        user_input = payload.get("input", "")
        history: List[BaseMessage] = payload.get("history", [])

        # Check for explicit order ID in the current query
        order_id = extract_order_id(user_input)

        # If not present in current query, check previous turns in session history
        prior_order_id = None
        if not order_id and history:
            for msg in reversed(history):
                content = str(msg.content)
                found = extract_order_id(content)
                if found:
                    prior_order_id = found
                    break

        effective_order_id = order_id or prior_order_id

        # Determine how to execute via the Crew
        if effective_order_id:
            has_policy_kw = any(kw in user_input.lower() for kw in [
                "return", "refund", "exchange", "sla", "warranty", "policy", "cancel", "damaged", "points", "size"
            ])
            if has_policy_kw:
                # Combined workflow using history-resolved order ID and policy query
                return self.crew.kickoff_combined_inquiry(effective_order_id, user_input)
            else:
                # Order inquiry workflow
                return self.crew.kickoff_order_inquiry(effective_order_id, user_input)
        else:
            # Pure policy or general inquiry
            return self.crew.kickoff_policy_inquiry(user_input)

    def invoke(self, session_id: str, query: str) -> str:
        """
        Execute a turn under the given session ID.
        Appends user query and assistant response to session history.
        """
        config = {"configurable": {"session_id": session_id}}
        result = self.chain.invoke({"input": query}, config=config)
        return str(result)

    def get_messages(self, session_id: str) -> List[BaseMessage]:
        """Return list of messages for a session."""
        return self.get_session_history(session_id).messages

    def get_transcript(self, session_id: str) -> List[Dict[str, str]]:
        """Return structured transcript of turns for a session."""
        msgs = self.get_messages(session_id)
        transcript = []
        for m in msgs:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            transcript.append({"role": role, "content": str(m.content)})
        return transcript

    def reset_session(self, session_id: str) -> None:
        """Reset conversation history for a specific session."""
        if session_id in self.session_store:
            self.session_store[session_id].clear()

    def list_sessions(self) -> List[str]:
        """List active session IDs."""
        return list(self.session_store.keys())
