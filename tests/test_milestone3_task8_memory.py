"""
Dedicated Test Suite for Task 8:
LangChain Session-Based Memory (InMemoryChatMessageHistory + RunnableWithMessageHistory).
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

from langchain_core.messages import AIMessage, HumanMessage
from nykaa_support.crew.memory import NykaaSessionMemoryManager
from nykaa_support.crew.support_crew import NykaaSupportCrew


def test_multi_turn_session_context_persistence():
    """
    Demonstrates state correctly carried across a multi-turn exchange within a session:
    Turn 1: Customer introduces order NYK-10001.
    Turn 2: Customer asks follow-up policy question without repeating order ID. Memory resolves order NYK-10001.
    """
    manager = NykaaSessionMemoryManager()
    session_id = "session_user_alice_123"

    # Turn 1: Introduce order ID
    t1_query = "Hi, can you check the status of my order NYK-10001?"
    t1_reply = manager.invoke(session_id, t1_query)
    print("\n[Turn 1 Reply]:\n", t1_reply)

    assert "NYK-10001" in t1_reply
    msgs_after_t1 = manager.get_messages(session_id)
    assert len(msgs_after_t1) == 2, f"Expected 2 messages after Turn 1, got {len(msgs_after_t1)}"
    assert isinstance(msgs_after_t1[0], HumanMessage)
    assert isinstance(msgs_after_t1[1], AIMessage)
    assert "NYK-10001" in msgs_after_t1[0].content

    # Turn 2: Follow-up question relying on memory (no explicit order ID)
    t2_query = "Can I exchange this item for a different size?"
    t2_reply = manager.invoke(session_id, t2_query)
    print("\n[Turn 2 Reply]:\n", t2_reply)

    # Verification: Turn 2 leveraged prior order ID from memory
    assert "NYK-10001" in t2_reply, "Turn 2 response must resolve prior order ID from conversation memory"
    assert "exchange" in t2_reply.lower() or "policy" in t2_reply.lower()

    msgs_after_t2 = manager.get_messages(session_id)
    assert len(msgs_after_t2) == 4, f"Expected 4 messages after Turn 2, got {len(msgs_after_t2)}"

    transcript = manager.get_transcript(session_id)
    assert len(transcript) == 4
    assert transcript[0]["role"] == "user"
    assert transcript[1]["role"] == "assistant"
    assert transcript[2]["role"] == "user"
    assert transcript[3]["role"] == "assistant"

    print("test_multi_turn_session_context_persistence: PASSED (Context persisted across turns)")


def test_fresh_session_isolation():
    """
    Demonstrates a separate fresh-conversation transcript showing state correctly absent/reset:
    Session 1 has order NYK-10001.
    Session 2 starts fresh with no prior context.
    """
    manager = NykaaSessionMemoryManager()
    session_1 = "session_customer_1"
    session_2 = "session_customer_2"

    # Session 1: Establish order context
    manager.invoke(session_1, "Check order NYK-10001 status")
    assert len(manager.get_messages(session_1)) == 2

    # Session 2: Fresh session asks follow-up query without order ID
    t1_s2_query = "Can I exchange this item for a different size?"
    t1_s2_reply = manager.invoke(session_2, t1_s2_query)
    print("\n[Session 2 Turn 1 Reply (Fresh Session)]:\n", t1_s2_reply)

    # Verification: Fresh session has NO awareness of NYK-10001
    assert "NYK-10001" not in t1_s2_reply, "Fresh session must NOT have access to Session 1 order ID"
    assert "exchange" in t1_s2_reply.lower() or "policy" in t1_s2_reply.lower()

    s2_msgs = manager.get_messages(session_2)
    assert len(s2_msgs) == 2, f"Expected 2 messages in Session 2, got {len(s2_msgs)}"

    # Session 1 remains intact
    s1_msgs = manager.get_messages(session_1)
    assert len(s1_msgs) == 2
    assert "NYK-10001" in s1_msgs[0].content

    print("test_fresh_session_isolation: PASSED (Fresh session correctly isolated)")


def test_session_reset():
    """Verify reset_session properly clears session state."""
    manager = NykaaSessionMemoryManager()
    session_id = "session_to_clear"

    manager.invoke(session_id, "Check order NYK-10001 status")
    assert len(manager.get_messages(session_id)) == 2

    manager.reset_session(session_id)
    assert len(manager.get_messages(session_id)) == 0

    # Next call after reset behaves like a fresh session
    reply = manager.invoke(session_id, "Can I exchange for a different size?")
    assert "NYK-10001" not in reply
    assert len(manager.get_messages(session_id)) == 2

    print("test_session_reset: PASSED (Session reset successfully)")


if __name__ == "__main__":
    test_multi_turn_session_context_persistence()
    test_fresh_session_isolation()
    test_session_reset()
    print("\nAll Task 8 LangChain Session Memory tests PASSED!")
