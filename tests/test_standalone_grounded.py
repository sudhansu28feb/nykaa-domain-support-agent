"""
Tests for Standalone Grounded Generation and LangChain BaseChatModel adapter.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from langchain_core.messages import HumanMessage, SystemMessage
from nykaa_support.mock_llm.standalone import GroundedGenerator, MockLangChainChatModel


def test_grounded_generator_direct():
    generator = GroundedGenerator()
    query = "What is the return window for Nykaa products?"
    context = "Nykaa allows returns within 15 days of delivery for eligible unused beauty items."

    response = generator.generate(query=query, context=context)
    print("\n[Direct Grounded Response]:\n", response)

    assert "15 days" in response
    assert "Nykaa" in response
    assert "Hello!" in response
    print("test_grounded_generator_direct PASSED")


def test_grounded_generator_order_tracking():
    generator = GroundedGenerator()
    query = "Where is my order NYK-89211?"

    response = generator.generate(query=query)
    print("\n[Order Tracking Response]:\n", response)

    assert "NYK-89211" in response
    assert "in transit" in response
    print("test_grounded_generator_order_tracking PASSED")


def test_langchain_mock_chat_model():
    model = MockLangChainChatModel()
    messages = [
        SystemMessage(content="Policy: Free shipping on orders above INR 499."),
        HumanMessage(content="Do I get free shipping on a 600 rupee order?")
    ]

    result = model.invoke(messages)
    print("\n[LangChain Model Output]:\n", result.content)

    assert "Free shipping" in result.content or "499" in result.content
    assert "Nykaa" in result.content
    print("test_langchain_mock_chat_model PASSED")


if __name__ == "__main__":
    test_grounded_generator_direct()
    test_grounded_generator_order_tracking()
    test_langchain_mock_chat_model()
    print("\nAll standalone grounded generation tests completed successfully!")
