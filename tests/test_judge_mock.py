"""
Tests for Evaluation Judge with Structured Pydantic Output.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from nykaa_support.mock_llm.judge import MockEvaluationJudge
from nykaa_support.mock_llm.engine import MockJudgeEvaluation


def test_judge_grounded_response():
    judge = MockEvaluationJudge()
    query = "How long do I have to return my unopened lipstick?"
    context = "Nykaa policy permits returns within 15 days of delivery for unopened makeup items with seals intact."
    response = (
        "Hello! Thank you for contacting Nykaa. Based on our policy, you can return unopened "
        "lipstick within 15 days of delivery provided the safety seals and packaging remain intact. "
        "Please let us know if you need assistance creating a return request!"
    )

    evaluation: MockJudgeEvaluation = judge.evaluate(
        query=query,
        context=context,
        response=response
    )

    print("\n[Judge Report (Grounded)]:\n", evaluation.model_dump_json(indent=2))

    assert evaluation.passed is True
    assert evaluation.verdict == "PASS"
    assert evaluation.accuracy >= 0.70
    assert evaluation.grounding >= 0.70
    assert evaluation.completeness >= 0.70
    assert evaluation.safety >= 0.70
    assert len(evaluation.reasoning) > 10
    print("test_judge_grounded_response PASSED")


def test_judge_hallucinated_response():
    judge = MockEvaluationJudge()
    query = "How long do I have to return my unopened lipstick?"
    context = "Nykaa policy permits returns within 15 days of delivery for unopened makeup items."
    response = (
        "You can return it within 90 days for a full cash payout delivered to your doorstep tomorrow."
    )

    evaluation: MockJudgeEvaluation = judge.evaluate(
        query=query,
        context=context,
        response=response
    )

    print("\n[Judge Report (Ungrounded)]:\n", evaluation.model_dump_json(indent=2))

    assert evaluation.grounding < 0.70 or evaluation.safety < 0.80
    assert "Grounding" in evaluation.reasoning or "Accuracy" in evaluation.reasoning
    print("test_judge_hallucinated_response PASSED")


if __name__ == "__main__":
    test_judge_grounded_response()
    test_judge_hallucinated_response()
    print("\nAll Evaluation Judge tests completed successfully!")
