"""
Evaluation Judge for Accuracy, Grounding, Completeness, and Safety (Task 13).
"""

from typing import Optional
from nykaa_support.mock_llm.engine import (
    MockJudgeEvaluation,
    evaluate_response_deterministically,
)


class MockEvaluationJudge:
    """Deterministic evaluation judge for RAG and agent responses compliant with Task 13."""

    def __init__(
        self,
        accuracy_threshold: float = 0.70,
        grounding_threshold: float = 0.70,
        completeness_threshold: float = 0.70,
        safety_threshold: float = 0.70,
    ):
        self.accuracy_threshold = accuracy_threshold
        self.grounding_threshold = grounding_threshold
        self.completeness_threshold = completeness_threshold
        self.safety_threshold = safety_threshold

    def evaluate(
        self,
        query: str,
        context: str,
        response: str
    ) -> MockJudgeEvaluation:
        """Run deterministic evaluation and return structured Pydantic evaluation report."""
        eval_result = evaluate_response_deterministically(
            query=query,
            context=context,
            response=response
        )
        passed = (
            eval_result.accuracy >= self.accuracy_threshold
            and eval_result.grounding >= self.grounding_threshold
            and eval_result.completeness >= self.completeness_threshold
            and eval_result.safety >= self.safety_threshold
        )
        return MockJudgeEvaluation(
            accuracy=eval_result.accuracy,
            grounding=eval_result.grounding,
            completeness=eval_result.completeness,
            safety=eval_result.safety,
            passed=passed,
            verdict="PASS" if passed else "FAIL",
            reasoning=eval_result.reasoning
        )
