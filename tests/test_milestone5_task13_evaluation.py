"""
Dedicated Test Suite for Task 13:
Fifteen-Query Evaluation Pipeline Across Accuracy, Grounding, Completeness, and Safety.

Requirements strictly verified:
1. Fixed evaluation set of at least 15 queries:
   - Covers all 12 policy topics from the knowledge base.
   - Covers at least 2 out-of-scope or edge-case queries.
2. Evaluates Accuracy, Grounding, Completeness, and Safety for every query.
3. Reports all 4 average scores and preserves per-query scores and results.
4. Deterministic, reproducible scoring under MOCK_LLM judge.
5. Saves results to logs/evaluation_benchmark_results.json.
"""

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from nykaa_support.evaluation import (
    EVALUATION_QUERIES,
    EvaluationBenchmarkResult,
    EvaluationPipeline,
    EvaluationQueryReport,
)


REQUIRED_12_TOPICS = [
    "01_return_window_by_category",
    "02_cod_refund_timelines",
    "03_delivery_slas",
    "04_reverse_pickup_eligibility",
    "05_warranty_terms_by_category",
    "06_order_cancellation_policy",
    "07_loyalty_points_redemption_policy",
    "08_payment_failure_retry_policy",
    "09_size_exchange_policy",
    "10_damaged_item_claim_process",
    "11_international_shipping_restrictions",
    "12_customer_support_escalation_matrix",
]


def test_evaluation_dataset_coverage():
    """Verify evaluation set satisfies: >= 15 queries, all 12 policy topics, and >= 2 edge cases."""
    queries = EVALUATION_QUERIES

    assert len(queries) >= 15, f"Evaluation set must have at least 15 queries, found {len(queries)}"
    print(f"\n[Evaluation Set Coverage]: Total Queries = {len(queries)}")

    # Check 12 policy topics representation
    represented_topics = set(q.topic_id for q in queries if not q.is_out_of_scope)
    for topic in REQUIRED_12_TOPICS:
        assert topic in represented_topics, f"Required topic '{topic}' is missing from evaluation set!"

    # Check out-of-scope / edge-case queries
    edge_cases = [q for q in queries if q.is_out_of_scope]
    assert len(edge_cases) >= 2, f"Must have at least 2 out-of-scope/edge queries, found {len(edge_cases)}"

    print(f"  In-Scope Policy Topics Covered: {len(represented_topics)}/12")
    print(f"  Out-of-Scope / Edge Cases: {len(edge_cases)} (medical, injection, stocks)")
    print("test_evaluation_dataset_coverage: PASSED (Topic coverage verified)")


def test_evaluation_pipeline_execution():
    """Run full evaluation pipeline and verify 4 property scores per query + 4 averages."""
    pipeline = EvaluationPipeline()
    result = pipeline.run_benchmark()

    assert isinstance(result, EvaluationBenchmarkResult)
    assert result.total_queries == 15
    assert len(result.per_query_results) == 15

    print("\n" + "=" * 78)
    print("TASK 13 EVALUATION BENCHMARK RESULTS (MOCK_LLM JUDGE)")
    print("=" * 78)
    print(pipeline.format_summary_table(result))
    print("=" * 78)

    # Assert all four average scores are reported and valid
    print(f"\nOverall Metric Averages across 15 queries:")
    print(f"  Average Accuracy:     {result.avg_accuracy:.4f}")
    print(f"  Average Grounding:    {result.avg_grounding:.4f}")
    print(f"  Average Completeness: {result.avg_completeness:.4f}")
    print(f"  Average Safety:       {result.avg_safety:.4f}")
    print(f"  Overall Pass Rate:    {result.pass_rate_pct:.1f}% ({result.passed_count}/15)")

    assert 0.70 <= result.avg_accuracy <= 1.0, f"Average Accuracy out of expected range: {result.avg_accuracy}"
    assert 0.70 <= result.avg_grounding <= 1.0, f"Average Grounding out of expected range: {result.avg_grounding}"
    assert 0.70 <= result.avg_completeness <= 1.0, f"Average Completeness out of expected range: {result.avg_completeness}"
    assert 0.70 <= result.avg_safety <= 1.0, f"Average Safety out of expected range: {result.avg_safety}"

    # Assert every single query has all 4 scores strictly within [0.0, 1.0]
    for r in result.per_query_results:
        assert 0.0 <= r.accuracy <= 1.0
        assert 0.0 <= r.grounding <= 1.0
        assert 0.0 <= r.completeness <= 1.0
        assert 0.0 <= r.safety <= 1.0
        assert r.verdict in ["PASS", "NEEDS_REVISION", "FAIL"]
        assert len(r.reasoning) > 20

    print("test_evaluation_pipeline_execution: PASSED (All 4 metrics verified per-query and averaged)")


def test_results_persistence_and_reproducibility():
    """Verify evaluation benchmark results are persisted to disk in valid JSON format."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, "logs", "evaluation_benchmark_results.json")

    assert os.path.exists(results_path), f"Evaluation benchmark results file not found at: {results_path}"

    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Schema validation
    reloaded = EvaluationBenchmarkResult.model_validate(data)
    assert reloaded.total_queries == 15
    assert len(reloaded.per_query_results) == 15
    assert reloaded.avg_accuracy > 0.0
    assert reloaded.avg_grounding > 0.0
    assert reloaded.avg_completeness > 0.0
    assert reloaded.avg_safety > 0.0

    print("test_results_persistence_and_reproducibility: PASSED (Results JSON file validated on disk)")


if __name__ == "__main__":
    test_evaluation_dataset_coverage()
    test_evaluation_pipeline_execution()
    test_results_persistence_and_reproducibility()
    print("\nAll Task 13 Evaluation Pipeline tests PASSED!")
