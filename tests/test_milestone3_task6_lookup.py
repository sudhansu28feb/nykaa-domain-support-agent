"""
Dedicated Test Suite for Task 6:
Order-Status Lookup Tool and Designed Escalation Scoring.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from dataset import (
    ORDERS,
    check_order_status,
    get_order_by_id,
)


def test_lookup_valid_records():
    """Verify lookup on all 50 generated orders returns correct fields and status."""
    for order in ORDERS:
        result = check_order_status(order.record_id)

        assert "error" not in result, f"Unexpected error for valid record {order.record_id}"
        assert result["record_id"] == order.record_id
        assert result["status"] == order.status
        assert result["order_value_inr"] == order.order_value_inr
        assert result["days_since_created"] == order.days_since_created
        assert result["delayed_shipment"] == order.delayed_shipment
        assert "escalation_score" in result
        assert "escalate_recommended" in result

    print(f"test_lookup_valid_records: PASSED (All {len(ORDERS)} records successfully retrieved)")


def test_lookup_invalid_and_edge_records():
    """Verify lookup behavior for non-existent, malformed, and empty record IDs."""
    invalid_ids = ["NYK-99999", "INVALID_ID", "", "   ", "ORD-UNKNOWN", "12345"]
    for inv_id in invalid_ids:
        result = check_order_status(inv_id)
        assert "error" in result, f"Expected error key for invalid record '{inv_id}'"
        assert result["status"] == "Unknown"
        assert result["order_value_inr"] == 0
        assert result["escalation_score"] == 0.0
        assert result["escalate_recommended"] is False

    print("test_lookup_invalid_and_edge_records: PASSED (Invalid IDs handled gracefully)")


def test_escalation_score_bounds_and_formula():
    """
    Verify escalation score formula:
    escalation_score = 0.60 * delay_signal + 0.40 * (days_since_created / 30.0)
    Ensures it is strictly within [0.0, 1.0] and NOT a bare boolean OR.
    """
    scores = []
    for order in ORDERS:
        res = check_order_status(order.record_id)
        score = res["escalation_score"]
        scores.append(score)

        # 1. Bounds check
        assert 0.0 <= score <= 1.0, f"Score {score} out of bounds [0, 1]"

        # 2. Formula verification
        delay_val = 1.0 if order.delayed_shipment else 0.0
        recency_val = order.days_since_created / 30.0
        expected_score = round(0.60 * delay_val + 0.40 * recency_val, 4)
        assert abs(score - expected_score) < 1e-4, f"Score mismatch: got {score}, expected {expected_score}"

        # 3. Not a bare boolean OR check
        # A bare boolean OR would only be 0.0 or 1.0. Here we have a continuous spectrum
        if order.delayed_shipment:
            assert score >= 0.60, f"Delayed order must have score >= 0.60, got {score}"
        else:
            assert score <= 0.40, f"Non-delayed order must have score <= 0.40, got {score}"

    # Verify score variance (proves non-binary continuous metric)
    unique_scores = set(scores)
    assert len(unique_scores) >= 15, f"Expected >= 15 distinct score values across 50 orders, got {len(unique_scores)}"
    print(f"test_escalation_score_bounds_and_formula: PASSED ({len(unique_scores)} distinct continuous scores)")


def test_escalation_threshold_and_dataset_percentile():
    """
    Verify threshold = 0.72 requires delayed_shipment=True and days_since_created >= 9 days.
    In the current generated dataset, 4 of 10 delayed orders meet this threshold.
    """
    delayed_orders = [o for o in ORDERS if o.delayed_shipment]
    non_delayed_orders = [o for o in ORDERS if not o.delayed_shipment]

    assert len(delayed_orders) == 10  # 20.0% of 50
    assert len(non_delayed_orders) == 40

    # Non-delayed orders should never recommend escalation (max score = 0.40 < 0.72)
    for o in non_delayed_orders:
        res = check_order_status(o.record_id)
        assert res["escalate_recommended"] is False

    # Check delayed orders
    escalated_delayed = []
    for o in delayed_orders:
        res = check_order_status(o.record_id)
        if res["escalate_recommended"]:
            escalated_delayed.append(res)
            # Days since created must be >= 9 days for score >= 0.72
            assert res["days_since_created"] >= 9
            assert res["escalation_score"] >= 0.72

    print(f"test_escalation_threshold_and_dataset_percentile: PASSED ({len(escalated_delayed)} delayed orders flagged for escalation)")


if __name__ == "__main__":
    test_lookup_valid_records()
    test_lookup_invalid_and_edge_records()
    test_escalation_score_bounds_and_formula()
    test_escalation_threshold_and_dataset_percentile()
    print("\nAll Task 6 Order-Status Lookup and Escalation Scoring tests PASSED!")
