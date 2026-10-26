"""
Dedicated Test Suite for Task 9:
Pydantic Structured Output Schema (CrewSupportResponse) and Explicit Validation.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from nykaa_support.crew.schemas import (
    CrewSupportResponse,
    OrderDetails,
    StructuredOutputError,
    format_crew_response,
    validate_crew_response,
)


def test_valid_structured_output():
    """Verify valid dictionary and models validate cleanly."""
    valid_dict = {
        "query": "What is the status of my order NYK-10001?",
        "final_answer": "Your order NYK-10001 is Returned. Total value is ₹1552.",
        "query_type": "order_status",
        "order_id": "NYK-10001",
        "order_data": {
            "record_id": "NYK-10001",
            "status": "Returned",
            "order_value_inr": 1552,
            "days_since_created": 7,
            "delayed_shipment": False,
            "escalation_score": 0.0933,
            "escalate_recommended": False
        },
        "retrieved_policy_docs": ["01_return_window_by_category.md"],
        "escalation_score": 0.0933,
        "escalation_recommended": False
    }

    res = validate_crew_response(valid_dict)
    assert isinstance(res, CrewSupportResponse)
    assert res.query == "What is the status of my order NYK-10001?"
    assert res.query_type == "order_status"
    assert res.order_id == "NYK-10001"
    assert res.escalation_score == 0.0933
    assert res.order_data is not None
    assert res.order_data.status == "Returned"
    assert res.escalation_recommended is False
    print("test_valid_structured_output: PASSED (Valid payload accepted)")


def test_missing_required_fields():
    """Verify validation explicitly fails when required fields are missing."""
    # 1. Missing 'query'
    bad_dict_1 = {
        "final_answer": "Here is an answer.",
        "query_type": "policy_inquiry"
    }
    caught = False
    try:
        validate_crew_response(bad_dict_1)
    except StructuredOutputError as e:
        caught = True
        assert "query" in str(e).lower() or any(err.get("loc", [""])[0] == "query" for err in e.errors)
    assert caught, "Expected StructuredOutputError for missing 'query'"

    # 2. Missing 'final_answer'
    bad_dict_2 = {
        "query": "Where is my package?",
        "query_type": "order_status"
    }
    caught = False
    try:
        validate_crew_response(bad_dict_2)
    except StructuredOutputError as e:
        caught = True
        assert "final_answer" in str(e).lower() or any(err.get("loc", [""])[0] == "final_answer" for err in e.errors)
    assert caught, "Expected StructuredOutputError for missing 'final_answer'"

    # 3. Missing 'query_type'
    bad_dict_3 = {
        "query": "Where is my package?",
        "final_answer": "In transit."
    }
    caught = False
    try:
        validate_crew_response(bad_dict_3)
    except StructuredOutputError as e:
        caught = True
        assert "query_type" in str(e).lower() or any(err.get("loc", [""])[0] == "query_type" for err in e.errors)
    assert caught, "Expected StructuredOutputError for missing 'query_type'"

    print("test_missing_required_fields: PASSED (Missing fields rejected explicitly)")


def test_invalid_field_types():
    """Verify validation explicitly fails when fields have invalid types."""
    # 1. Non-numeric escalation_score
    bad_dict_1 = {
        "query": "Order status?",
        "final_answer": "Shipped.",
        "query_type": "order_status",
        "escalation_score": "high"  # string instead of float
    }
    caught = False
    try:
        validate_crew_response(bad_dict_1)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for non-numeric escalation_score"

    # 2. Non-boolean escalation_recommended
    bad_dict_2 = {
        "query": "Order status?",
        "final_answer": "Shipped.",
        "query_type": "order_status",
        "escalation_recommended": "strongly_recommended"  # non-boolean string
    }
    caught = False
    try:
        validate_crew_response(bad_dict_2)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for non-boolean escalation_recommended"

    # 3. Non-list retrieved_policy_docs
    bad_dict_3 = {
        "query": "Policy question?",
        "final_answer": "15 days.",
        "query_type": "policy_inquiry",
        "retrieved_policy_docs": 12345  # integer instead of list
    }
    caught = False
    try:
        validate_crew_response(bad_dict_3)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for non-list retrieved_policy_docs"

    print("test_invalid_field_types: PASSED (Invalid types rejected explicitly)")


def test_invalid_values_and_bounds():
    """Verify validation explicitly enforces value constraints and bounds."""
    # 1. Score > 1.0
    bad_dict_1 = {
        "query": "Status check",
        "final_answer": "Delayed.",
        "query_type": "order_status",
        "escalation_score": 1.75
    }
    caught = False
    try:
        validate_crew_response(bad_dict_1)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for escalation_score > 1.0"

    # 2. Score < 0.0
    bad_dict_2 = {
        "query": "Status check",
        "final_answer": "On schedule.",
        "query_type": "order_status",
        "escalation_score": -0.45
    }
    caught = False
    try:
        validate_crew_response(bad_dict_2)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for escalation_score < 0.0"

    # 3. Invalid query_type literal
    bad_dict_3 = {
        "query": "Hello",
        "final_answer": "Hi",
        "query_type": "unrecognized_category_123"
    }
    caught = False
    try:
        validate_crew_response(bad_dict_3)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for invalid query_type literal"

    # 4. Empty/whitespace query string
    bad_dict_4 = {
        "query": "    ",
        "final_answer": "Some answer",
        "query_type": "general"
    }
    caught = False
    try:
        validate_crew_response(bad_dict_4)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for empty query"

    # 5. Empty/whitespace final_answer string
    bad_dict_5 = {
        "query": "Some query",
        "final_answer": "",
        "query_type": "general"
    }
    caught = False
    try:
        validate_crew_response(bad_dict_5)
    except StructuredOutputError:
        caught = True
    assert caught, "Expected StructuredOutputError for empty final_answer"

    print("test_invalid_values_and_bounds: PASSED (Value bounds strictly enforced)")


def test_format_crew_response_factory():
    """Verify the factory helper builds conforming, validated models."""
    from dataset import check_order_status

    order_info = check_order_status("NYK-10002")
    resp = format_crew_response(
        query="Can I return order NYK-10002?",
        final_answer="Order NYK-10002 was shipped. Returns are accepted within 15 days.",
        order_info=order_info,
        policy_docs=["01_return_window_by_category.md"]
    )

    assert isinstance(resp, CrewSupportResponse)
    assert resp.order_id == "NYK-10002"
    assert resp.query_type == "combined"
    assert resp.escalation_score == order_info["escalation_score"]
    assert resp.order_data is not None
    assert resp.order_data.record_id == "NYK-10002"
    assert resp.order_data.status == order_info["status"]
    assert len(resp.retrieved_policy_docs) == 1

    print("test_format_crew_response_factory: PASSED (Factory output validated)")


if __name__ == "__main__":
    test_valid_structured_output()
    test_missing_required_fields()
    test_invalid_field_types()
    test_invalid_values_and_bounds()
    test_format_crew_response_factory()
    print("\nAll Task 9 Structured Output Schema tests PASSED!")
