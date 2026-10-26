"""
Comprehensive Audit Test Suite for Milestone 1 (Tasks 1 & 2):
Validates Dataset Schema, Generator (dataset.py), check_order_status (Task 6),
and All 12 Distinct Knowledge-Base Policy Documents (2-5 sentences each).
"""

import json
import os
import re
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Verify top-level dataset.py deliverable and ORDERS export
import dataset
from dataset import (
    DEFAULT_ORDER_COUNT,
    DEFAULT_SEED,
    ORDERS,
    PRICE_RANGE_REASONING,
    REQUIRED_CATEGORIES,
    REQUIRED_STATUSES,
    OrderRecord,
    check_order_status,
    export_orders_to_json,
    generate_orders,
    get_dataset_summary,
    get_order_by_id,
)


def test_root_dataset_deliverable_and_orders():
    """Verify that root-level dataset.py delivers the canonical ORDERS list."""
    assert hasattr(dataset, "ORDERS"), "dataset.py missing module-level ORDERS export"
    assert len(ORDERS) >= 40, f"Expected >= 40 orders in ORDERS list, got {len(ORDERS)}"
    assert isinstance(PRICE_RANGE_REASONING, str) and len(PRICE_RANGE_REASONING) > 20
    print(f"test_root_dataset_deliverable_and_orders: PASSED ({len(ORDERS)} orders exported in dataset.ORDERS)")


def test_dataset_required_schema_fields():
    """Verify that every record contains the exact 6 fields specified in the capstone brief."""
    for i, o in enumerate(ORDERS):
        # 1. record_id
        assert hasattr(o, "record_id") and isinstance(o.record_id, str) and len(o.record_id) > 0, f"Record {i} missing valid record_id"
        # 2. category
        assert hasattr(o, "category") and o.category in REQUIRED_CATEGORIES, f"Record {i} has invalid category: {o.category}"
        # 3. status
        assert hasattr(o, "status") and o.status in REQUIRED_STATUSES, f"Record {i} has invalid status: {o.status}"
        # 4. order_value_inr
        assert hasattr(o, "order_value_inr") and isinstance(o.order_value_inr, (int, float)) and o.order_value_inr >= 100, f"Record {i} has invalid order_value_inr"
        # 5. days_since_created (integer from 0 through 30)
        assert hasattr(o, "days_since_created") and isinstance(o.days_since_created, int) and (0 <= o.days_since_created <= 30), f"Record {i} invalid days_since_created: {o.days_since_created}"
        # 6. delayed_shipment (boolean)
        assert hasattr(o, "delayed_shipment") and isinstance(o.delayed_shipment, bool), f"Record {i} invalid delayed_shipment: {o.delayed_shipment}"

        # Preserved domain fields check
        assert hasattr(o, "customer_name") and len(o.customer_name) > 0
        assert hasattr(o, "carrier") and len(o.carrier) > 0

    print(f"test_dataset_required_schema_fields: PASSED ({len(ORDERS)} records verified with exact schema)")


def test_category_coverage():
    """Verify all 5 required categories have at least 3 records."""
    summary = get_dataset_summary(ORDERS)
    for cat in REQUIRED_CATEGORIES:
        count = summary.categories.get(cat, 0)
        assert count >= 3, f"Category '{cat}' has {count} records (minimum 3 required)"

    print("test_category_coverage: PASSED (All 5 categories have >= 3 records)")
    for cat in REQUIRED_CATEGORIES:
        print(f"  - {cat}: {summary.categories[cat]} records")


def test_status_coverage():
    """Verify all 5 required statuses have at least 1 record."""
    summary = get_dataset_summary(ORDERS)
    for status in REQUIRED_STATUSES:
        count = summary.statuses.get(status, 0)
        assert count >= 1, f"Status '{status}' has {count} records (minimum 1 required)"

    print("test_status_coverage: PASSED (All 5 statuses have >= 1 record)")
    for status in REQUIRED_STATUSES:
        print(f"  - {status}: {summary.statuses[status]} records")


def test_delayed_shipment_rate():
    """Verify delayed shipment rate is strictly between 10% and 30%."""
    summary = get_dataset_summary(ORDERS)
    assert 10.0 <= summary.delayed_percentage <= 30.0, f"Delayed shipment rate {summary.delayed_percentage}% outside 10%-30% range"
    assert summary.delayed_orders_count == 10  # Exactly 20.0% of 50
    print(f"test_delayed_shipment_rate: PASSED ({summary.delayed_orders_count}/{summary.total_orders} = {summary.delayed_percentage}%)")


def test_days_since_created_range():
    """Verify days_since_created integers are strictly within [0, 30]."""
    summary = get_dataset_summary(ORDERS)
    assert summary.min_days_since_created >= 0
    assert summary.max_days_since_created <= 30
    print(f"test_days_since_created_range: PASSED (Range: [{summary.min_days_since_created}, {summary.max_days_since_created}])")


def test_deterministic_regeneration():
    """Verify that repeated generation with the same seed produces identical results."""
    run_a = generate_orders(seed=DEFAULT_SEED, count=DEFAULT_ORDER_COUNT)
    run_b = generate_orders(seed=DEFAULT_SEED, count=DEFAULT_ORDER_COUNT)

    for a, b in zip(run_a, run_b):
        assert a.model_dump() == b.model_dump()

    print("test_deterministic_regeneration: PASSED (100% bitwise identical across runs)")


def test_check_order_status_tool():
    """Verify Task 6 lookup tool returns status, order_value_inr, and designed escalation_score in [0, 1]."""
    for o in ORDERS[:5]:
        res = check_order_status(o.record_id)
        assert "status" in res and res["status"] == o.status
        assert "order_value_inr" in res and res["order_value_inr"] == o.order_value_inr
        assert "escalation_score" in res
        score = res["escalation_score"]
        assert 0.0 <= score <= 1.0, f"escalation_score {score} outside [0, 1]"

        # Check recency and delay correlation
        if o.delayed_shipment:
            assert score >= 0.60, f"Delayed shipment expected score >= 0.60, got {score}"
        else:
            assert score <= 0.40, f"Non-delayed shipment expected score <= 0.40, got {score}"

    print("test_check_order_status_tool: PASSED (Tool signature and escalation score verified)")


def count_sentences(text: str) -> int:
    """Count sentences in markdown text excluding headers."""
    lines = [line.strip() for line in text.split("\n") if line.strip() and not line.strip().startswith("#")]
    clean_text = " ".join(lines)
    # Match sentences ending with ., !, or ?
    sentences = re.findall(r'[^.!?]+[.!?]', clean_text)
    return len(sentences)


def test_all_12_distinct_knowledge_base_documents():
    """
    Verify Task 2 requirements:
    - Author >=12 distinct documents
    - 2 to 5 sentences each
    - Covering all 12 required topics from the brief
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    policies_dir = os.path.join(base_dir, "data", "policies")

    required_topics = [
        (1, "01_return_window_by_category.md", "return window by product category", ["return", "15 calendar days", "cosmetics", "apparel", "footwear"]),
        (2, "02_cod_refund_timelines.md", "COD refund timelines", ["cash on delivery", "refund", "24 business hours", "neft", "wallet"]),
        (3, "03_delivery_slas.md", "delivery SLAs", ["delivery", "slas", "tier 1 metro", "2 to 3 business days", "499", "70"]),
        (4, "04_reverse_pickup_eligibility.md", "reverse-pickup eligibility", ["reverse-pickup", "19,000", "doorstep", "24 to 48", "150"]),
        (5, "05_warranty_terms_by_category.md", "warranty terms by category", ["warranty", "1 to 2 years", "apparel", "30-day", "shelf-life"]),
        (6, "06_order_cancellation_policy.md", "order-cancellation policy", ["cancellation", "my orders", "placed", "shipped", "rto"]),
        (7, "07_loyalty_points_redemption_policy.md", "loyalty-points redemption policy", ["loyalty", "privé", "100", "50%", "12 calendar months"]),
        (8, "08_payment_failure_retry_policy.md", "payment-failure/retry policy", ["payment", "failure", "24 to 48", "retry link", "30 minutes"]),
        (9, "09_size_exchange_policy.md", "size-exchange policy", ["size exchange", "footwear", "apparel", "15 calendar days", "simultaneous"]),
        (10, "10_damaged_item_claim_process.md", "damaged-item claim process", ["damaged", "48 hours", "photographic", "outer cardboard", "2,000"]),
        (11, "11_international_shipping_restrictions.md", "international shipping restrictions", ["international", "10 to 15 business days", "aerosol", "300 ml"]),
        (12, "12_customer_support_escalation_matrix.md", "the customer-support escalation matrix", ["escalation matrix", "level 1", "level 2", "level 3", "grievance officer"]),
    ]

    all_files = [f for f in os.listdir(policies_dir) if f.endswith(".md")]
    assert len(all_files) >= 12, f"Expected >= 12 policy documents, found {len(all_files)}"

    print(f"\ntest_all_12_distinct_knowledge_base_documents: Found {len(all_files)} documents")

    for topic_num, filename, topic_desc, keywords in required_topics:
        filepath = os.path.join(policies_dir, filename)
        assert os.path.exists(filepath), f"Missing document for Topic {topic_num}: {filename}"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Sentence count check (Requirement: 2 to 5 sentences each)
        s_count = count_sentences(content)
        assert 2 <= s_count <= 5, f"{filename} has {s_count} sentences (requirement: 2-5 sentences each)"

        # Keyword domain check
        lowered = content.lower()
        for kw in keywords:
            assert kw.lower() in lowered, f"{filename} missing required domain concept: '{kw}'"

        print(f"  [PASS] Topic {topic_num:02d}: {filename:<42} ({s_count} sentences) - {topic_desc}")

    print("test_all_12_distinct_knowledge_base_documents: PASSED (All 12 documents verified with 2-5 sentences)")


if __name__ == "__main__":
    test_root_dataset_deliverable_and_orders()
    test_dataset_required_schema_fields()
    test_category_coverage()
    test_status_coverage()
    test_delayed_shipment_rate()
    test_days_since_created_range()
    test_deterministic_regeneration()
    test_check_order_status_tool()
    test_all_12_distinct_knowledge_base_documents()
    print("\nAll Milestone 1 audited requirements passed with 100% compliance!")
