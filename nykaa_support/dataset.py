"""
Deterministic Nykaa E-Commerce Order Dataset Generator.

Fully compliant with original Capstone Project Brief specifications:
Required Fields:
- record_id
- category (Apparel, Electronics, Home, Footwear, Beauty)
- status (Placed, Shipped, Delivered, Returned, Refunded)
- order_value_inr
- days_since_created (integer 0-30)
- delayed_shipment (boolean, 10-30% rate)
"""

import json
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

DEFAULT_SEED = 42
DEFAULT_ORDER_COUNT = 50

# The 5 Required Categories from Capstone Project Brief
REQUIRED_CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]

# The 5 Required Statuses from Capstone Project Brief
REQUIRED_STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]

PRODUCTS_BY_CATEGORY: Dict[str, List[Dict[str, int]]] = {
    "Apparel": [
        {"name": "Gajra Gang by Nykaa Embroidered Anarkali Kurta", "min_price": 1899, "max_price": 2799},
        {"name": "Twenty Dresses by Nykaa Polka Dot Midi Dress", "min_price": 1299, "max_price": 1899},
        {"name": "Nykd by Nykaa Cotton Comfort Lounge Pyjama Set", "min_price": 899, "max_price": 1399},
        {"name": "RSVP by Nykaa Satin Slip Evening Dress", "min_price": 1999, "max_price": 2899},
        {"name": "Nykaa Fashion Velvet Winter Shrug", "min_price": 1499, "max_price": 2199},
    ],
    "Electronics": [
        {"name": "Philips MoistureProtect Sensor Hair Dryer", "min_price": 2499, "max_price": 3499},
        {"name": "Vega Adore Hair Straightener with Ceramic Plates", "min_price": 1199, "max_price": 1699},
        {"name": "Alan Truman Blow Dryer Brush", "min_price": 2299, "max_price": 3199},
        {"name": "Foreo Luna Mini Sonic Facial Cleanser", "min_price": 3499, "max_price": 4899},
        {"name": "Nova Foldable Quick-Dry Travel Hair Dryer", "min_price": 599, "max_price": 899},
    ],
    "Home": [
        {"name": "Nykaa Scented Soy Wax Candle - French Vanilla", "min_price": 499, "max_price": 799},
        {"name": "Nykaa Maison Ultrasonic Aroma Diffuser", "min_price": 1599, "max_price": 2399},
        {"name": "Wonderchef Granite Non-Stick Cookware Set 3pc", "min_price": 2199, "max_price": 3299},
        {"name": "Spaces Luxury 100% Cotton Bath Towel Set", "min_price": 899, "max_price": 1499},
        {"name": "Ellementry Handcrafted Ceramic Coffee Mug Set", "min_price": 749, "max_price": 1199},
    ],
    "Footwear": [
        {"name": "Nykaa Fashion Embellished Festive Juttis", "min_price": 1299, "max_price": 1899},
        {"name": "Truffle Collection Block Heel Party Sandals", "min_price": 1499, "max_price": 2299},
        {"name": "Carlton London Classic White Court Sneakers", "min_price": 1799, "max_price": 2699},
        {"name": "Bata Red Label Pointed Toe Stiletto Heels", "min_price": 1399, "max_price": 1999},
        {"name": "Metro Cushioned Slip-On Everyday Flats", "min_price": 899, "max_price": 1399},
    ],
    "Beauty": [
        {"name": "Nykaa Matte to Last Liquid Lipstick - Chai", "min_price": 499, "max_price": 649},
        {"name": "The Ordinary Niacinamide 10% + Zinc 1% 30ml", "min_price": 600, "max_price": 750},
        {"name": "Dot & Key Vitamin C + E Sunscreen SPF 50", "min_price": 445, "max_price": 595},
        {"name": "Kay Beauty 24Hr Matte Liquid Eyeliner", "min_price": 749, "max_price": 899},
        {"name": "Forest Essentials Soundarya Radiance Cream", "min_price": 3450, "max_price": 4250},
    ],
}

CUSTOMER_POOL = [
    ("Priya Sharma", "priya.sharma@example.com", "+91-98112-34501", "Mumbai"),
    ("Ananya Verma", "ananya.v@example.com", "+91-98223-45602", "New Delhi"),
    ("Rohan Mehta", "rohan.mehta@example.com", "+91-98334-56703", "Bengaluru"),
    ("Deepika Nair", "deepika.nair@example.com", "+91-98445-67804", "Chennai"),
    ("Sneha Roy", "sneha.roy@example.com", "+91-98556-78905", "Kolkata"),
    ("Aditya Patel", "aditya.patel@example.com", "+91-98667-89006", "Ahmedabad"),
    ("Pooja Iyer", "pooja.iyer@example.com", "+91-98778-90107", "Hyderabad"),
    ("Vikram Singh", "vikram.s@example.com", "+91-98889-01208", "Jaipur"),
    ("Neha Kulkarni", "neha.k@example.com", "+91-98990-12309", "Pune"),
    ("Arjun Kapoor", "arjun.k@example.com", "+91-98101-23410", "Chandigarh"),
    ("Tanvi Deshmukh", "tanvi.d@example.com", "+91-98212-34511", "Mumbai"),
    ("Manish Tiwari", "manish.t@example.com", "+91-98323-45612", "Lucknow"),
    ("Shreya Sen", "shreya.sen@example.com", "+91-98434-56713", "Kolkata"),
    ("Kabir Das", "kabir.das@example.com", "+91-98545-67814", "Guwahati"),
    ("Divya Menon", "divya.menon@example.com", "+91-98656-78915", "Kochi"),
]

PAYMENT_METHODS = ["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking"]
PAYMENT_WEIGHTS = [0.45, 0.25, 0.15, 0.10, 0.05]

STATUS_WEIGHTS = [0.15, 0.25, 0.40, 0.10, 0.10]  # Placed, Shipped, Delivered, Returned, Refunded
CARRIERS = ["BlueDart", "Delhivery", "Shadowfax", "DTDC"]

DELAY_REASONS = [
    "Severe weather disruption at regional distribution hub",
    "High festive order volume backlog at transit sorting facility",
    "Address verification pending with local delivery associate",
    "Road transit congestion along national highway freight corridor",
]


class OrderRecord(BaseModel):
    """
    Schema strictly implementing the Capstone Project Brief requirements,
    with preserved Nykaa e-commerce domain attributes.
    """

    # --- Exact Required Brief Fields ---
    record_id: str = Field(description="Unique record identifier (e.g., NYK-10001)")
    category: str = Field(description="Required category: Apparel, Electronics, Home, Footwear, or Beauty")
    status: str = Field(description="Required status: Placed, Shipped, Delivered, Returned, or Refunded")
    order_value_inr: int = Field(ge=100, le=10000, description="Order transaction value in INR")
    days_since_created: int = Field(ge=0, le=30, description="Number of days since order creation (0 to 30)")
    delayed_shipment: bool = Field(description="True if shipment is delayed beyond delivery SLA")

    # --- Preserved Nykaa Domain Extensions ---
    order_id: str = Field(description="Customer-facing Nykaa order reference")
    customer_name: str = Field(description="Customer full name")
    customer_email: str = Field(description="Customer email address")
    customer_phone: str = Field(description="Customer contact phone")
    order_date: str = Field(description="Date of order creation (YYYY-MM-DD)")
    product_name: str = Field(description="Purchased item title")
    payment_method: str = Field(description="Selected payment method")
    delivery_date: Optional[str] = Field(default=None, description="Actual or estimated delivery date")
    shipping_city: str = Field(description="Destination city")
    carrier: str = Field(description="Logistics partner")
    tracking_number: str = Field(description="Carrier tracking identifier")
    delay_reason: Optional[str] = Field(default=None, description="Logistics explanation if delayed")

    # Backward compatibility aliases for property access
    @property
    def price(self) -> int:
        return self.order_value_inr

    @property
    def product_category(self) -> str:
        return self.category

    @property
    def order_status(self) -> str:
        return self.status

    @property
    def is_delayed(self) -> bool:
        return self.delayed_shipment


class DatasetSummary(BaseModel):
    """Summary metrics of the generated dataset."""

    total_orders: int
    categories: Dict[str, int]
    statuses: Dict[str, int]
    delayed_orders_count: int
    delayed_percentage: float
    total_revenue_inr: int
    average_order_value_inr: float
    min_days_since_created: int
    max_days_since_created: int


def generate_orders(seed: int = DEFAULT_SEED, count: int = DEFAULT_ORDER_COUNT) -> List[OrderRecord]:
    """Generate a deterministic, seeded list of Nykaa order records satisfying all capstone criteria."""
    rng = random.Random(seed)
    reference_date = datetime(2026, 10, 25)

    orders: List[OrderRecord] = []

    # Exactly 20.0% delayed shipments (e.g. 10 out of 50)
    target_delayed_count = round(count * 0.20)
    delayed_indices = set(rng.sample(range(count), target_delayed_count))

    # Guarantee at least 3 records per category by initial round-robin
    category_assignments: List[str] = []
    for cat in REQUIRED_CATEGORIES:
        category_assignments.extend([cat] * 3)  # 15 guaranteed records
    # Fill remaining
    while len(category_assignments) < count:
        category_assignments.append(rng.choice(REQUIRED_CATEGORIES))
    rng.shuffle(category_assignments)

    # Guarantee at least 1 record per status by initial round-robin
    status_assignments: List[str] = list(REQUIRED_STATUSES)  # 5 guaranteed records
    # Fill remaining based on status weights
    remaining_count = count - len(status_assignments)
    status_assignments.extend(rng.choices(REQUIRED_STATUSES, weights=STATUS_WEIGHTS, k=remaining_count))
    rng.shuffle(status_assignments)

    for i in range(count):
        order_num = 10001 + i
        order_id = f"NYK-{order_num}"
        record_id = order_id

        category = category_assignments[i]
        status = status_assignments[i]

        product_info = rng.choice(PRODUCTS_BY_CATEGORY[category])
        product_name = product_info["name"]
        order_value_inr = rng.randint(product_info["min_price"], product_info["max_price"])

        # days_since_created: strictly integer between 0 and 30
        days_since_created = rng.randint(0, 30)
        order_dt = reference_date - timedelta(days=days_since_created)
        order_date_str = order_dt.strftime("%Y-%m-%d")

        cust = rng.choice(CUSTOMER_POOL)
        customer_name, customer_email, customer_phone, default_city = cust

        payment_method = rng.choices(PAYMENT_METHODS, weights=PAYMENT_WEIGHTS, k=1)[0]
        carrier = rng.choice(CARRIERS)

        delayed_shipment = i in delayed_indices
        delay_reason = rng.choice(DELAY_REASONS) if delayed_shipment else None

        carrier_prefix = carrier[:2].upper()
        tracking_num = f"{carrier_prefix}-{rng.randint(100000, 999999)}"

        # Realistic delivery dates based on status
        if status == "Delivered":
            deliv_offset = min(days_since_created, rng.randint(2, 5))
            delivery_dt = order_dt + timedelta(days=deliv_offset)
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d")
        elif status == "Shipped":
            eta_days = rng.randint(5, 8) if delayed_shipment else rng.randint(2, 4)
            delivery_dt = order_dt + timedelta(days=eta_days)
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d (Estimated)")
        elif status == "Placed":
            delivery_dt = order_dt + timedelta(days=4)
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d (Estimated)")
        elif status == "Returned":
            deliv_offset = min(days_since_created, rng.randint(3, 6))
            delivery_dt = order_dt + timedelta(days=deliv_offset)
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d (Returned)")
        elif status == "Refunded":
            deliv_offset = min(days_since_created, rng.randint(4, 7))
            delivery_dt = order_dt + timedelta(days=deliv_offset)
            delivery_date_str = delivery_dt.strftime("%Y-%m-%d (Refunded)")
        else:
            delivery_date_str = None

        record = OrderRecord(
            record_id=record_id,
            category=category,
            status=status,
            order_value_inr=order_value_inr,
            days_since_created=days_since_created,
            delayed_shipment=delayed_shipment,
            order_id=order_id,
            customer_name=customer_name,
            customer_email=customer_email,
            customer_phone=customer_phone,
            order_date=order_date_str,
            product_name=product_name,
            payment_method=payment_method,
            delivery_date=delivery_date_str,
            shipping_city=default_city,
            carrier=carrier,
            tracking_number=tracking_num,
            delay_reason=delay_reason,
        )
        orders.append(record)

    return orders


def get_dataset_summary(orders: List[OrderRecord]) -> DatasetSummary:
    """Calculate summary statistics for a collection of orders."""
    total = len(orders)
    cat_counts: Dict[str, int] = {c: 0 for c in REQUIRED_CATEGORIES}
    status_counts: Dict[str, int] = {s: 0 for s in REQUIRED_STATUSES}
    delayed_count = 0
    total_rev = 0
    days_list: List[int] = []

    for o in orders:
        cat_counts[o.category] = cat_counts.get(o.category, 0) + 1
        status_counts[o.status] = status_counts.get(o.status, 0) + 1
        if o.delayed_shipment:
            delayed_count += 1
        total_rev += o.order_value_inr
        days_list.append(o.days_since_created)

    delayed_pct = round((delayed_count / total) * 100, 2) if total > 0 else 0.0
    aov = round(total_rev / total, 2) if total > 0 else 0.0
    min_days = min(days_list) if days_list else 0
    max_days = max(days_list) if days_list else 0

    return DatasetSummary(
        total_orders=total,
        categories=cat_counts,
        statuses=status_counts,
        delayed_orders_count=delayed_count,
        delayed_percentage=delayed_pct,
        total_revenue_inr=total_rev,
        average_order_value_inr=aov,
        min_days_since_created=min_days,
        max_days_since_created=max_days,
    )


def export_orders_to_json(filepath: Optional[str] = None, seed: int = DEFAULT_SEED, count: int = DEFAULT_ORDER_COUNT) -> str:
    """Export deterministic orders to JSON file and return absolute path."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, "orders.json")

    orders = generate_orders(seed=seed, count=count)
    data = [o.model_dump() for o in orders]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return os.path.abspath(filepath)


def get_order_by_id(record_or_order_id: str, orders: Optional[List[OrderRecord]] = None) -> Optional[OrderRecord]:
    """Retrieve an order record by record_id or order_id."""
    if orders is None:
        orders = ORDERS
    clean_id = record_or_order_id.strip().upper()
    for o in orders:
        if o.record_id.upper() == clean_id or o.order_id.upper() == clean_id:
            return o
    return None


def check_order_status(record_id: str, orders: Optional[List[OrderRecord]] = None) -> dict:
    """
    Lookup tool compliant with Task 6 of the Capstone Project Brief:
    Returns the order's status, order_value_inr, and a designed escalation_score in [0, 1]
    combining delayed_shipment with a normalized recency signal derived from days_since_created.
    """
    order = get_order_by_id(record_id, orders=orders)
    if not order:
        return {
            "error": f"Order {record_id} not found",
            "status": "Unknown",
            "order_value_inr": 0,
            "escalation_score": 0.0,
            "escalate_recommended": False,
        }

    # Normalized recency signal: 0 to 30 days mapped to [0.0, 1.0]
    recency_norm = min(1.0, max(0.0, order.days_since_created / 30.0))
    delay_signal = 1.0 if order.delayed_shipment else 0.0

    # Designed escalation formula: 60% delay signal + 40% recency signal
    escalation_score = round(0.60 * delay_signal + 0.40 * recency_norm, 4)

    # Escalation recommended threshold: 0.72 corresponds to delayed orders (0.60 base) with days_since_created >= 9 days (0.40 * 9/30 = 0.12)
    escalate_recommended = bool(escalation_score >= 0.72)

    return {
        "record_id": order.record_id,
        "status": order.status,
        "order_value_inr": order.order_value_inr,
        "days_since_created": order.days_since_created,
        "delayed_shipment": order.delayed_shipment,
        "escalation_score": escalation_score,
        "escalate_recommended": escalate_recommended,
    }


# Authoritative module-level ORDERS list generated with DEFAULT_SEED
ORDERS: List[OrderRecord] = generate_orders(seed=DEFAULT_SEED, count=DEFAULT_ORDER_COUNT)
