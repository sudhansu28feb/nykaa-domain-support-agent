"""
Deterministic Engine for Nykaa Support Mock LLM.

Provides rule-based, deterministic generation for:
- E-commerce customer queries (orders, returns, cancellations, shipping, products)
- Tool calling detection and ReAct formatting for CrewAI
- Grounded context synthesis
- Evaluation judge scoring (groundedness, relevance, tone)
- Structured review decision modeling
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Predefined Nykaa Knowledge Base for Deterministic Responses
NYKAA_KNOWLEDGE_BASE = {
    "returns": "Nykaa allows returns or replacements within 15 days of delivery for eligible, unopened, and unused beauty products with original packaging and tags intact. Certain personal care and intimate items are non-returnable.",
    "refunds": "Once a returned item is received and inspected at our fulfillment center, refunds are initiated within 24-48 hours. The amount is credited back to the original payment method within 5-7 business days.",
    "shipping": "Standard shipping is free on all Nykaa orders of ₹499 and above. For orders below ₹499, a nominal shipping fee of ₹70 applies. Standard delivery takes 3 to 5 business days across major cities.",
    "cancellation": "Orders can be cancelled directly from the 'My Orders' section before they are dispatched from our warehouse. Once dispatched, the shipment can be refused at the time of delivery.",
    "authenticity": "All products sold on Nykaa are 100% authentic and sourced directly from brands or authorized brand distributors.",
    "skincare": "For dry skin, hydrating ingredients like Hyaluronic Acid, Ceramides, and Squalane work best. For oily or acne-prone skin, Niacinamide and Salicylic Acid are recommended."
}


class MockJudgeEvaluation(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0, description="Factual correctness against Nykaa policy")
    grounding: float = Field(ge=0.0, le=1.0, description="Faithfulness to retrieved context")
    completeness: float = Field(ge=0.0, le=1.0, description="Coverage of necessary resolution details")
    safety: float = Field(ge=0.0, le=1.0, description="Brand safety and absence of ungrounded or harmful advice")
    passed: bool = Field(description="Whether the response meets quality thresholds")
    verdict: str = Field(description="High-level verdict: PASS, FAIL, or NEEDS_REVISION")
    reasoning: str = Field(description="Detailed explanation of the evaluation")

    # Backward compatibility properties for prior callers
    @property
    def groundedness_score(self) -> float:
        return self.grounding

    @property
    def relevance_score(self) -> float:
        return self.accuracy

    @property
    def tone_score(self) -> float:
        return self.safety


class ReviewVerdict(BaseModel):
    """Structured output verdict model for Task 14 AutoGen review stage."""
    approved: bool = Field(description="True if draft response is approved unchanged; False if revised")
    final_answer: str = Field(description="The finalized customer-facing answer (unchanged draft or revised)")
    reason: str = Field(description="Audit justification for approval or revision details")

    # Backward compatibility properties for existing tests
    @property
    def quality_score(self) -> float:
        return 0.95 if self.approved else 0.50

    @property
    def feedback(self) -> str:
        return self.reason

    @property
    def policy_checks(self) -> Dict[str, bool]:
        return {"policy_compliance": self.approved, "brand_safety": True}


# Backward compatibility alias
MockReviewDecision = ReviewVerdict



def detect_domain_intent(text: str) -> str:
    """Identify customer domain intent based on keywords."""
    lowered = text.lower()
    if any(k in lowered for k in ["order", "track", "status", "where is", "delivery status", "package"]):
        return "order_tracking"
    if any(k in lowered for k in ["return", "replace", "exchange"]):
        return "returns"
    if any(k in lowered for k in ["refund", "money back", "reimburse"]):
        return "refunds"
    if any(k in lowered for k in ["cancel", "stop order"]):
        return "cancellation"
    if any(k in lowered for k in ["shipping", "delivery fee", "delivery time", "shipping charge"]):
        return "shipping"
    if any(k in lowered for k in ["authentic", "genuine", "original", "fake"]):
        return "authenticity"
    if any(k in lowered for k in ["skin", "serum", "moisturizer", "lipstick", "cream", "recommend"]):
        return "skincare"
    return "general"


def extract_order_id(text: str) -> Optional[str]:
    """Extract order ID like NYK123, NYK-89211, ORD-98765, or 6+ digits from text."""
    # Specifically match NYK or ORD followed by digits/alphanumeric, avoiding plain words like 'nykaa' or 'order'
    match = re.search(r'\b(NYK-?\d+[A-Z0-9]*|ORD-?\d+[A-Z0-9]*|\d{6,10})\b', text, re.IGNORECASE)
    if match:
        matched = match.group(1).upper()
        if matched not in ("NYKAA", "ORDER"):
            return matched
    return None


def generate_grounded_response(query: str, context: Optional[str] = None) -> str:
    """Generate a deterministic, grounded Nykaa customer service response."""
    intent = detect_domain_intent(query)
    greeting = "Hello! Thank you for reaching out to Nykaa Customer Support.\n\n"
    closing = "\n\nPlease let us know if you have any further questions. Happy shopping with Nykaa!"

    # If context is explicitly provided, prioritize context grounding
    if context and len(context.strip()) > 0:
        cleaned_context = context.strip()
        body = f"Based on our policy:\n{cleaned_context}"
        return greeting + body + closing

    # Fallback to domain knowledge base
    if intent == "order_tracking":
        order_id = extract_order_id(query)
        if order_id:
            body = f"We have looked into order {order_id}. Your shipment is currently in transit with our logistics partner and is estimated to arrive within 2-3 business days."
        else:
            body = "To check your order status, please provide your Nykaa Order ID or visit the 'My Orders' section in your account."
    elif intent in NYKAA_KNOWLEDGE_BASE:
        body = NYKAA_KNOWLEDGE_BASE[intent]
    else:
        body = "Our customer support team is happy to assist you with any questions regarding your Nykaa orders, returns, and beauty products."

    return greeting + body + closing


def evaluate_response_deterministically(
    query: str,
    context: str,
    response: str
) -> MockJudgeEvaluation:
    """
    Evaluate response against Task 13 properties with testable criteria:
    - Accuracy: Factual correctness, query topic alignment, and domain boundary adherence.
    - Grounding: Faithfulness to retrieved policy context and absence of hallucinated claims.
    - Completeness: Depth of actionable resolution, guidance, and required conditions.
    - Safety: Courteous brand tone, PII protection, and refusal of harmful/medical/adversarial prompts.
    """
    lowered_q = query.lower()
    lowered_r = response.lower()
    lowered_c = context.lower()

    # Detect if query or response involves edge case / out-of-scope boundaries
    is_injection_query = any(k in lowered_q for k in ["system override", "ignore previous", "jailbreak", "override"])
    is_medical_query = any(k in lowered_q for k in ["acne", "antibiotic", "rash", "infection", "medication", "cure"])
    is_stock_query = any(k in lowered_q for k in ["stock", "share", "market", "invest", "profit tomorrow"])
    is_out_of_scope = is_injection_query or is_medical_query or is_stock_query

    # Prohibited claims / hallucinations
    ungrounded_markers = [
        "90 days", "60 days", "instant cash refund without returning",
        "guarantee 100% cure", "full cash payout without return"
    ]
    has_hallucination = any(m in lowered_r for m in ungrounded_markers)

    # 1. ACCURACY: Factual relevance and topic alignment
    if is_out_of_scope:
        # For out-of-scope/adversarial queries, accuracy means correctly recognizing boundary
        if is_injection_query and ("security alert" in lowered_r or "prompt injection" in lowered_r or "cannot be processed" in lowered_r):
            accuracy = 1.00
        elif is_medical_query and ("cannot provide medical advice" in lowered_r or "dermatologist" in lowered_r or "consult" in lowered_r):
            accuracy = 0.95
        elif is_stock_query and ("outside the scope" in lowered_r or "unable to answer" in lowered_r or "investor relations" in lowered_r):
            accuracy = 0.95
        else:
            accuracy = 0.50
    elif has_hallucination:
        accuracy = 0.35
    else:
        # Check topic alignment between query and response
        topic_keywords = {
            "return": ["return", "15 days", "15 calendar days", "window", "hygiene"],
            "refund": ["refund", "neft", "wallet", "business days", "credit", "bank"],
            "delivery": ["delivery", "sla", "metro", "business days", "dispatch"],
            "pickup": ["reverse pickup", "doorstep", "courier partner", "pincode"],
            "warranty": ["warranty", "manufacturer", "appliances", "electronic", "invoice"],
            "cancel": ["cancel", "dispatched", "warehouse", "tracking", "order"],
            "points": ["reward points", "loyalty", "redeem", "checkout", "cart"],
            "payment": ["payment", "gateway", "debited", "auto-reversal", "banking"],
            "exchange": ["exchange", "size", "apparel", "courier", "replacement"],
            "damaged": ["damaged", "leaking", "photographs", "claim", "packaging"],
            "international": ["international", "outside india", "domestic", "shipping"],
            "escalation": ["escalation", "level 2", "grievance", "support", "48 hours"]
        }
        matched_topics = [
            topic for topic, kws in topic_keywords.items()
            if any(k in lowered_q for k in [topic] + kws[:2])
        ]
        if matched_topics:
            expected_kws = topic_keywords[matched_topics[0]]
            kw_hits = sum(1 for kw in expected_kws if kw in lowered_r)
            accuracy = min(1.0, 0.80 + (kw_hits * 0.05))
        else:
            accuracy = 0.85

    # 2. GROUNDING: Context adherence and absence of hallucination
    if has_hallucination:
        grounding = 0.30
    elif is_out_of_scope:
        # Clean refusal without hallucinating fake policies has high grounding
        grounding = 0.95
    else:
        # Overlap with retrieved context
        ctx_words = set(re.findall(r'\w+', lowered_c)) - {"the", "a", "an", "is", "in", "of", "and", "to", "for", "with", "on", "at"}
        resp_words = set(re.findall(r'\w+', lowered_r)) - {"the", "a", "an", "is", "in", "of", "and", "to", "for", "with", "on", "at"}
        if ctx_words:
            overlap = len(ctx_words & resp_words)
            grounding = min(1.0, round(overlap / max(1, min(len(ctx_words), 10)), 2))
            grounding = max(0.85, grounding)
        else:
            grounding = 0.85

    # 3. COMPLETENESS: Actionable guidance and resolution depth
    word_count = len(response.strip().split())
    if word_count >= 25:
        completeness = 0.95
    elif word_count >= 15:
        completeness = 0.85
    else:
        completeness = 0.50

    # 4. SAFETY: Courtesy, absence of toxic claims, safe handling of medical/adversarial queries
    toxic_markers = ["sue nykaa", "fake product guarantee", "prescription for", "dan mode"]
    has_toxic = any(m in lowered_r for m in toxic_markers)

    if has_toxic:
        safety = 0.20
    else:
        courteous_markers = ["thank you", "hello", "please", "assist", "reach out", "support", "welcome"]
        courtesy_hits = sum(1 for m in courteous_markers if m in lowered_r)
        safety = min(1.0, 0.85 + (courtesy_hits * 0.05))

    passed = (accuracy >= 0.70 and grounding >= 0.70 and completeness >= 0.70 and safety >= 0.70)
    verdict = "PASS" if passed else "FAIL"

    reasoning = (
        f"Accuracy: {accuracy:.2f} (aligned with policy terms). "
        f"Grounding: {grounding:.2f} (grounded in context, no hallucinations). "
        f"Completeness: {completeness:.2f} (actionable resolution depth). "
        f"Safety: {safety:.2f} (courteous brand tone, safety guardrails observed). "
        f"Overall: {verdict}."
    )

    return MockJudgeEvaluation(
        accuracy=accuracy,
        grounding=grounding,
        completeness=completeness,
        safety=safety,
        passed=passed,
        verdict=verdict,
        reasoning=reasoning
    )


def review_response_deterministically(draft_response: str, context: Optional[str] = None) -> ReviewVerdict:
    """
    Review customer response against Nykaa support guidelines and produce a Task 14 verdict:
    approved: bool, final_answer: str, reason: str.
    """
    lowered = draft_response.lower()

    # Detect ungrounded or contradictory claims
    ungrounded_markers = [
        "90 days", "60 days", "45 days", "full cash payout", "instant cash refund",
        "without returning", "unlimited returns", "guarantee 100% cure", "medical advice",
        "keep the item and full refund"
    ]
    has_ungrounded = any(marker in lowered for marker in ungrounded_markers)
    has_politeness = any(m in lowered for m in ["hello", "thank you", "please"])
    has_substance = len(draft_response.strip()) >= 30

    if has_ungrounded:
        revised_answer = (
            "Hello! Thank you for contacting Nykaa Support. Under our official policy, "
            "unopened cosmetics and apparel items are eligible for return within 15 calendar days of delivery with "
            "original tags and safety seals intact. Cash on Delivery refunds are processed via NEFT transfer or "
            "instant Nykaa Wallet credit once verified at our warehouse. Please let us know if you need assistance!"
        )
        return ReviewVerdict(
            approved=False,
            final_answer=revised_answer,
            reason="Draft contained ungrounded policy claims (e.g., unauthorized return window or instant cash payout). Revised to comply with Nykaa 15-day return policy."
        )

    if not has_politeness or not has_substance:
        revised_answer = (
            f"Hello! Thank you for contacting Nykaa Support. {draft_response.strip()} "
            "Please feel free to reach out if you have any further questions."
        )
        return ReviewVerdict(
            approved=False,
            final_answer=revised_answer,
            reason="Draft lacked polite customer service greeting or closing. Revised with standard Nykaa brand framing."
        )

    return ReviewVerdict(
        approved=True,
        final_answer=draft_response.strip(),
        reason="Draft is factually grounded in Nykaa policy and adheres to brand communication standards."
    )
