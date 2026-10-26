"""
Input Guardrails for Nykaa Support Agent (Task 10).

Strictly complies with Capstone Project Brief (Pages 4 and 6):
1. Fixed-Format PII Masking:
   - Phone numbers: Indian formats (+91-XXXXXXXXXX, 10-digit mobile) -> [PHONE_MASKED]
   - Payment cards: 16-digit card numbers & last-4-digit references -> [CARD_MASKED]
   - Scope exclusion: Customer name and delivery address are free text with no reliable
     pattern to match under a keyless, MOCK_LLM masker, and are explicitly acknowledged
     as out of scope by the brief.
2. Prompt-Injection Detection:
   - Blocks jailbreaks, instruction overrides, prompt leaks, and role hijacks.
3. Dual-Application:
   - Sanitized text is supplied to both model prompt and structured logging (Task 12).
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field


class InputGuardrailResult(BaseModel):
    """Structured result of input guardrail inspection and sanitization."""
    original_text: str = Field(description="Raw original user query")
    sanitized_text: str = Field(description="PII-masked query safe for model and disk logs")
    pii_detected: bool = Field(default=False, description="Whether fixed-format PII was detected")
    pii_types: List[str] = Field(default_factory=list, description="List of detected PII types ('phone', 'card')")
    is_injection_attempt: bool = Field(default=False, description="Whether prompt injection attack was detected")
    injection_pattern: Optional[str] = Field(default=None, description="Matched injection trigger phrase")
    passed: bool = Field(default=True, description="False if query is blocked (e.g. injection); True if safe to process")
    blocked_reason: Optional[str] = Field(default=None, description="Explanation if query was rejected")


# Regex patterns for fixed-format PII
# 1. Indian Phone Numbers: +91-XXXXXXXXXX, +91 XXXXXXXXXX, 0XXXXXXXXXX, 10-digit mobile [6-9]XXXXXXXXX
PHONE_PATTERNS = [
    re.compile(r'(?:\+91[\s\-]?)?[6-9]\d{4}[\s\-]?\d{5}\b'),
    re.compile(r'\b0[6-9]\d{9}\b'),
]

# 2. Payment Cards: 16 digits (with optional spaces/dashes), or card ending in XXXX
CARD_FULL_PATTERN = re.compile(r'\b(?:\d{4}[\s\-]?){3}\d{4}\b')
CARD_LAST4_PATTERNS = [
    re.compile(r'(?:card\s*(?:ending\s*(?:in|with)|last\s*4|#|\*{3,8})\s*[:\s]*?)(\d{4})\b', re.IGNORECASE),
    re.compile(r'\b(?:\*{4}[\s\-]?){3}(\d{4})\b'),
]

# Prompt-injection trigger patterns
INJECTION_PATTERNS = [
    r'ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions',
    r'system\s+override',
    r'override\s+(?:system\s+)?prompt',
    r'jailbreak',
    r'dan\s+mode',
    r'developer\s+mode\s+enabled',
    r'you\s+are\s+now\s+(?:an?\s+)?unrestricted',
    r'reveal\s+(?:your\s+)?(?:system\s+)?prompt',
    r'print\s+(?:your\s+)?instructions',
    r'leak\s+(?:the\s+)?system\s+prompt',
    r'disregard\s+(?:all\s+)?safety\s+guidelines',
    r'bypass\s+(?:all\s+)?filters',
    r'ignore\s+nykaa\s+policies',
]


def mask_pii(text: str) -> tuple[str, bool, List[str]]:
    """
    Mask fixed-format phone numbers and payment card info.
    Returns (sanitized_text, pii_detected, pii_types).
    """
    sanitized = text
    detected_types = []

    # 1. Mask full 16-digit payment card numbers
    if CARD_FULL_PATTERN.search(sanitized):
        sanitized = CARD_FULL_PATTERN.sub("[CARD_MASKED]", sanitized)
        detected_types.append("payment_card_full")

    # 2. Mask card last-4 references
    for pattern in CARD_LAST4_PATTERNS:
        if pattern.search(sanitized):
            sanitized = pattern.sub("card ending in [CARD_MASKED]", sanitized)
            if "payment_card_last4" not in detected_types:
                detected_types.append("payment_card_last4")

    # 3. Mask phone numbers
    for pattern in PHONE_PATTERNS:
        # Check that we don't accidentally mask already masked tokens or order IDs (e.g. NYK-10001)
        matches = pattern.finditer(sanitized)
        for m in matches:
            val = m.group(0)
            # Avoid masking order IDs or small numbers
            if not val.startswith("NYK") and not val.startswith("ORD"):
                sanitized = sanitized.replace(val, "[PHONE_MASKED]")
                if "phone_number" not in detected_types:
                    detected_types.append("phone_number")

    pii_found = len(detected_types) > 0
    return sanitized, pii_found, detected_types


def detect_prompt_injection(text: str) -> tuple[bool, Optional[str]]:
    """
    Detect adversarial prompt injection attacks.
    Returns (is_injection, matched_pattern).
    """
    for pattern_str in INJECTION_PATTERNS:
        match = re.search(pattern_str, text, re.IGNORECASE)
        if match:
            return True, match.group(0)
    return False, None


def apply_input_guardrails(raw_query: str) -> InputGuardrailResult:
    """
    Run comprehensive input guardrails on customer query:
    1. Check for prompt injection attacks.
    2. Apply fixed-format PII masking for phone numbers and cards.
    """
    # 1. Prompt Injection Check
    is_injection, injection_match = detect_prompt_injection(raw_query)
    if is_injection:
        return InputGuardrailResult(
            original_text=raw_query,
            sanitized_text="[BLOCKED: PROMPT INJECTION ATTEMPT DETECTED]",
            pii_detected=False,
            pii_types=[],
            is_injection_attempt=True,
            injection_pattern=injection_match,
            passed=False,
            blocked_reason=f"Security alert: Prompt injection trigger detected ('{injection_match}'). Query blocked."
        )

    # 2. Fixed-format PII Masking
    sanitized_text, pii_found, pii_types = mask_pii(raw_query)

    return InputGuardrailResult(
        original_text=raw_query,
        sanitized_text=sanitized_text,
        pii_detected=pii_found,
        pii_types=pii_types,
        is_injection_attempt=False,
        injection_pattern=None,
        passed=True,
        blocked_reason=None
    )


# Convenient aliases for uniform naming across modules
mask_pii_for_input = mask_pii
inspect_input = apply_input_guardrails
