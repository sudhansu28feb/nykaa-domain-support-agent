"""
Nykaa Support Observability Package (Task 12).
"""

from nykaa_support.observability.logger import (
    DEFAULT_LOG_FILE,
    StructuredAuditLogger,
    audit_logger,
)

__all__ = [
    "StructuredAuditLogger",
    "audit_logger",
    "DEFAULT_LOG_FILE",
]
