"""
Structured Request Logging and PII Masking for Nykaa Support Agent (Task 12).

Strictly complies with Capstone Project Brief (Pages 8-9):
1. Emits one JSON-Lines entry per request with a trace ID and timing information.
2. The logged request text must NOT contain the raw, unmasked value of any fixed-format PII field.
3. Applies identical masking to disk logs as model inputs (Indian phone numbers -> [PHONE_MASKED], payment cards -> [CARD_MASKED]).
4. Supports ELK-style structured logging (Elasticsearch, Logstash, Kibana) via valid JSONL format.
"""

import datetime
import json
import os
import threading
from typing import Any, Dict, List, Optional
import uuid

from nykaa_support.guardrails.input_guardrails import mask_pii_for_input

# Alias for external convenience
mask_text_pii = mask_pii_for_input

# Default log file path
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")
DEFAULT_LOG_FILE = os.path.join(DEFAULT_LOG_DIR, "audit_requests.jsonl")


class StructuredAuditLogger:
    """
    ELK-compatible JSON-Lines structured audit logger with guaranteed PII sanitization.
    """

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or DEFAULT_LOG_FILE
        self._lock = threading.Lock()
        self._ensure_log_directory()

    def _ensure_log_directory(self) -> None:
        """Create parent log directory if it does not exist."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

    def log_request(
        self,
        endpoint: str,
        query: str,
        final_answer: Optional[str] = None,
        duration_ms: float = 0.0,
        status_code: int = 200,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        query_type: Optional[str] = None,
        order_id: Optional[str] = None,
        escalation_score: Optional[float] = None,
        escalation_recommended: Optional[bool] = None,
        governance_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Record a structured JSON-Lines entry. Fixed-format PII is strictly masked before writing to disk.
        """
        active_trace_id = trace_id or str(uuid.uuid4())
        timestamp_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Strict PII sanitization: unmasked PII NEVER reaches disk
        masked_query, _, _ = mask_pii_for_input(query)
        masked_answer = None
        if final_answer is not None:
            masked_answer, _, _ = mask_pii_for_input(final_answer)

        entry: Dict[str, Any] = {
            "trace_id": active_trace_id,
            "timestamp": timestamp_iso,
            "endpoint": endpoint,
            "duration_ms": round(duration_ms, 2),
            "status_code": status_code,
            "query": masked_query,
            "final_answer": masked_answer,
            "session_id": session_id,
            "query_type": query_type,
            "order_id": order_id,
            "escalation_score": round(escalation_score, 4) if escalation_score is not None else None,
            "escalation_recommended": escalation_recommended,
            "governance": governance_info or {},
            "error": error,
        }

        if extra:
            entry["extra"] = extra

        serialized = json.dumps(entry, ensure_ascii=False)

        with self._lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(serialized + "\n")

        return entry

    def read_entries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Read and parse emitted JSON-Lines entries from disk."""
        if not os.path.exists(self.log_file):
            return []

        entries = []
        with self._lock:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line_clean = line.strip()
                    if line_clean:
                        try:
                            entries.append(json.loads(line_clean))
                        except json.JSONDecodeError:
                            continue

        if limit is not None:
            return entries[-limit:]
        return entries

    def clear(self) -> None:
        """Clear log file contents (used for isolated test fixtures)."""
        with self._lock:
            if os.path.exists(self.log_file):
                with open(self.log_file, "w", encoding="utf-8") as f:
                    f.write("")


# Global singleton instance
audit_logger = StructuredAuditLogger()
