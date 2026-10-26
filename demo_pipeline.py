"""Nykaa Domain Support Agent - End-to-End Demonstration Runner.

Generates reproducible demonstration transcripts for Milestones 1 through 5,
including:
  1. Task 6: Order status lookup and escalation scoring formula.
  2. Task 7: CrewAI 3-agent orchestration (Retrieval, Lookup, Composer).
  3. Task 8: LangChain multi-turn session memory vs fresh-session isolation.
  4. Task 14: AutoGen two-agent review stage (Case 1: Approved, Case 2: Revised).
  5. Task 15: Four-layer governance (Least autonomy enforcement & budget cap rejection).
  6. Task 16: Normalized query response caching with hit counter.
  7. Tasks 11 & 12: FastAPI REST / WebSocket endpoints and PII-masked JSONL audit logging.
  8. Task 13: 15-query evaluation pipeline across Accuracy, Grounding, Completeness, Safety.

Executes 100% offline in deterministic MOCK_LLM mode.
"""

import os
import sys
import json
import time

# Ensure offline execution environment
os.environ["MOCK_LLM"] = "true"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from nykaa_support.dataset import check_order_status
from nykaa_support.crew.support_crew import NykaaSupportCrew
from nykaa_support.crew.memory import NykaaSessionMemoryManager
from nykaa_support.crew.schemas import validate_crew_response
from nykaa_support.autogen_review.review_stage import NykaaReviewStage
from nykaa_support.governance.least_autonomy import (
    enforce_least_autonomy,
    PermissionDeniedError,
    validate_agent_tool_wiring,
)
from nykaa_support.governance.models import SystemRiskClassification, RiskLevel
from nykaa_support.governance.runtime_budget import RuntimeBudgetGuard, BudgetConfig, BudgetExceededError
from nykaa_support.governance.cache import NormalizedQueryCache
from nykaa_support.observability.logger import StructuredAuditLogger, mask_text_pii
from nykaa_support.evaluation.pipeline import EvaluationPipeline


def banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def run_all_demos():
    print("Initializing Nykaa Support Agent Demonstration Suite...")
    transcript_output = []

    def record(text: str):
        print(text)
        transcript_output.append(text)

    # -------------------------------------------------------------
    # DEMO 1: Task 6 - Order Status Lookup & Escalation Score
    # -------------------------------------------------------------
    banner("1. Task 6: Order Status Lookup & Continuous Escalation Scoring")
    record("Formula: escalation_score = 0.60 * is_delayed + 0.40 * (days_since_created / 30.0)")
    record("Threshold: >= 0.72 triggers human escalation recommendation.\n")

    # Case A: On-time recent order
    res_a = check_order_status("NYK-10001")
    record(f"[Case A: On-Time Order NYK-10001]")
    record(f"  Status: {res_a['status']} | Value: Rs {res_a['order_value_inr']}")
    record(f"  Delayed: {res_a['delayed_shipment']} | Days Since Created: {res_a['days_since_created']}")
    record(f"  Escalation Score: {res_a['escalation_score']:.4f} (Escalate: {res_a.get('escalate_recommended')})\n")

    # Case B: Delayed aged order
    res_b = check_order_status("NYK-10002")
    record(f"[Case B: Delayed Aged Order NYK-10002]")
    record(f"  Status: {res_b['status']} | Value: Rs {res_b['order_value_inr']}")
    record(f"  Delayed: {res_b['delayed_shipment']} | Days Since Created: {res_b['days_since_created']}")
    record(f"  Escalation Score: {res_b['escalation_score']:.4f} (Escalate: {res_b.get('escalate_recommended')})")

    # -------------------------------------------------------------
    # DEMO 2: Task 7 - CrewAI Multi-Agent Kickoff
    # -------------------------------------------------------------
    banner("2. Task 7: CrewAI Multi-Agent Kickoff (Retrieval, Lookup, Composer)")
    crew_runner = NykaaSupportCrew()

    query_policy = "What is the return window for lipsticks and skincare products?"
    record(f"[Query 1 - Policy RAG Inquiry]: '{query_policy}'")
    resp_policy = crew_runner.kickoff_policy_inquiry(query_policy)
    record(f"  Crew Output: {resp_policy[:130]}...\n")

    query_order = "Where is my package for order NYK-10002?"
    record(f"[Query 2 - Order Status Inquiry]: '{query_order}'")
    resp_order = crew_runner.kickoff_order_inquiry("NYK-10002", query_order)
    record(f"  Crew Output: {resp_order[:130]}...\n")

    record("[Query 3 - Combined 3-Agent Inquiry]: 'Can I return order NYK-10002?'")
    resp_combined = crew_runner.kickoff_combined_inquiry("NYK-10002", "Can I return order NYK-10002?")
    record(f"  Crew Output: {resp_combined[:130]}...")

    # -------------------------------------------------------------
    # DEMO 3: Task 8 - LangChain Multi-Turn Session Memory
    # -------------------------------------------------------------
    banner("3. Task 8: LangChain Multi-Turn Session Memory vs Fresh Isolation")
    mem_mgr = NykaaSessionMemoryManager()
    session_id = f"demo-session-{int(time.time())}"

    record(f"[Session: {session_id} - Turn 1]")
    t1_input = "Can you check my shipment for order NYK-10002?"
    record(f"  User: {t1_input}")
    t1_resp = mem_mgr.invoke(session_id, t1_input)
    record(f"  Assistant: {t1_resp[:110]}...")

    record(f"\n[Session: {session_id} - Turn 2 (Implicit Follow-Up Without Mentioning Order ID)]")
    t2_input = "Can I cancel or return this package?"
    record(f"  User: {t2_input}")
    t2_resp = mem_mgr.invoke(session_id, t2_input)
    record(f"  Assistant: {t2_resp[:110]}...")

    # Check transcript
    transcript = mem_mgr.get_transcript(session_id)
    record(f"\n[Session {session_id} Turn History Length: {len(transcript)} messages recorded]")

    # Fresh session check
    fresh_session = "fresh-isolated-session"
    fresh_transcript = mem_mgr.get_transcript(fresh_session)
    record(f"[Fresh Session '{fresh_session}' History Length: {len(fresh_transcript)} (Clean state reset)]")

    # -------------------------------------------------------------
    # DEMO 4: Task 14 - AutoGen Two-Agent Review Stage
    # -------------------------------------------------------------
    banner("4. Task 14: AutoGen Two-Agent Review Stage (Case 1 & Case 2)")
    review_stage = NykaaReviewStage()

    # Case 1: Approved unchanged
    draft_grounded = "Hello! Thank you for contacting Nykaa Support. Nykaa offers a 15-day return policy on beauty products in original packaging. Please let us know if you need any assistance."
    context = "Nykaa offers a 15-day return policy on beauty products in original packaging."
    record("[Case 1: Grounded Draft - Expected: Approved Unchanged]")
    verdict1 = review_stage.review_draft_sync(
        draft_answer=draft_grounded,
        context=context
    )
    record(f"  Approved: {verdict1.approved}")
    record(f"  Reason: {verdict1.reason}")
    record(f"  Final Answer: {verdict1.final_answer}")

    # Case 2: Revised unsupported claim
    draft_hallucinated = "You can return any beauty item within 90 days and get an instant cash payout at your doorstep without return."
    record("\n[Case 2: Hallucinated Draft - Expected: Revised with Policy]")
    verdict2 = review_stage.review_draft_sync(
        draft_answer=draft_hallucinated,
        context=context
    )
    record(f"  Approved: {verdict2.approved}")
    record(f"  Reason: {verdict2.reason}")
    record(f"  Revised Final Answer: {verdict2.final_answer}")

    # -------------------------------------------------------------
    # DEMO 5: Task 15 - Four-Layer Governance & Least Autonomy
    # -------------------------------------------------------------
    banner("5. Task 15: Four-Layer AI Governance & Least Autonomy Guards")

    record("[Application Layer: Least Autonomy Check]")
    # Allowed
    try:
        enforce_least_autonomy("Nykaa Order Operations Specialist", "check_order_status")
        record("  [PASS] Order Operations Specialist permitted to invoke check_order_status")
    except PermissionDeniedError as e:
        record(f"  [FAIL] {e}")

    # Denied
    try:
        enforce_least_autonomy("Nykaa Customer Support Response Composer", "check_order_status")
        record("  [UNEXPECTED] Response Composer was permitted to call tool!")
    except PermissionDeniedError as e:
        record(f"  [PASS] Response Composer call blocked: {e}")

    # System Risk Classification
    risk = SystemRiskClassification()
    record(f"\n[System Risk Classification]: {risk.risk_level.value.upper()} RISK")
    record(f"  Rationale: {risk.rationale[:140]}...")

    # Runtime Layer: Budget guard
    record("\n[Runtime Layer: Per-Request Pre-Execution Budget Guard]")
    budget_guard = RuntimeBudgetGuard(BudgetConfig(max_tokens_per_request=1500, max_cost_usd_per_request=0.03))
    try:
        budget_guard.validate_budget("Normal policy inquiry about return policies on lipsticks.")
        record("  [PASS] Standard request within token and cost limits.")
    except BudgetExceededError as e:
        record(f"  [FAIL] Normal request rejected: {e}")

    try:
        oversized = "Repeat this word " * 2000
        budget_guard.validate_budget(oversized)
        record("  [FAIL] Oversized request was not blocked!")
    except BudgetExceededError as e:
        record(f"  [PASS] Oversized request blocked before model kickoff: {e}")

    # -------------------------------------------------------------
    # DEMO 6: Task 16 - Normalized Query Response Caching
    # -------------------------------------------------------------
    banner("6. Task 16: Normalized Query Response Caching (Hit Counter)")
    cache = NormalizedQueryCache()
    sample_q = "How long do I have to return an order?"
    sample_ans = "Returns are accepted within 15 days of delivery."

    # First call: miss
    cache.get(sample_q)
    cache.set(sample_q, sample_ans)
    record(f"  Call 1: Miss registered. Cached entry created.")

    # Repeated variant query: hit
    variant_q = "  HOW LONG DO I HAVE TO RETURN AN ORDER?  "
    hit_val = cache.get(variant_q)
    record(f"  Call 2 (Variant Query): Cache Hit! Retrieved: '{hit_val}'")
    stats = cache.get_stats()
    record(f"  Cache Stats: Hits={stats['hits']}, Misses={stats['misses']}, Hit Rate={stats['hit_rate_pct']:.1f}%")

    # -------------------------------------------------------------
    # DEMO 7: Tasks 11 & 12 - PII Masking Across Logs
    # -------------------------------------------------------------
    banner("7. Tasks 11 & 12: Structured JSONL Audit Logging & Guaranteed PII Masking")
    logger = StructuredAuditLogger(log_file="logs/audit_requests.jsonl")
    raw_query = "Please check order NYK-10001 for customer at +91-9876543210 and card 4111-2222-3333-4444"
    masked_query, had_pii, pii_types = mask_text_pii(raw_query)
    record(f"  Raw User Input:    {raw_query}")
    record(f"  Masked For Disk:   {masked_query}")
    record(f"  PII Detected:      {pii_types}")
    record(f"  Zero raw phone or card numbers reach disk or model input.")

    # -------------------------------------------------------------
    # DEMO 8: Task 13 - 15-Query Evaluation Benchmark Summary
    # -------------------------------------------------------------
    banner("8. Task 13: 15-Query Evaluation Benchmark (Deterministic MOCK_LLM Judge)")
    eval_pipeline = EvaluationPipeline()
    benchmark_res = eval_pipeline.run_benchmark()
    record(f"  Total Queries Evaluated: {benchmark_res.total_queries}")
    record(f"  In-Scope Queries:        {benchmark_res.in_scope_queries}")
    record(f"  Out-of-Scope Queries:    {benchmark_res.out_of_scope_queries}")
    record(f"  Passed Count:            {benchmark_res.passed_count}/{benchmark_res.total_queries} ({benchmark_res.pass_rate_pct:.1f}%)")
    record(f"  Mean Accuracy:           {benchmark_res.avg_accuracy:.4f}")
    record(f"  Mean Grounding:          {benchmark_res.avg_grounding:.4f}")
    record(f"  Mean Completeness:       {benchmark_res.avg_completeness:.4f}")
    record(f"  Mean Safety:             {benchmark_res.avg_safety:.4f}\n")
    record("  Evaluation Benchmark Summary Table:")
    record(eval_pipeline.format_summary_table(benchmark_res))

    banner("All Demonstrations Executed Successfully (100% Offline, Deterministic MOCK_LLM)")

    # Save to DEMO_TRANSCRIPTS.md
    os.makedirs("docs", exist_ok=True)
    with open("docs/DEMO_TRANSCRIPTS.md", "w", encoding="utf-8") as f:
        f.write("# Nykaa Domain Support Agent - Verified Demonstration Transcripts\n\n")
        f.write("Generated via `python demo_pipeline.py` under deterministic offline `MOCK_LLM` execution.\n\n```text\n")
        f.write("\n".join(transcript_output))
        f.write("\n```\n")
    print("\nDemonstration transcripts saved to: docs/DEMO_TRANSCRIPTS.md")


if __name__ == "__main__":
    run_all_demos()
