"""
FastAPI Server and WebSocket Endpoints for Nykaa Support Agent (Task 11 & Task 12).

Strictly complies with Capstone Project Brief (Pages 8-9):
1. Exposes >=2 HTTP endpoints:
   - POST /ask (Pydantic AskRequest -> AskResponse)
   - POST /add-document (Pydantic AddDocumentRequest -> AddDocumentResponse)
   - GET /health (Health check)
2. Exposes 1 WebSocket endpoint:
   - WS /ws/chat (real-time multi-turn conversational chat)
   - Catches WebSocketDisconnect gracefully without crashing the server for other clients
3. Integrates with:
   - Runtime token/cost budget cap (blocks oversized requests before expensive ops)
   - Input guardrails (PII masking and prompt injection detection)
   - CrewAI 3-agent crew with session memory
   - AutoGen 2-agent compliance review stage
   - In-memory normalized query response caching
   - ELK-compatible structured JSON-Lines request logging (Task 12)
"""

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from nykaa_support.api.schemas import (
    AddDocumentRequest,
    AddDocumentResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
)
from nykaa_support.autogen_review.review_stage import NykaaReviewStage
from nykaa_support.crew.memory import NykaaSessionMemoryManager
from nykaa_support.crew.support_crew import NykaaSupportCrew
from nykaa_support.governance.cache import NormalizedQueryCache
from nykaa_support.governance.models import BudgetExceededError
from nykaa_support.governance.runtime_budget import RuntimeBudgetGuard
from nykaa_support.guardrails.input_guardrails import inspect_input, mask_pii_for_input
from nykaa_support.observability.logger import audit_logger
from nykaa_support.rag.store import COLLECTION_SENTENCE_BASED, RAGStore
from dataset import check_order_status
from nykaa_support.crew.schemas import format_crew_response
from nykaa_support.mock_llm.engine import extract_order_id


def create_app(
    crew: Optional[NykaaSupportCrew] = None,
    review_stage: Optional[NykaaReviewStage] = None,
    rag_store: Optional[RAGStore] = None,
    memory_manager: Optional[NykaaSessionMemoryManager] = None,
    budget_guard: Optional[RuntimeBudgetGuard] = None,
    query_cache: Optional[NormalizedQueryCache] = None,
) -> FastAPI:
    """Application factory for Nykaa Support FastAPI service."""

    app = FastAPI(
        title="Nykaa Domain Support Agent API",
        description="FastAPI endpoints and WebSocket chat for Nykaa Customer Support multi-agent system",
        version="1.0.0"
    )

    # Initialize shared components
    active_rag_store = rag_store or RAGStore(local_files_only=True)
    # Ensure policy documents are ingested if collection is empty
    if active_rag_store.col_sentence.count() == 0:
        active_rag_store.ingest_documents()

    active_crew = crew or NykaaSupportCrew()
    active_review_stage = review_stage or NykaaReviewStage()
    active_memory_mgr = memory_manager or NykaaSessionMemoryManager()
    active_budget_guard = budget_guard or RuntimeBudgetGuard()
    active_cache = query_cache or NormalizedQueryCache()

    # Store components on app.state
    app.state.rag_store = active_rag_store
    app.state.crew = active_crew
    app.state.review_stage = active_review_stage
    app.state.memory_mgr = active_memory_mgr
    app.state.budget_guard = active_budget_guard
    app.state.query_cache = active_cache

    def process_support_inquiry(
        raw_query: str,
        session_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> AskResponse:
        """
        Shared pipeline for HTTP and WebSocket requests:
        1. Runtime budget evaluation (Task 15)
        2. Input PII masking & injection check (Task 10)
        3. Response caching lookup (Task 16)
        4. Multi-agent crew execution with session memory (Tasks 7 & 8)
        5. AutoGen compliance review stage (Task 14)
        6. Structured JSON-Lines logging (Task 12)
        """
        req_trace_id = trace_id or str(uuid.uuid4())
        active_session = session_id or f"session-{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Step 1: Runtime budget guard
        try:
            active_budget_guard.enforce_request_budget(raw_query)
        except BudgetExceededError as budget_err:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            audit_logger.log_request(
                endpoint="POST /ask",
                query=raw_query,
                final_answer="Request blocked by runtime budget cap.",
                duration_ms=duration_ms,
                status_code=400,
                trace_id=req_trace_id,
                session_id=active_session,
                error=str(budget_err)
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(budget_err)
            )

        # Step 2: Input guardrails (PII masking & prompt injection)
        guard_res = inspect_input(raw_query)
        masked_query = guard_res.sanitized_text

        if not guard_res.passed:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            blocked_msg = "Security Alert: Query contains adversarial instructions or prompt injection patterns and cannot be processed."
            audit_logger.log_request(
                endpoint="POST /ask",
                query=masked_query,
                final_answer=blocked_msg,
                duration_ms=duration_ms,
                status_code=200,
                trace_id=req_trace_id,
                session_id=active_session,
                query_type="injection_blocked",
                error=guard_res.blocked_reason
            )
            return AskResponse(
                query=masked_query,
                final_answer=blocked_msg,
                query_type="injection_blocked",
                order_id=None,
                order_data=None,
                retrieved_policy_docs=[],
                escalation_score=0.0,
                escalation_recommended=False,
                review_verdict={"approved": False, "reason": guard_res.blocked_reason or "Prompt injection blocked", "final_answer": blocked_msg},
                session_id=active_session,
                trace_id=req_trace_id
            )

        # Step 3: Check response cache (Task 16)
        cached_response = active_cache.get(masked_query)
        if cached_response is not None and isinstance(cached_response, AskResponse):
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            audit_logger.log_request(
                endpoint="POST /ask",
                query=masked_query,
                final_answer=cached_response.final_answer,
                duration_ms=duration_ms,
                status_code=200,
                trace_id=req_trace_id,
                session_id=active_session,
                query_type=cached_response.query_type,
                order_id=cached_response.order_id,
                escalation_score=cached_response.escalation_score,
                escalation_recommended=cached_response.escalation_recommended,
                governance_info={"cache_hit": True}
            )
            # Return copy with updated trace and session
            return AskResponse(
                query=masked_query,
                final_answer=cached_response.final_answer,
                query_type=cached_response.query_type,
                order_id=cached_response.order_id,
                order_data=cached_response.order_data,
                retrieved_policy_docs=cached_response.retrieved_policy_docs,
                escalation_score=cached_response.escalation_score,
                escalation_recommended=cached_response.escalation_recommended,
                review_verdict=cached_response.review_verdict,
                session_id=active_session,
                trace_id=req_trace_id
            )

        # Step 4: Multi-agent crew execution with session memory (Tasks 7, 8, 9)
        crew_resp_text = active_memory_mgr.invoke(active_session, masked_query)

        # Extract order id and query context
        extracted_order_id = extract_order_id(masked_query)
        if not extracted_order_id:
            for msg in reversed(active_memory_mgr.get_messages(active_session)):
                found = extract_order_id(str(msg.content))
                if found:
                    extracted_order_id = found
                    break

        order_info = None
        if extracted_order_id:
            order_info = check_order_status(extracted_order_id)

        # Policy docs retrieval
        retrieved_docs = []
        q_results = active_rag_store.query(COLLECTION_SENTENCE_BASED, masked_query, top_k=2)
        if q_results and any(r["similarity"] >= 0.35 for r in q_results):
            retrieved_docs = list(set(r["topic_title"] for r in q_results if r["similarity"] >= 0.35))

        crew_resp = format_crew_response(
            query=masked_query,
            final_answer=crew_resp_text,
            order_info=order_info,
            policy_docs=retrieved_docs
        )

        # Step 5: AutoGen compliance review stage (Task 14)
        context_str = " ".join(retrieved_docs) if retrieved_docs else None
        verdict = active_review_stage.review_draft_sync(crew_resp.final_answer, context=context_str)

        final_customer_answer = verdict.final_answer if verdict.final_answer else crew_resp.final_answer
        order_data_dict = crew_resp.order_data.model_dump() if crew_resp.order_data else None

        response_obj = AskResponse(
            query=masked_query,
            final_answer=final_customer_answer,
            query_type=crew_resp.query_type,
            order_id=crew_resp.order_id,
            order_data=order_data_dict,
            retrieved_policy_docs=crew_resp.retrieved_policy_docs,
            escalation_score=crew_resp.escalation_score or 0.0,
            escalation_recommended=crew_resp.escalation_recommended,
            review_verdict=verdict.model_dump(),
            session_id=active_session,
            trace_id=req_trace_id
        )

        # Store in query cache
        active_cache.set(masked_query, response_obj)

        # Step 6: Structured JSONL logging (Task 12)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        audit_logger.log_request(
            endpoint="POST /ask",
            query=masked_query,
            final_answer=final_customer_answer,
            duration_ms=duration_ms,
            status_code=200,
            trace_id=req_trace_id,
            session_id=active_session,
            query_type=crew_resp.query_type,
            order_id=crew_resp.order_id,
            escalation_score=crew_resp.escalation_score,
            escalation_recommended=crew_resp.escalation_recommended,
            governance_info={"cache_hit": False, "verdict_approved": verdict.approved}
        )

        return response_obj

    # -----------------------------------------------------------------
    # HTTP Endpoint 1: GET /health
    # -----------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse()

    # -----------------------------------------------------------------
    # HTTP Endpoint 2: POST /ask
    # -----------------------------------------------------------------
    @app.post("/ask", response_model=AskResponse)
    def ask_endpoint(payload: AskRequest):
        """
        Primary customer inquiry endpoint.
        Accepts questions, validates against budget/guardrails, runs CrewAI + AutoGen,
        and returns structured response.
        """
        try:
            return process_support_inquiry(
                raw_query=payload.query,
                session_id=payload.session_id
            )
        except HTTPException:
            raise
        except Exception as e:
            trace_id = str(uuid.uuid4())
            audit_logger.log_request(
                endpoint="POST /ask",
                query=payload.query,
                duration_ms=0.0,
                status_code=500,
                trace_id=trace_id,
                error=str(e)
            )
            # Never leak unmasked PII or raw internal stack trace
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred processing your support request (Trace ID: {trace_id})."
            )

    # -----------------------------------------------------------------
    # HTTP Endpoint 3: POST /add-document
    # -----------------------------------------------------------------
    @app.post("/add-document", response_model=AddDocumentResponse)
    def add_document_endpoint(payload: AddDocumentRequest):
        """
        Dynamic knowledge base ingestion endpoint.
        Adds and chunks new policy documentation into the sentence-based ChromaDB vector store.
        """
        start_time = time.perf_counter()
        trace_id = str(uuid.uuid4())

        clean_filename = payload.filename.strip()
        clean_content = payload.content.strip()

        if len(clean_content) < 10:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Document content must contain at least 10 characters."
            )

        try:
            chunks_count = active_rag_store.add_document_content(
                filename=clean_filename,
                content=clean_content,
                topic_title=payload.category
            )

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            audit_logger.log_request(
                endpoint="POST /add-document",
                query=f"Ingest document: {clean_filename}",
                final_answer=f"Successfully indexed {chunks_count} chunks into {COLLECTION_SENTENCE_BASED}.",
                duration_ms=duration_ms,
                status_code=200,
                trace_id=trace_id,
                extra={"filename": clean_filename, "chunks_added": chunks_count}
            )

            return AddDocumentResponse(
                status="success",
                filename=clean_filename,
                chunks_added=chunks_count,
                collection_name=COLLECTION_SENTENCE_BASED,
                trace_id=trace_id
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            audit_logger.log_request(
                endpoint="POST /add-document",
                query=f"Ingest document: {clean_filename}",
                duration_ms=duration_ms,
                status_code=500,
                trace_id=trace_id,
                error=str(e)
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to ingest document '{clean_filename}': {str(e)}"
            )

    # -----------------------------------------------------------------
    # WebSocket Endpoint: WS /ws/chat
    # -----------------------------------------------------------------
    @app.websocket("/ws/chat")
    async def websocket_chat_endpoint(websocket: WebSocket):
        """
        Real-time multi-turn conversational chat over WebSocket.
        Gracefully handles WebSocketDisconnect without crashing the server.
        """
        await websocket.accept()
        ws_session_id = f"ws-session-{uuid.uuid4().hex[:8]}"
        ws_trace_id = str(uuid.uuid4())

        audit_logger.log_request(
            endpoint="WS /ws/chat",
            query="[WEBSOCKET_CONNECTED]",
            final_answer="Connection established",
            status_code=101,
            trace_id=ws_trace_id,
            session_id=ws_session_id
        )

        try:
            while True:
                # Receive message frame
                try:
                    data = await websocket.receive_text()
                except WebSocketDisconnect:
                    # Catch WebSocketDisconnect gracefully as strictly required by brief
                    audit_logger.log_request(
                        endpoint="WS /ws/chat",
                        query="[CLIENT_DISCONNECTED]",
                        final_answer="Client closed connection cleanly",
                        status_code=1000,
                        trace_id=ws_trace_id,
                        session_id=ws_session_id
                    )
                    break

                # Parse incoming text (support JSON format or raw text)
                incoming_query = data.strip()
                incoming_session = ws_session_id

                if data.strip().startswith("{") and data.strip().endswith("}"):
                    import json
                    try:
                        parsed = json.loads(data)
                        if isinstance(parsed, dict) and "query" in parsed:
                            incoming_query = str(parsed["query"])
                            incoming_session = str(parsed.get("session_id", ws_session_id))
                    except json.JSONDecodeError:
                        pass

                if not incoming_query:
                    await websocket.send_json({
                        "error": "Query cannot be empty.",
                        "session_id": incoming_session
                    })
                    continue

                # Process through support pipeline in worker thread to prevent event loop collision
                try:
                    resp = await asyncio.to_thread(
                        process_support_inquiry,
                        raw_query=incoming_query,
                        session_id=incoming_session,
                        trace_id=str(uuid.uuid4())
                    )
                    await websocket.send_text(resp.model_dump_json())
                except HTTPException as http_err:
                    await websocket.send_json({
                        "error": http_err.detail,
                        "status_code": http_err.status_code,
                        "session_id": incoming_session
                    })
                except Exception as ex:
                    audit_logger.log_request(
                        endpoint="WS /ws/chat",
                        query=incoming_query,
                        status_code=500,
                        session_id=incoming_session,
                        error=str(ex)
                    )
                    await websocket.send_json({
                        "error": f"Internal error processing chat turn: {str(ex)}",
                        "session_id": incoming_session
                    })

        except WebSocketDisconnect:
            # Outer safety catch
            audit_logger.log_request(
                endpoint="WS /ws/chat",
                query="[CLIENT_DISCONNECTED_OUTER]",
                final_answer="Client disconnected",
                status_code=1000,
                trace_id=ws_trace_id,
                session_id=ws_session_id
            )
        except Exception as ws_err:
            audit_logger.log_request(
                endpoint="WS /ws/chat",
                query="[WEBSOCKET_ERROR]",
                status_code=500,
                trace_id=ws_trace_id,
                session_id=ws_session_id,
                error=str(ws_err)
            )

    return app


# Default ASGI app instance for uvicorn
app = create_app()
