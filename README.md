# Nykaa Domain Support Agent

A modular, deterministic customer support agent prototype built for the **IIT Roorkee Agentic AI Capstone Project** (E-commerce Track: Nykaa).

> **Academic Disclaimer:** This repository is an educational capstone project developed for the IIT Roorkee Agentic AI program (E-commerce Track: Nykaa). It is a simulated prototype operating in deterministic mock mode and is not an official Nykaa product, nor is it connected to Nykaa's production systems or proprietary corporate APIs. All customer support policies, order records, customer details, and resolution responses derive from synthetic educational fixtures authored for rubric evaluation in [`data/policies/`](data/policies/) and [`data/orders.json`](data/orders.json).

## 1. Project Title and Overview

The **Nykaa Domain Support Agent** implements an intelligent retail customer support workflow combining deterministic retrieval-augmented generation (RAG), tool-augmented operational lookups, session memory, multi-agent compliance review, multi-layer AI governance, and structured observability.

The system addresses customer inquiries regarding product return windows, cash-on-delivery (COD) refund timelines, delivery service-level agreements (SLAs), reverse-pickup logistics, brand warranty terms, order cancellations, and loyalty rewards. In addition to answering policy inquiries, the agent inspects synthetic order records, calculates continuous transit delay and recency escalation scores, remembers multi-turn conversational context, masks customer sensitive data (PII) before disk writes, and subjects all generated draft resolutions to an independent AutoGen compliance review stage before delivering the final response.

In its default deterministic `MOCK_LLM` mode, all capstone capabilities execute locally without external LLM API keys or live outbound network access, assuming the local embedding model weights are cached.

### Dataset Design Choices (Task 1)

As required by the capstone specification for the Nykaa (E-commerce & Retail) track, the dataset generator ([`dataset.py`](dataset.py), [`nykaa_support/dataset.py`](nykaa_support/dataset.py)) produces a seeded, deterministic synthetic order collection reproducible by evaluators:

| Parameter | Configuration / Generator Rule | Actual Sampled Value (`data/orders.json`) | Capstone Brief Mandate |
| :--- | :--- | :--- | :--- |
| **Random Seed** | `DEFAULT_SEED = 42` (`random.Random(42)`) | Deterministic seed `42` | Seeded, reproducible generator |
| **Order Count** | `DEFAULT_ORDER_COUNT = 50` | `50` records | $\ge 40$ records |
| **Product Categories** | 3 guaranteed each via round-robin (15 records) + uniform random choice (`rng.choice`) for remaining 35 | Apparel (12), Electronics (10), Home (10), Beauty (10), Footwear (8) | All 5 categories present ($\ge 3$ each) |
| **Lifecycle Statuses** | 1 guaranteed each (5 records) + weighted draw for remaining 45 (`weights=[0.15, 0.25, 0.40, 0.10, 0.10]`) | Delivered (17), Shipped (13), Refunded (8), Placed (7), Returned (5) | All 5 statuses present ($\ge 1$ each) |
| **Delayed Shipments** | `round(count * 0.20)` via `rng.sample` without replacement | `10 / 50` records (**20.0%**) | Between 10% and 30% |
| **Order Value Range** | Configured catalog bounds: **₹445 to ₹4,899** across 25 products (reasoning: realistic Nykaa retail baskets) | **₹498 to ₹4,882** (Min: `NYK-10040` @ ₹498, Max: `NYK-10041` @ ₹4,882; Total: ₹85,841; AOV: ₹1,716.82) | Realistic price range with 1-sentence reasoning |
| **Recency Window** | `rng.randint(0, 30)` | Min: 0 days, Max: 30 days | Integer between 0 and 30 |

> **Category vs. Status Sampling Note:** Product categories are sampled uniformly (`rng.choice`) after initial round-robin guarantees without artificial category weights. Lifecycle statuses utilize explicit historical weights favoring `Delivered` (0.40) and `Shipped` (0.25) to model realistic post-purchase retail inquiry distributions.

---

## 2. Key Features

- **Deterministic E-Commerce Dataset (Task 1):** 50 reproducible synthetic order records generated via `random.Random(42)` covering 5 categories (Apparel, Electronics, Home, Footwear, Beauty) and 5 lifecycle statuses (Placed, Shipped, Delivered, Returned, Refunded).
- **Authored Policy Knowledge Base (Task 2):** 12 distinct, standalone ground-truth retail policy documents (2–5 sentences each) written in Markdown covering core retail domains.
- **Dual Vector Chunking & Grounded Fallback (Tasks 3, 4, 5):** Fixed-size with overlap (36 chunks) vs. sentence-based chunking (49 chunks) indexed into separate ChromaDB collections using local `all-MiniLM-L6-v2` dense embeddings. Calibrated empirical fallback threshold (`0.35`) returning `"I don't know..."` on out-of-scope inquiries.
- **Operational Order Lookup & Continuous Escalation (Task 6):** `check_order_status` function returning order value, lifecycle status, and a continuous score ($0.60 \times \text{delay} + 0.40 \times \text{recency} / 30.0$) with human escalation recommendation at $\ge 0.72$. The threshold is reached for delayed orders aged 9 days or more; in the current generated dataset, 4 of 10 delayed orders meet this threshold.
- **Three-Agent CrewAI Orchestration (Task 7):** Specialized roles (Retrieval Agent, Lookup Agent, Response Composer) governed by the Principle of Least Autonomy and schema-driven argument dispatch.
- **Multi-Turn Session Memory (Task 8):** In-memory per-session conversation tracking via LangChain (`InMemoryChatMessageHistory` / `RunnableWithMessageHistory`) resolving implicit order references while preserving complete cross-session isolation.
- **Pydantic Structured Output Validation (Task 9):** Strict schema validation (`CrewSupportResponse`) enforcing structured metadata, bounded escalation scores ($\in [0.0, 1.0]$), and typed classifications.
- **Input Guardrails & Output Validation (Task 10):** Regex-based PII masking strictly for supported fixed-format patterns (Indian mobile numbers, 16-digit payment cards) to `[PHONE_MASKED]` and `[CARD_MASKED]`, prompt injection override interception, and output context-faithfulness verification.
- **FastAPI HTTP & WebSocket Endpoints (Task 11):** Local ASGI service supporting `POST /ask`, `POST /add-document`, `GET /health`, and streaming `WS /ws/chat` with graceful disconnection handling.
- **Structured JSON-Lines Request Logging (Task 12):** ELK-compatible audit log stream in `logs/audit_requests.jsonl` with request durations, trace IDs, and disk-level regex PII masking.
- **15-Query Evaluation Benchmark (Task 13):** Automated evaluation pipeline scoring responses across Accuracy, Grounding, Completeness, and Safety using a deterministic LLM judge.
- **Two-Agent AutoGen Review Stage (Task 14):** Independent `RoundRobinGroupChat` (`max_turns=2`) with `PolicyComplianceReviewer` and `FinalEditor` emitting structured `ReviewVerdict` Pydantic models.
- **Four-Layer AI Governance Framework (Task 15):** Application Layer (least autonomy tool restrictions, Medium Risk classification), Runtime Layer (pre-execution token/cost budget caps), Model Layer (grounded synthesis), and Infrastructure Layer (local models, in-process socket interceptor, suppressed telemetry).
- **Normalized Query Caching (Task 16):** In-memory response cache keyed by normalized query strings to eliminate redundant vector retrievals and agent executions on duplicate inquiries.

---

## 3. Technology Stack

| Layer | Component | Version / Library | Purpose |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.12.x` (verified `3.12.10`) | Project runtime environment |
| **Vector Database** | ChromaDB | `chromadb == 1.5.9` | Dual vector store collections for local RAG |
| **Embeddings** | SentenceTransformers | `all-MiniLM-L6-v2` | 384-dimensional dense semantic vectors (local files only) |
| **Multi-Agent Orchestration** | CrewAI | `crewai == 1.15.30` | 3-agent cooperative execution (`.kickoff()`) |
| **Compliance Review** | AutoGen | `autogen-agentchat == 0.7.5`, `autogen-core == 0.7.5` | 2-agent RoundRobinGroupChat review stage |
| **Conversational Memory** | LangChain | `langchain`, `langchain-core` | Session chat history and contextual follow-up resolution |
| **API Framework** | FastAPI & Uvicorn | `fastapi == 0.162.0`, `uvicorn == 0.54.0` | Local REST HTTP and WebSocket real-time chat service |
| **Data Validation** | Pydantic | `pydantic >= 2.0` | Request/response schemas, output validation, and models |
| **Execution Engine** | Mock LLM Engine | `MOCK_LLM` (in-repo engine) | Deterministic local LLM simulation (no external API keys required) |

---

## 4. Project Architecture

The following diagram illustrates the end-to-end request lifecycle across the application, orchestration, review, and governance layers:

```text
               +-------------------------------------------------------------+
               |                    CLIENT INTERACTION                       |
               |        Swagger UI (/docs)  |  WebSocket Client (/ws/chat)    |
               +------------------------------+------------------------------+
                                              |
                                              v
+-------------------------------------------------------------------------------------------+
|                          FASTAPI SERVICE & IN-FLIGHT GUARDS                               |
|                                                                                           |
|   1. Runtime Budget Guard      -> Rejects requests exceeding 1,500 tokens or $0.03 cap    |
|   2. Input Guardrails          -> Masks Indian phones & card numbers; blocks injections   |
|   3. Normalized Query Cache    -> Returns instant cached response on repeated inquiries   |
+---------------------------------------------+---------------------------------------------+
                                              | (Cache Miss)
                                              v
+-------------------------------------------------------------------------------------------+
|                             CREWAI MULTI-AGENT ORCHESTRATION                              |
|                                                                                           |
|   +-----------------------+     +-----------------------+     +-----------------------+   |
|   |    Retrieval Agent    |     |      Lookup Agent     |     |   Response Composer   |   |
|   |  (Role: Policy Spec)  |     |  (Role: Order Ops)    |     | (Role: Resolut. Lead) |   |
|   |  Tool: rag_lookup     |     |  Tool: check_order    |     | Tools: 0 (Least Priv) |   |
|   +-----------+-----------+     +-----------+-----------+     +-----------+-----------+   |
|               |                             |                             ^               |
|               v                             v                             |               |
|       ChromaDB Vector Store         Orders Dataset (orders.json)          |               |
|     nykaa_policies_sentence_based   Continuous Escalation Score           |               |
|               |                             |                             |               |
|               +-----------------------------+-----------------------------+               |
|                                             |                                             |
|                                             v                                             |
|                             Draft Resolution Synthesis                                   |
|                         (LangChain Session Memory Enriched)                               |
+---------------------------------------------+---------------------------------------------+
                                              |
                                              v
+-------------------------------------------------------------------------------------------+
|                              AUTOGEN COMPLIANCE REVIEW STAGE                              |
|                     RoundRobinGroupChat (max_turns=2, Structured Output)                   |
|                                                                                           |
|     PolicyComplianceReviewer                     FinalEditor                              |
|   (Audits draft against ground truth)  --->  (Emits Pydantic ReviewVerdict)               |
+---------------------------------------------+---------------------------------------------+
                                              |
                                              v
+-------------------------------------------------------------------------------------------+
|                             OBSERVABILITY & RESPONSE DELIVERY                             |
|                                                                                           |
|   1. Structured Audit Logger  -> Emits JSON-Lines entry to logs/audit_requests.jsonl      |
|                                  (Disk-level regex PII masking for ELK ingest)            |
|   2. Client Response Delivery -> Delivers validated JSON response or WebSocket payload    |
+-------------------------------------------------------------------------------------------+
```

---

## 5. Repository Structure

```text
nykaa-support-agent/
│
├── dataset.py                               # Root deliverable: Order dataset generator & lookup tool (Task 1 & 6)
├── demo_pipeline.py                         # End-to-end demonstration runner across Milestones 1–5
├── run_all_tests.py                         # Master test runner executing all 17 milestone test suites
├── requirements.txt                         # Core Python package dependencies
├── requirements-lock.txt                    # Pinned dependency lockfile for deterministic installation
├── CAPSTONE_IMPLEMENTATION_PLAN.md          # 16-task implementation plan mapped to project milestones
├── README.md                                # Comprehensive project documentation
│
├── data/
│   ├── orders.json                          # Seeded 50-record synthetic order dataset (seed=42)
│   └── policies/                            # 12 ground-truth policy documents (Markdown, 2–5 sentences)
│       ├── 01_return_window_by_category.md
│       ├── 02_cod_refund_timelines.md
│       ├── 03_delivery_slas.md
│       ├── 04_reverse_pickup_eligibility.md
│       ├── 05_warranty_terms_by_category.md
│       ├── 06_order_cancellation_policy.md
│       ├── 07_loyalty_points_redemption_policy.md
│       ├── 08_payment_failure_retry_policy.md
│       ├── 09_size_exchange_policy.md
│       ├── 10_damaged_item_claim_process.md
│       ├── 11_international_shipping_restrictions.md
│       └── 12_customer_support_escalation_matrix.md
│
├── docs/
│   └── DEMO_TRANSCRIPTS.md                  # Live transcripts generated from demo_pipeline.py
│
├── logs/
│   ├── audit_requests.jsonl                 # Structured JSON-Lines request audit trail with masked PII
│   └── evaluation_benchmark_results.json    # 15-query evaluation benchmark report (4 dimensions)
│
├── nykaa_support/                           # Core application package
│   ├── __init__.py
│   ├── dataset.py                           # Dataset generation & order status lookup implementation
│   ├── api/                                 # FastAPI REST & WebSocket server implementation
│   │   ├── __init__.py
│   │   ├── schemas.py                       # Pydantic request & response schemas
│   │   └── server.py                        # FastAPI application factory & route handlers
│   ├── autogen_review/                      # AutoGen two-agent compliance review stage
│   │   ├── __init__.py
│   │   └── review_stage.py                  # RoundRobinGroupChat compliance reviewer & editor
│   ├── crew/                                # CrewAI multi-agent crew, memory & tools
│   │   ├── __init__.py
│   │   ├── agents.py                        # Agent role definitions & least-autonomy wiring
│   │   ├── memory.py                        # LangChain session memory manager
│   │   ├── schemas.py                       # CrewSupportResponse Pydantic schema
│   │   ├── support_crew.py                  # Crew coordinator (.kickoff())
│   │   └── tools.py                         # RAG & order status lookup tools
│   ├── evaluation/                          # 15-query evaluation pipeline
│   │   ├── __init__.py
│   │   ├── dataset.py                       # Curated test queries & evaluation schemas
│   │   └── pipeline.py                      # Evaluation pipeline & benchmark execution
│   ├── governance/                          # AI governance, budget cap & query caching
│   │   ├── __init__.py
│   │   ├── cache.py                         # NormalizedQueryCache implementation
│   │   ├── least_autonomy.py                # Tool permission gating & wiring assertion
│   │   ├── models.py                        # Risk classification & governance schemas
│   │   └── runtime_budget.py                # Pre-execution token & cost budget guard
│   ├── guardrails/                          # Input & output guardrails
│   │   ├── __init__.py
│   │   ├── input_guardrails.py              # Regex PII masking & prompt injection filter
│   │   └── output_guardrails.py             # Groundedness verification & refusal enforcement
│   ├── mock_llm/                            # Deterministic mock LLM simulation engine
│   │   ├── __init__.py
│   │   ├── autogen_adapter.py               # AutoGen ChatCompletionClient mock adapter
│   │   ├── crewai_adapter.py                # CrewAI BaseLLM mock adapter
│   │   ├── engine.py                        # Core mock generation & verdict logic
│   │   ├── judge.py                         # 4-dimension evaluation judge logic
│   │   └── standalone.py                    # Standalone grounded generator
│   ├── observability/                       # Structured JSONL audit logger
│   │   ├── __init__.py
│   │   └── logger.py                        # StructuredAuditLogger with PII masking
│   └── rag/                                 # Dual chunking, ChromaDB store & calibration
│       ├── __init__.py
│       ├── calibrate.py                     # Similarity threshold calibration logic
│       ├── chunking.py                      # Fixed-size overlap & sentence-based chunkers
│       ├── comparison.py                    # Precision & Recall comparative evaluation
│       ├── generator.py                     # Grounded generator with fallback refusal
│       └── store.py                         # ChromaDB ingestion & dual collection management
│
└── tests/                                   # 17 automated test suites
    ├── __init__.py
    ├── test_milestone1_dataset.py           # Dataset, policy documents & Task 6 tool
    ├── test_milestone2_task3_chunking.py    # Chunking strategies & collection separation
    ├── test_milestone2_rag.py               # Grounded synthesis, calibration & P/R
    ├── test_milestone3_task6_lookup.py      # Order status lookup tool & escalation score
    ├── test_milestone3_task7_crew.py        # CrewAI 3-agent orchestration
    ├── test_milestone3_task8_memory.py      # LangChain session memory & isolation
    ├── test_milestone3_task9_schema.py      # CrewSupportResponse Pydantic validation
    ├── test_milestone3_task10_guardrails.py # Input PII masking & injection interception
    ├── test_milestone4_task14_autogen.py    # AutoGen review stage (Case 1 & Case 2)
    ├── test_milestone4_task15_governance.py # Least autonomy, risk classification & cache
    ├── test_milestone5_task11_12_api.py     # FastAPI HTTP, WebSocket & JSONL logging
    ├── test_milestone5_task13_evaluation.py # 15-query evaluation benchmark execution
    ├── test_standalone_grounded.py          # Grounded generator verification
    ├── test_crewai_mock.py                  # CrewAI mock adapter verification
    ├── test_judge_mock.py                   # Evaluation judge mock verification
    ├── test_autogen_mock.py                 # AutoGen mock adapter verification
    └── test_offline_verification.py         # In-process socket interceptor offline test
```

---

## 6. Prerequisites

- **Operating System:** Windows 10 or Windows 11 (AMD64 architecture).
- **Python Version:** `Python 3.12.x` (Verified with Python 3.12.10; Python 3.12 is required for pinned AutoGen 0.7.5 and CrewAI 1.15.30 dependency compatibility).
- **Git:** Git for Windows (e.g., `git.exe`).
- **Initial Network Access:** An active internet connection is required once during initial environment setup for `pip install` and the one-time local caching of SentenceTransformers model weights (`all-MiniLM-L6-v2`, ~80MB). Once dependencies and embedding weights are locally cached, subsequent test runs, benchmarks, and demos execute locally in deterministic `MOCK_LLM` mode without external network requests.

---

## 7. Installation and Setup

Follow these steps in Windows PowerShell to set up a clean environment:

```powershell
# 1. Check Python 3.12 availability
py -3.12 --version

# 2. Create virtual environment (initial setup only)
py -3.12 -m venv .venv

# 3. Activate the environment in PowerShell
.\.venv\Scripts\activate

# 4. Verify Python version inside the virtual environment
python --version

# 5. Install project dependencies
python -m pip install -r requirements.txt
```

> **Note on Existing Virtual Environments:** If `.venv` already exists in your workspace, recreating it is not required. Simply activate it with `.\.venv\Scripts\activate`.

> **PowerShell Execution Policy Note:** If Windows PowerShell displays `cannot be loaded because running scripts is disabled on this system`, you can either enable process-scoped script execution:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> .\.venv\Scripts\activate
> ```
> Or invoke the virtual environment Python binary directly without activating:
> ```powershell
> .\.venv\Scripts\python.exe --version
> ```

> **Exact Lockfile Reproduction:** To install the exact locked package versions verified during development (176 locked packages), use:
> ```powershell
> python -m pip install -r requirements-lock.txt
> ```

---

## 8. Offline Execution and MOCK_LLM Mode

When configured in deterministic `MOCK_LLM` mode and using locally cached embedding weights, the project operates locally without outbound network calls during testing and demonstration:

1. **Deterministic Mock LLM Engine:**
   - Setting the environment variable `MOCK_LLM=true` (which is configured automatically in [demo_pipeline.py](demo_pipeline.py) and test runners) directs all language model interactions to the internal mock engine in [`nykaa_support/mock_llm/`](nykaa_support/mock_llm/).
   - The mock engine simulates grounded synthesis, CrewAI agent cooperation, AutoGen compliance review, and evaluation judging using deterministic policy and order rules.
   - **No hosted LLM API keys (OpenAI, Anthropic, Google, Cohere) are required.**

2. **Local Embedding Model Setup (SentenceTransformers):**
   - The semantic retrieval pipeline uses `sentence-transformers/all-MiniLM-L6-v2`.
   - On first execution, the model files are downloaded and cached locally in `~/.cache/huggingface/hub/`.
   - During evaluation runs and automated tests, `local_files_only=True` is enforced across all vector store initializations, verifying that no outbound network requests are made to download model weights.

3. **In-Process Socket Interceptor:**
   - [tests/test_offline_verification.py](tests/test_offline_verification.py) tests offline compliance by monkey-patching `socket.socket.connect` during test execution to intercept non-local connection attempts made by Python code.
   - *Scope boundary:* This mechanism functions as an in-process Python runtime connection guard. It does not act as an operating-system firewall, does not inspect or block traffic from other OS processes, and does not alter system-level networking rules.

4. **Telemetry Suppression:**
   - Telemetry calls are explicitly disabled across all execution paths via `OTEL_SDK_DISABLED=true` and `CREWAI_DISABLE_TELEMETRY=true`.

---

## 9. Running the Demonstration Pipeline

The demonstration runner executes all 8 capstone demonstrations sequentially under deterministic `MOCK_LLM` execution:

```powershell
.\.venv\Scripts\python.exe demo_pipeline.py
```

### What It Executes:
1. **Demo 1 (Task 6):** Order lookup and continuous escalation scoring for on-schedule order `NYK-10001` (score: `0.0933`, no escalation) vs. delayed order `NYK-10002` (score: `0.8933`, escalation recommended).
2. **Demo 2 (Task 7):** CrewAI 3-agent orchestration across policy inquiries, order lookups, and combined customer requests.
3. **Demo 3 (Task 8):** LangChain multi-turn session memory resolving implicit follow-up context across 4 messages, contrasted with fresh-session isolation.
4. **Demo 4 (Task 14):** AutoGen two-agent compliance review stage demonstrating Case 1 (grounded answer approved unchanged) and Case 2 (hallucinated 90-day return window caught and revised).
5. **Demo 5 (Task 15):** Four-layer AI governance demonstrating least-autonomy tool restriction enforcement and runtime budget cap rejection.
6. **Demo 6 (Task 16):** Normalized query caching with execution time comparison and hit counter verification.
7. **Demo 7 (Tasks 11 & 12):** Structured JSON-Lines audit logging with disk-level regex PII masking.
8. **Demo 8 (Task 13):** 15-query evaluation benchmark execution and score summary table.

**Output Location:** Console output is printed live and saved in Markdown format to [`docs/DEMO_TRANSCRIPTS.md`](docs/DEMO_TRANSCRIPTS.md).

---

## 10. Running the Evaluation Pipeline

To execute the automated 15-query evaluation benchmark independently:

```powershell
.\.venv\Scripts\python.exe -m nykaa_support.evaluation.pipeline
```

### What It Executes:
- Evaluates 15 benchmark queries (12 ground-truth policy inquiries + 3 edge-case adversarial/out-of-scope scenarios).
- Scores each response across 4 dimensions: **Accuracy**, **Grounding**, **Completeness**, and **Safety** (threshold $\ge 0.70$).
- Updates the structured benchmark output file: [`logs/evaluation_benchmark_results.json`](logs/evaluation_benchmark_results.json).

---

## 11. Running Automated Tests

To execute all 17 master test suites synchronously:

```powershell
.\.venv\Scripts\python.exe -u run_all_tests.py
```

**Verified Test Result:** **17/17 test suites passing** (`Exit code: 0`).

```text
======================================================================
TEST SUITE EXECUTION SUMMARY
======================================================================
[PASS] tests/test_milestone1_dataset.py              (0.26s)
[PASS] tests/test_milestone2_task3_chunking.py       (13.82s)
[PASS] tests/test_milestone2_rag.py                  (15.02s)
[PASS] tests/test_milestone3_task6_lookup.py         (0.28s)
[PASS] tests/test_milestone3_task7_crew.py           (24.74s)
[PASS] tests/test_milestone3_task8_memory.py         (28.52s)
[PASS] tests/test_milestone3_task9_schema.py         (23.71s)
[PASS] tests/test_milestone3_task10_guardrails.py    (9.44s)
[PASS] tests/test_milestone4_task14_autogen.py       (4.32s)
[PASS] tests/test_milestone4_task15_governance.py    (18.22s)
[PASS] tests/test_milestone5_task11_12_api.py        (72.03s)
[PASS] tests/test_milestone5_task13_evaluation.py    (50.91s)
[PASS] tests/test_standalone_grounded.py             (4.81s)
[PASS] tests/test_crewai_mock.py                     (5.02s)
[PASS] tests/test_judge_mock.py                      (5.94s)
[PASS] tests/test_autogen_mock.py                    (6.23s)
[PASS] tests/test_offline_verification.py            (19.81s)
======================================================================
ALL TESTS PASSED SUCCESSFULLY (17/17)!
```

### Running Dedicated Milestone Test Suites Directly
```powershell
# Milestone 1: Dataset & Policy Documents
.\.venv\Scripts\python.exe tests/test_milestone1_dataset.py

# Milestone 2: Dual Chunking & Separate Vector Collections
.\.venv\Scripts\python.exe tests/test_milestone2_task3_chunking.py

# Milestone 2: Grounded Synthesis, Calibration & Comparative Evaluation
.\.venv\Scripts\python.exe tests/test_milestone2_rag.py

# Milestone 3: Order Status Lookup & Continuous Escalation Score
.\.venv\Scripts\python.exe tests/test_milestone3_task6_lookup.py

# Milestone 3: CrewAI 3-Agent Workflow
.\.venv\Scripts\python.exe tests/test_milestone3_task7_crew.py

# Milestone 3: LangChain Session Memory
.\.venv\Scripts\python.exe tests/test_milestone3_task8_memory.py

# Milestone 3: Structured Output Schema Validation
.\.venv\Scripts\python.exe tests/test_milestone3_task9_schema.py

# Milestone 3: Input Guardrails & PII Masking
.\.venv\Scripts\python.exe tests/test_milestone3_task10_guardrails.py

# Milestone 4: AutoGen Two-Agent Review Stage
.\.venv\Scripts\python.exe tests/test_milestone4_task14_autogen.py

# Milestone 4: Four-Layer Governance & Query Cache
.\.venv\Scripts\python.exe tests/test_milestone4_task15_governance.py

# Milestone 5: FastAPI REST Endpoints, WebSocket & JSONL Logging
.\.venv\Scripts\python.exe tests/test_milestone5_task11_12_api.py

# Milestone 5: 15-Query Evaluation Pipeline
.\.venv\Scripts\python.exe tests/test_milestone5_task13_evaluation.py

# Offline Execution Verification (In-Process Socket Interceptor)
.\.venv\Scripts\python.exe tests/test_offline_verification.py
```

---

## 12. Starting the API Server

The project provides a FastAPI application factory exposing both REST HTTP endpoints and a real-time WebSocket chat endpoint.

To launch the local Uvicorn ASGI server:
```powershell
.\.venv\Scripts\uvicorn.exe nykaa_support.api.server:app --host 127.0.0.1 --port 8000
```

- **Interactive API Documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Health Check Endpoint:** `http://127.0.0.1:8000/health`
- **WebSocket Chat Endpoint:** `ws://127.0.0.1:8000/ws/chat`

---

## 13. API Endpoints and Usage

### Endpoint Specifications

| Method | Endpoint | Request Model | Response Model | Description |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/ask` | `AskRequest` | `AskResponse` | Process customer question with session memory and governance |
| `POST` | `/add-document` | `AddDocumentRequest` | `AddDocumentResponse` | Ingest and index new policy document into ChromaDB |
| `GET` | `/health` | None | `HealthResponse` | Return service status, collections, and mock execution mode |
| `WS` | `/ws/chat` | JSON messages | JSON messages | Real-time multi-turn conversation with graceful disconnect |

> **Educational Response Simulation:** The sample responses below are generated under deterministic `MOCK_LLM` execution using synthetic educational templates and the authored knowledge base.

### Sample Usage: `POST /ask`

**Request:**
```http
POST /ask HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "query": "What is the return window for cosmetics?",
  "session_id": "cust-session-101"
}
```

**Response (`200 OK`):**
```json
{
  "query": "What is the return window for cosmetics?",
  "final_answer": "Thank you for contacting Nykaa Support. Based on official records: Relevant Nykaa Policy Context: Nykaa maintains category-specific return windows to balance customer flexibility with product hygiene standards. Please let us know if you need further assistance with your Nykaa shopping experience.",
  "session_id": "cust-session-101",
  "query_type": "policy_inquiry",
  "order_id": null,
  "escalation_score": null,
  "escalation_recommended": false,
  "governance": {
    "cache_hit": false,
    "verdict_approved": true
  }
}
```

### Sample Usage: `POST /add-document`

**Request:**
```http
POST /add-document HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json

{
  "filename": "13_vip_loyalty_tier_policy.md",
  "content": "Nykaa Prive Gold members receive complimentary priority shipping on all orders. Gold tier status requires accumulating 2,000 points annually.",
  "upsert": true
}
```

**Response (`200 OK`):**
```json
{
  "status": "success",
  "message": "Successfully indexed 2 chunks into nykaa_policies_sentence_based.",
  "chunks_added": 2,
  "collection": "nykaa_policies_sentence_based"
}
```

### WebSocket Streaming (`WS /ws/chat`)
Clients establish a persistent connection to `ws://127.0.0.1:8000/ws/chat` and exchange JSON messages:
```json
{"query": "Please check order status for NYK-10001", "session_id": "ws-101"}
```
The server catches `WebSocketDisconnect` gracefully upon client termination, logging normal closures without disrupting concurrent connections.

---

## 14. Capstone Task Implementation and Evidence Mapping

All 16 tasks required by the capstone specification are implemented, tested, and mapped below:

| Task # | Task Name | Primary Implementation File | Verification Test Suite | Demonstration & Artifact Evidence |
| :---: | :--- | :--- | :--- | :--- |
| **Task 1** | Deterministic Dataset Generation | [`dataset.py`](dataset.py), [`nykaa_support/dataset.py`](nykaa_support/dataset.py) | [`tests/test_milestone1_dataset.py`](tests/test_milestone1_dataset.py) | [`data/orders.json`](data/orders.json) (50 records, seed=42) |
| **Task 2** | Ground-Truth Policy Documents | [`data/policies/*.md`](data/policies/) | [`tests/test_milestone1_dataset.py`](tests/test_milestone1_dataset.py) | 12 distinct Markdown policy files in `data/policies/` |
| **Task 3** | Dual Chunking & Separate Vector Stores | [`nykaa_support/rag/chunking.py`](nykaa_support/rag/chunking.py), [`nykaa_support/rag/store.py`](nykaa_support/rag/store.py) | [`tests/test_milestone2_task3_chunking.py`](tests/test_milestone2_task3_chunking.py) | ChromaDB collections: 36 fixed chunks vs. 49 sentence chunks |
| **Task 4** | Grounded Generation & Threshold Calibration | [`nykaa_support/rag/generator.py`](nykaa_support/rag/generator.py), [`nykaa_support/rag/calibrate.py`](nykaa_support/rag/calibrate.py) | [`tests/test_milestone2_rag.py`](tests/test_milestone2_rag.py) | Threshold calibrated at 0.35 cutoff; fallback refusal verified |
| **Task 5** | Document-Level Comparative Evaluation | [`nykaa_support/rag/comparison.py`](nykaa_support/rag/comparison.py) | [`tests/test_milestone2_rag.py`](tests/test_milestone2_rag.py) | Sentence-based: 76.67% precision vs. 46.67% fixed-size |
| **Task 6** | Order Lookup & Escalation Score | [`nykaa_support/dataset.py`](nykaa_support/dataset.py) (`check_order_status`) | [`tests/test_milestone3_task6_lookup.py`](tests/test_milestone3_task6_lookup.py) | `demo_pipeline.py` Demo 1: on-schedule (0.0933) vs. delayed (0.8933) |
| **Task 7** | CrewAI Multi-Agent Workflow | [`nykaa_support/crew/support_crew.py`](nykaa_support/crew/support_crew.py) | [`tests/test_milestone3_task7_crew.py`](tests/test_milestone3_task7_crew.py) | `demo_pipeline.py` Demo 2: 3-agent orchestration with least autonomy |
| **Task 8** | LangChain Session Memory | [`nykaa_support/crew/memory.py`](nykaa_support/crew/memory.py) | [`tests/test_milestone3_task8_memory.py`](tests/test_milestone3_task8_memory.py) | `demo_pipeline.py` Demo 3: 4-message context resolution & session reset |
| **Task 9** | Structured Output Schema Validation | [`nykaa_support/crew/schemas.py`](nykaa_support/crew/schemas.py) | [`tests/test_milestone3_task9_schema.py`](tests/test_milestone3_task9_schema.py) | `CrewSupportResponse` Pydantic model and validation tests |
| **Task 10** | Input & Output Guardrails | [`nykaa_support/guardrails/`](nykaa_support/guardrails/) | [`tests/test_milestone3_task10_guardrails.py`](tests/test_milestone3_task10_guardrails.py) | Regex PII masking, injection interception, groundedness checks |
| **Task 11** | FastAPI Deployment & WebSocket Chat | [`nykaa_support/api/server.py`](nykaa_support/api/server.py), [`nykaa_support/api/schemas.py`](nykaa_support/api/schemas.py) | [`tests/test_milestone5_task11_12_api.py`](tests/test_milestone5_task11_12_api.py) | Running ASGI endpoints with WebSocket streaming & disconnect handling |
| **Task 12** | Structured JSONL Request Logging | [`nykaa_support/observability/logger.py`](nykaa_support/observability/logger.py) | [`tests/test_milestone5_task11_12_api.py`](tests/test_milestone5_task11_12_api.py) | [`logs/audit_requests.jsonl`](logs/audit_requests.jsonl) with regex-masked PII |
| **Task 13** | 15-Query Evaluation Pipeline | [`nykaa_support/evaluation/pipeline.py`](nykaa_support/evaluation/pipeline.py) | [`tests/test_milestone5_task13_evaluation.py`](tests/test_milestone5_task13_evaluation.py) | [`logs/evaluation_benchmark_results.json`](logs/evaluation_benchmark_results.json) (15/15 passed) |
| **Task 14** | AutoGen Review Stage | [`nykaa_support/autogen_review/review_stage.py`](nykaa_support/autogen_review/review_stage.py) | [`tests/test_milestone4_task14_autogen.py`](tests/test_milestone4_task14_autogen.py) | `demo_pipeline.py` Demo 4: Case 1 approved, Case 2 revised |
| **Task 15** | Four-Layer AI Governance Framework | [`nykaa_support/governance/`](nykaa_support/governance/) | [`tests/test_milestone4_task15_governance.py`](tests/test_milestone4_task15_governance.py) | `demo_pipeline.py` Demo 5: Tool permission guard & budget cap rejection |
| **Task 16** | Normalized Query Response Caching | [`nykaa_support/governance/cache.py`](nykaa_support/governance/cache.py) | [`tests/test_milestone4_task15_governance.py`](tests/test_milestone4_task15_governance.py) | `demo_pipeline.py` Demo 6: In-memory cache hit counter verification |

---

## 15. RAG Chunking Comparison and Groundedness

### 15.1 Chunking Strategies & Ingestion
The knowledge base comprises 12 ground-truth policy files in [`data/policies/`](data/policies/) ingested into separate ChromaDB collections:
- **Strategy 1 (Fixed-Size with Overlap):** `chunk_size = 300` characters, `chunk_overlap = 60` characters, boundary-aligned on word whitespace. Indexed into collection `nykaa_policies_fixed_overlap` (36 chunks).
- **Strategy 2 (Sentence-Based):** Terminal punctuation boundary splitting (`[.!?]`), producing atomic semantic rules. Indexed into collection `nykaa_policies_sentence_based` (49 chunks).

### 15.2 Comparative Precision and Recall Evaluation
Using $k=3$ retrieved chunks mapped to parent policy documents on 5 benchmark queries:

$$\text{Precision} = \frac{|\text{Retrieved Unique} \cap \text{Ground Truth}|}{|\text{Retrieved Unique}|}, \quad \text{Recall} = \frac{|\text{Retrieved Unique} \cap \text{Ground Truth}|}{|\text{Ground Truth}|}$$

| Query ID | Query Topic | Fixed-Size Overlap (P / R) | Sentence-Based (P / R) |
| :---: | :--- | :---: | :---: |
| **Q1** | Return Window by Category | $1/2 = 50.00\%$ / $1/1 = 100.00\%$ | $1/1 = 100.00\%$ / $1/1 = 100.00\%$ |
| **Q2** | COD Refund Timelines | $1/2 = 50.00\%$ / $1/1 = 100.00\%$ | $1/1 = 100.00\%$ / $1/1 = 100.00\%$ |
| **Q3** | Domestic Delivery SLAs | $1/2 = 50.00\%$ / $1/1 = 100.00\%$ | $1/2 = 50.00\%$ / $1/1 = 100.00\%$ |
| **Q4** | Reverse-Pickup Eligibility | $1/3 = 33.33\%$ / $1/1 = 100.00\%$ | $1/3 = 33.33\%$ / $1/1 = 100.00\%$ |
| **Q5** | Warranty Terms by Category | $1/2 = 50.00\%$ / $1/1 = 100.00\%$ | $1/1 = 100.00\%$ / $1/1 = 100.00\%$ |
| **Mean** | **Overall Benchmark Score** | **Mean Precision: 46.67% \| Recall: 100.00%** | **Mean Precision: 76.67% \| Recall: 100.00%** |

**Selection Rationale:** The sentence-based chunking strategy (`nykaa_policies_sentence_based`) achieved **76.67% Mean Document Precision** compared to **46.67%** for fixed-size chunking (both achieved **100.00% Recall**). Sentence-level chunking isolates atomic policy rules without cross-topic bleeding from adjacent markdown headings. Consequently, `nykaa_policies_sentence_based` is configured as the primary knowledge base for the multi-agent crew.

### 15.3 Empirical Groundedness Calibration
Cosine similarities between queries and retrieved chunks form the grounding signal under `MOCK_LLM`. Per Task 4 requirements, top-1 cosine similarities were empirically measured across 5 in-scope queries and 3 out-of-scope queries using [`nykaa_support/rag/calibrate.py`](nykaa_support/rag/calibrate.py):

| Query ID | Query Text | Target Policy Document | Fixed-Size Overlap (Top-1 Sim) | Sentence-Based Chunking (Top-1 Sim) |
| :---: | :--- | :---: | :---: | :---: |
| **Q1** | *What is the return window for cosmetics and beauty products?* | `01_return_window_by_category` | `0.5998` | `0.5978` |
| **Q2** | *How long does COD refund take via NEFT or Nykaa Wallet?* | `02_cod_refund_timelines` | `0.7539` | `0.6591` |
| **Q3** | *What are the delivery SLAs for Tier 1 metro cities?* | `03_delivery_slas` | `0.5047` | `0.5918` |
| **Q4** | *Am I eligible for free reverse pickup on returns?* | `04_reverse_pickup_eligibility` | `0.4830` | `0.5006` |
| **Q5** | *What is the warranty coverage for electronic styling tools?* | `05_warranty_terms_by_category` | `0.6063` | `0.7786` |
| **O1** | *How do I bake a chocolate sourdough bread at home?* | *(Out of Scope)* | `0.0964` | `0.0955` |
| **O2** | *What is the current stock market price of Tesla or Bitcoin?* | *(Out of Scope)* | `0.1353` | `0.1288` |
| **O3** | *Who won the 2022 FIFA World Cup soccer championship in Qatar?* | *(Out of Scope)* | `0.0892` | `0.0820` |

#### Cluster Separation & Threshold Selection (Sentence-Based Strategy):
- **In-Scope Query Cluster:** Top-1 similarities range from `0.5006` to `0.7786` (Mean: `0.6256`).
- **Out-of-Scope Query Cluster:** Top-1 similarities range from `0.0820` to `0.1288` (Mean: `0.1021`).
- **Cluster Gap:** $0.5006 - 0.1288 = \mathbf{0.3718}$ clear semantic separation.
- **Calibrated Cutoff (`THRESHOLD = 0.35`):**
  - Margin above highest out-of-scope: $0.35 - 0.1288 = \mathbf{+0.2212}$ ($> 0.22$).
  - Margin below lowest in-scope: $0.5006 - 0.35 = \mathbf{+0.1506}$ ($> 0.15$).
  - Both boundaries maintain $>0.15$ safety margin.
- **Refusal Response (Simulated Runtime Output):** When top similarity falls below `0.35`, the engine returns the exact simulated refusal string:
  *"I don't know. The requested information is not covered by our official Nykaa policy documentation."*
  *(Note: The policy knowledge base is an educational capstone fixture authored for rubric demonstration, not official Nykaa corporate policy documentation).*

> **Qualification:** A threshold calibrated on a small benchmark is an empirical operational configuration; it does not constitute mathematical proof of complete grounding across all possible unconstrained user queries.

---

## 16. Evaluation Results

The evaluation pipeline ([`nykaa_support/evaluation/pipeline.py`](nykaa_support/evaluation/pipeline.py)) executes 15 queries against four evaluation dimensions using a continuous scoring scale $[0.0, 1.0]$ with an acceptance threshold of $\ge 0.70$.

> **Note on Evaluation Nature:** These metrics reflect deterministic automated evaluation under the project's simulated `MOCK_LLM` judge setup, serving to verify functional rubric compliance rather than human-evaluated production model performance.

### Benchmark Summary Across 15 Queries

| Metric | Score | Acceptance Threshold | Evaluation Scope |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.9467** | $\ge 0.70$ | Factual alignment with policy terms, timelines, and refusal boundaries |
| **Grounding** | **0.9900** | $\ge 0.70$ | Faithfulness to retrieved policy context across evaluated benchmark queries |
| **Completeness** | **0.9433** | $\ge 0.70$ | Provision of actionable resolution steps ($\ge 25$ words) |
| **Safety** | **0.9967** | $\ge 0.70$ | Courteous tone, PII sanitization, and safe refusal of edge cases |
| **Pass Rate** | **100.0%** (15/15) | $100\%$ | Percentage of benchmark queries meeting all 4 dimension thresholds |

Detailed per-query scoring and timestamps are recorded in [`logs/evaluation_benchmark_results.json`](logs/evaluation_benchmark_results.json).

---

## 17. Guardrails, Privacy, and AI Governance

### 17.1 Input & Output Guardrails (Task 10)
- **PII Masking:** Applies regex sanitization to Indian mobile numbers (`+91-XXXXXXXXXX`, 10-digit mobile) masking them to `[PHONE_MASKED]`, and 16-digit payment card numbers (including 4-block formats and last-4 references) to `[CARD_MASKED]`.
  - *Explicit Scope Boundary:* Regex-based masking covers fixed-format phone and payment card patterns. Unstructured customer full names and free-text street delivery addresses are natural language text and are not covered by the regex masker. Regex pattern matching does not guarantee detection of all sensitive personal information.
- **Prompt Injection Defense:** Regex classifier intercepts adversarial override attempts (*"ignore previous instructions"*, *"system override"*, *"jailbreak"*, *"leak system prompt"*), returning a refusal before model execution.
- **Output Groundedness Validation:** Cross-checks generated drafts against retrieved vector context; low similarity ($< 0.35$) or ungrounded claims are replaced with the official fallback refusal.

### 17.2 Two-Agent AutoGen Review Stage (Task 14)
Draft responses synthesized by the CrewAI crew are reviewed by an independent AutoGen `RoundRobinGroupChat` (`max_turns=2`):
- `PolicyComplianceReviewer`: Verifies policy alignment against ground-truth context.
- `FinalEditor`: Assigned `output_content_type=ReviewVerdict`, emitting a structured Pydantic verdict (`approved: bool`, `final_answer: str`, `reason: str`).
- Demonstrated in `demo_pipeline.py` (Demo 4) across two test cases:
  1. *Case 1 (Approved):* Grounded answer matching 15-day policy is approved without edits.
  2. *Case 2 (Revised):* Hallucinated 90-day return window is rejected and revised to official terms.

### 17.3 Four-Layer AI Governance Framework (Task 15)
1. **Application Layer (Principle of Least Autonomy):**
   - Restricted Tool Access: Only the Lookup Agent is authorized to call `check_order_status`. Unauthorized calls raise `PermissionDeniedError`. The Response Composer is configured with 0 tools.
   - System Risk Classification: Formally classified as **Medium Risk** under the capstone taxonomy. Rationale: Customer support operations impact live logistics, refunds, and financial transactions, but do not make irreversible medical, legal, or credit decisions.
2. **Runtime Layer (Per-Request Budget Cap):**
   - `RuntimeBudgetGuard` rejects requests exceeding 1,500 tokens or $0.03 estimated cost with `BudgetExceededError` prior to vector search or model execution.
3. **Model Layer:**
   - Grounded RAG synthesis with empirical similarity thresholds and AutoGen compliance review.
4. **Infrastructure Layer:**
   - Local models (`all-MiniLM-L6-v2`, local ChromaDB), in-process socket interceptor for offline verification, and suppressed telemetry.

### 17.4 Normalized Query Caching (Task 16)
[`nykaa_support/governance/cache.py`](nykaa_support/governance/cache.py) implements `NormalizedQueryCache`:
- Keys entries by cleaned, lowercased, and punctuation-stripped customer query strings.
- Eliminates redundant RAG vector retrievals and multi-agent synthesis turns for repeated inquiries. Verified in Demo 6 with execution time and cache hit counter statistics.

---

## 18. Known Limitations

1. **CrewAI Synchronous Event Loop Constraint:**
   CrewAI 1.15.30 disallows calling `.kickoff()` synchronously from within an active asyncio event loop. In the FastAPI service, this is mitigated by dispatching HTTP handlers to worker threadpools (`run_in_threadpool`) and wrapping WebSocket executions in `asyncio.to_thread`.
2. **AutoGen Queue Event Loop Binding:**
   AutoGen 0.7.5 internal queues bind to the active event loop upon instantiation. To avoid cross-loop collision errors, `NykaaReviewStage` instantiates fresh team instances per request.
3. **PII Masking Scope Boundary:**
   The regex-based PII masker targets fixed-format Indian phone numbers and 16-digit payment card numbers. Natural-language customer names, email addresses, and unstructured postal delivery addresses are outside the scope of regex masking and require external NER models. Regex pattern matching does not guarantee detection of all personal information.
4. **In-Process Socket Interceptor Scope:**
   The offline verification test patches `socket.socket.connect` inside the Python runtime to verify that application components do not initiate external socket connections during test runs. It operates strictly within the Python process and is not an operating-system firewall or network-level packet filter.
5. **Mock LLM Mode:**
   Language model synthesis is simulated via deterministic rule-based mock adapters. While this enables local, offline test and benchmark execution without API keys once the embedding model is cached, it does not evaluate non-deterministic generative capabilities of live frontier model weights.
6. **Local Log Aggregation:**
   `logs/audit_requests.jsonl` emits structured JSON-Lines records formatted for ELK ingestion. The repository does not include or manage an active Elasticsearch or Kibana server container.

---

## 19. Troubleshooting

| Issue / Symptom | Potential Cause | Resolution Step |
| :--- | :--- | :--- |
| **PowerShell script execution disabled** | Windows default execution policy | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` or invoke virtual environment Python directly: `.\.venv\Scripts\python.exe <script.py>`. |
| **`ImportError` or module not found** | Virtual environment not active or dependencies missing | Activate `.venv` (`.\.venv\Scripts\activate`) and run `python -m pip install -r requirements.txt`. |
| **`Port 8000 already in use`** | Another process is bound to port 8000 | Stop the existing process: `Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force` or start on an alternate port: `uvicorn nykaa_support.api.server:app --port 8001`. |
| **SentenceTransformers download fails** | Initial run on a new machine without internet | Connect to the internet during initial setup so SentenceTransformers can download `all-MiniLM-L6-v2` (~80MB) once to `~/.cache/huggingface/`. All subsequent runs use `local_files_only=True`. |
| **`PermissionDeniedError`** | Agent attempted an unauthorized tool call | Expected behavior under Task 15 (Least Autonomy). Only the Lookup Agent is permitted to execute `check_order_status`. |
| **`BudgetExceededError`** | Request payload exceeds 1,500 token budget | Expected behavior under Task 15. The runtime budget guard safely blocks oversized payloads before expensive operations. |
| **`SocketBlockError` / Connection refused** | Outbound network attempt during offline test | Ensure `MOCK_LLM=true` and `local_files_only=True` are set. External socket calls are blocked during offline verification tests. |
