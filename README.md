# AI-Powered Insurance Claims Intelligence Assistant

A Retrieval-Augmented Generation (RAG) system for intelligent insurance claims analysis and fraud detection. Built with LangGraph multi-agent orchestration, hybrid semantic + keyword search, and a React dashboard.

---

## Architecture Overview

```
fraud_oracle.csv
      │
      ▼
┌──────────────────────────────────────────────────┐
│  ETL Pipeline (offline)                          │
│  extract.py  →  transform.py  →  load.py         │
│  Embed (OpenAI text-embedding-3-small, 1536-dim) │
│  → Pinecone (semantic)  +  BM25 (keyword index)  │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
              User Query (HTTP POST /api/v1/query)
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                               │
│                                                                │
│  [1] Small-talk fast path                                      │
│      regex match (hi / hello / thanks …) ──► canned reply      │
│                                                                │
│  [2] Input Guardrails  (input_guardrails.py)                   │
│      ├─ length check (max 2000 chars)                          │
│      ├─ prompt injection detection (regex)                     │
│      ├─ harmful content blocking (weapons / drugs / violence)  │
│      ├─ domain relevance check (insurance keywords)            │
│      └─ PII anonymization via Presidio (optional)              │
│      FAIL ──────────────────────────────────► HTTP 400         │
│                                                                │
│  [3] Query Cache  (query_cache.py)                             │
│      ├─ Stage 1: MD5 exact match                               │
│      └─ Stage 2: cosine similarity ≥ 0.92 (semantic dedup)     │
│      HIT ───────────────────────────────► cached response      │
│                                                                │
│  [4] LangGraph Orchestrator  (90 s timeout)                    │
│      │                                                         │
│      ├─► retrieval_agent                                       │
│      │     Pinecone 60% + BM25 40% → merge → rerank (cross encoder)  │
│      │     optional contextual compression                     │
│      │                                                         │
│      ├─► fraud_agent                                           │
│      │     LLM fraud analysis + rule-based fallback scoring    │
│      │     risk_score ≥ 0.75 → escalate = True                 │
│      │                                                         │
│      ├─► policy_agent  (skipped when escalate = True)          │
│      │     LLM policy compliance check                         │
│      │                                                         │
│      └─► recommendation_agent                                  │
│            synthesises all agent outputs → final answer        │
│                                                                │
│  [5] Output Guardrails  (output_guardrails.py)                 │
│      ├─ clamp risk_score to [0.0, 1.0]                         │
│      ├─ remove hallucinated claim IDs                          │
│      └─ reject empty answer → safe fallback                    │
│                                                                │
│  [6] DeepEval inline scoring                                   │
│      Faithfulness · AnswerRelevancy · ContextualPrecision      │
│                                                                │
│  [7] Cache write  (MD5 key + embedding, TTL 600 s)             │
│                                                                │
│  LLM fallback chain:                                           │
│  OpenAI → Groq → HuggingFace → fastembed → Pinecone (cloud vector db)           
└────────────────────────────────────────────────────────────────┘
                     │
                     ▼
              React Frontend  (Vite)
              Dashboard · Investigate · Analytics
              Data Ingestion · System Flow · Claim Detail
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A `.env` file in the project root (see below)
- A Pinecone account and API key

### 2. Environment Setup

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

`.env` variables:

```env
# OpenAI / Org Gateway
OPENAI_API_KEY=your_key
OPENAI_BASE_URL=https://keygateway.arshnivlabs.com
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Fallback LLMs (optional)
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
HF_API_KEY=
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# Vector Store (Pinecone)
PINECONE_API_KEY=
PINECONE_INDEX_NAME=insurance-claims
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Reranker (optional)
COHERE_API_KEY=

# Data
RAW_DATA_PATH=./data/source_data/fraud_oracle.csv

# Ingestion
CHUNK_SIZE=512
CHUNK_OVERLAP=64
EMBED_BATCH_SIZE=256
INGEST_SAMPLE_N=           # optional: limit rows for testing

# Retrieval
DENSE_TOP_K=20
SPARSE_TOP_K=20
RERANK_TOP_K=5

# Guardrails
ENABLE_PII_DETECTION=false
ENABLE_CONTEXTUAL_COMPRESSION=false

# Cache
CACHE_TTL_SECONDS=600
CACHE_SIMILARITY_THRESHOLD=0.92

# App
APP_HOST=0.0.0.0
APP_PORT=8000

# LangSmith observability (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=insurance-claims-assistant
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only if ENABLE_PII_DETECTION=true
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## Data Ingestion

Run the ETL pipeline to load `fraud_oracle.csv` into Pinecone and build the BM25 index:

```bash
python -m etl.run_etl
```

Expected output:
```
[ETL] Loaded 15419 rows from fraud_oracle.csv
[ETL] Preprocessed 15419 records
[ETL] Chunked into ~15419 documents
[ETL] Embedding + storing in Pinecone...
[ETL] BM25 index built and saved.
[ETL] Ingestion complete. 15419 vectors stored.
```

You can also trigger ingestion via the API or the UI's **Data Ingestion** page.

---

## Running the Application

### Backend (FastAPI)

```bash
python -m backend.main
```

Server starts at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### Frontend (React + Vite)

```bash
cd frontend
npm run dev
```

App available at `http://localhost:5173`.

---

## API Reference

All endpoints are prefixed with `/api/v1`.

### Health Check
```
GET /api/v1/health
→ {
    "status": "ok",
    "pinecone": "connected",
    "bm25_index": "loaded",
    "llm_model": "gpt-4o-mini",
    "cache_size": 12
  }
```

### Query (RAG + Multi-Agent)
```
POST /api/v1/query
Content-Type: application/json

{
  "query": "Was claim POL-1001 fraudulent?",
  "filters": {"policy_type": "Vehicle"}
}

→ {
    "query": "Was claim POL-1001 fraudulent?",
    "answer": "...",
    "risk_score": 0.82,
    "fraud_signals": [
      {"signal": "No police report", "severity": "high"},
      {"signal": "External agent", "severity": "high"}
    ],
    "fraud_cluster": "high-risk-pattern",
    "policy_violations": [...],
    "policy_compliance": "violations_found",
    "recommendations": ["Request police report immediately", ...],
    "matched_claims": [...],
    "matched_claim_ids": ["POL-1001"],
    "model_used": "openai",
    "escalated": true,
    "cache_hit": false,
    "eval_scores": {
      "FaithfulnessMetric": 0.89,
      "AnswerRelevancyMetric": 0.92,
      "HallucinationMetric": 0.15
    },
    "error": null
  }
```

### Analytics
```
GET /api/v1/analytics
→ {
    "total_claims": 15419,
    "fraud_count": 923,
    "fraud_rate": 0.060,
    "by_region": {"Northeast": 212, "Southwest": 98, ...},
    "by_policy_type": {"Vehicle": 450, "Home": 310, ...}
  }
```

### Claim Lookup
```
GET /api/v1/claim/{claim_id}
→ {"id": "POL-1001", "text": "...", "metadata": {...}}
```

### Trigger Ingestion (by path)
```
POST /api/v1/ingest
{"csv_path": "./data/source_data/fraud_oracle.csv"}
→ {"status": "started", "message": "Ingestion pipeline triggered."}
```

### Upload and Ingest
```
POST /api/v1/ingest/upload
Content-Type: multipart/form-data
file: <CSV file>
→ {"rows_loaded": 15419, "chunks_created": 15419, "vectors_stored": 15419, ...}
```

### Ingestion Status
```
GET /api/v1/ingest/status
→ {"status": "running" | "idle" | "done" | "error", "message": "..."}
```

---

## System Flow

### Data Ingestion Pipeline (offline)

```
fraud_oracle.csv
    │
    ▼ etl/extract.py
Extract & normalize columns (15,419 rows, 33 cols → 12 metadata fields)
Synthesize incident_date from incident_month + incident_year
Add "POL-" prefix to claim IDs
    │
    ▼ etl/transform.py
Convert each row → structured text block (21 fields)
Recursive character splitter: chunk_size=512, overlap=64
Each chunk inherits metadata + chunk_index + parent_id
    │
    ▼ etl/load.py
Embed in batches (OpenAI text-embedding-3-small, 1536-dim)
Upsert to Pinecone in batches of 100 vectors
Build BM25Okapi index, pickle to ./data/bm25_index/bm25.pkl
```

### Query Pipeline (online, per request)

```
User Query (natural language)
    │
    ▼ backend/routers/query.py
Small-talk fast path → regex check → canned reply (~1ms, bypasses LLM)
    │
    ▼ Input Guardrails (backend/guardrails/input_guardrails.py)
Max length 2000 chars
Prompt injection pattern detection (regex)
Harmful / unsafe content detection (weapons, drugs, violence, malware, self-harm)
Domain relevance check (must relate to insurance-claims domain)
Optional Presidio PII anonymization (disabled by default)
    │
    ▼ Query Cache (backend/cache/query_cache.py)
Stage 1: MD5 exact match (case-insensitive)
Stage 2: Cosine similarity ≥ 0.92 → cache hit → return cached response (TTL 600s)
    │
    ▼ LangGraph Orchestrator (90s timeout)
    │
    ├── NODE 1: retrieval_agent
    │     Hybrid search: 60% Pinecone semantic + 40% BM25 keyword
    │     Merge by parent_id, CrossEncoder reranking → top-5 results
    │     Optional contextual compression (trim chunks to relevant excerpt)
    │
    ├── NODE 2: fraud_agent
    │     LLM analyzes claims for fraud signals
    │     Rule-based fallback scoring (no police report +0.25, external agent +0.20, etc.)
    │     risk_score ≥ 0.75 → set escalate=True
    │
    ├── CONDITIONAL EDGE: escalate?
    │     True  → jump to NODE 4 (skip policy check)
    │     False → continue to NODE 3
    │
    ├── NODE 3: policy_agent  (skipped if escalated)
    │     LLM checks: filing window, required docs, coverage match, policy active
    │     Returns violations with severity (high / medium / low)
    │
    └── NODE 4: recommendation_agent
          LLM synthesizes all agent outputs
          Generates 3-5 investigation actions + priority level
          Writes final answer
    │
    ▼ Output Guardrails (backend/guardrails/output_guardrails.py)
Clamp risk_score to [0.0, 1.0]
Remove hallucinated claim_ids not in retrieved context
Validate non-empty answer
    │
    ▼ DeepEval inline scoring (optional)
FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
    │
    ▼ Cache update (store result, TTL 600s)
    │
    ▼ HTTP Response → React Frontend
```

---

## Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| **Dashboard** | `/` | Stats cards (total claims, fraud count, high-risk, system status), recent high-risk claims table, fraud-by-policy-type bar chart |
| **Investigate** | `/investigate` | Natural language query input, predefined query chips, AI answer, matched claims, fraud signals, risk score, DeepEval scores, cache hit badge |
| **Analytics** | `/analytics` | Fraud stats, accident-area bar chart, policy-type pie chart, top fraud indicators table |
| **Data Ingestion** | `/ingestion` | Source path or file upload, ingestion config display, live 6-step progress log, results summary |
| **System Flow** | `/system-flow` | Interactive visual diagram of the full pipeline (ingestion + query + agents) |
| **Claim Detail** | `/claim/:id` | Full claim text and metadata fetched from `/api/v1/claim/{id}` |

---

## Sample Queries

| Query | Expected Behaviour |
|---|---|
| `"Show me high-risk vehicle claims in the Northeast"` | Retrieves claims filtered by region + policy type |
| `"Was claim POL-5042 fraudulent?"` | Direct claim lookup + fraud scoring |
| `"What patterns indicate fraud in home insurance?"` | Semantic search across all home claims |
| `"Compare fraud rates by region"` | Returns analytics breakdown |
| `"List claims with missing police reports"` | Keyword + semantic hybrid retrieval |

---

## Evaluation

Run DeepEval metrics (Faithfulness, AnswerRelevancy, Hallucination) in batch mode:

```bash
python -m backend.evaluation.evaluator --samples 3
```

Save results to JSON:

```bash
python -m backend.evaluation.evaluator --samples 10 --output eval_results.json
```

Scores are also computed inline on every `/api/v1/query` request and returned in the `eval_scores` field.

---

## Project Structure

```
AI-Powered-Insurance-Claims-Intelligence-Assistant/
├── backend/
│   ├── agents/
│   │   ├── orchestrator.py       # LangGraph StateGraph (4 nodes, A2A escalation)
│   │   ├── state.py              # ClaimsState schema
│   │   ├── retrieval_agent.py    # Hybrid search + contextual compression
│   │   ├── fraud_agent.py        # Risk scoring + fraud signal detection
│   │   ├── policy_agent.py       # Policy compliance checks
│   │   └── recommendation_agent.py  # Final synthesis + action items
│   ├── cache/
│   │   └── query_cache.py        # In-memory cache (MD5 + cosine, TTL 600s)
│   ├── config/
│   │   ├── settings.py           # Pydantic-settings from .env
│   │   └── llm.py                # LLM + embedding factory with fallback chains
│   ├── evaluation/
│   │   └── evaluator.py          # DeepEval batch runner
│   ├── guardrails/
│   │   ├── input_guardrails.py   # Injection detection + optional PII (Presidio)
│   │   └── output_guardrails.py  # Risk bounds + hallucination + non-empty check
│   ├── retrieval/
│   │   ├── hybrid_search.py      # Fuses Pinecone + BM25, calls reranker
│   │   ├── vector_store.py       # Pinecone client wrapper
│   │   ├── bm25_search.py        # BM25Okapi index (pickled)
│   │   ├── reranker.py           # Cohere → cosine fallback reranking
│   │   └── compressor.py         # Optional LLM contextual compression
│   ├── routers/
│   │   ├── query.py              # POST /api/v1/query
│   │   ├── ingest.py             # POST /api/v1/ingest, POST /upload, GET /status
│   │   ├── analytics.py          # GET /api/v1/analytics, GET /claim/{id}
│   │   └── health.py             # GET /api/v1/health
│   ├── models.py                 # Pydantic request/response schemas
│   └── main.py                   # FastAPI app entry point + CORS
├── etl/
│   ├── extract.py                # Load CSV, column mapping, date synthesis
│   ├── transform.py              # row_to_text(), build_metadata(), chunk_texts()
│   ├── load.py                   # embed_and_store() → Pinecone, build_bm25_index()
│   └── run_etl.py                # Orchestrates extract → transform → load
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Investigate.jsx
│   │   │   ├── Analytics.jsx
│   │   │   ├── Ingestion.jsx
│   │   │   ├── SystemFlow.jsx
│   │   │   └── ClaimDetail.jsx
│   │   ├── components/
│   │   │   └── Layout.jsx        # Sidebar nav + header
│   │   ├── hooks/
│   │   │   └── usePersistedState.js  # localStorage state persistence
│   │   ├── api.js                # Axios clients (120s query, 10s quick, 600s upload)
│   │   └── styles/global.css
│   └── package.json
├── data/
│   ├── source_data/              # fraud_oracle.csv (15,419 rows)
│   ├── bm25_index/               # bm25.pkl (auto-created by ETL)
│   └── chroma/                   # (legacy, replaced by Pinecone)
├── docs/                         # Instruction documents
├── .env                          # Local secrets (never commit)
├── .env.example                  # Template
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

| Feature | Implementation |
|---------|----------------|
| **Hybrid search** | 60% Pinecone semantic + 40% BM25 keyword, merged by `parent_id`, reranked by CrossEncoder |
| **Fallback chains** | LLM: OpenAI → Groq; Embeddings: OpenAI → HuggingFace → fastembed → numpy hash |
| **Multi-agent graph** | LangGraph `StateGraph`, 4 nodes, shared `ClaimsState` passed between nodes |
| **A2A escalation** | `fraud_score ≥ 0.75` → skip `policy_agent`, jump directly to `recommendation_agent` |
| **Two-stage cache** | MD5 exact match first; cosine similarity ≥ 0.92 as semantic near-duplicate fallback |
| **Query timeout** | 90-second hard ceiling per `/query` request via `asyncio.wait_for` |
| **Contextual compression** | Optional LLM-based chunk trimming (~40% token reduction), disabled by default |
| **Guardrails** | Input: injection detection + harmful content blocking + domain relevance check + optional Presidio PII; Output: risk bounds + hallucination check + non-empty answer |
| **Inline evaluation** | DeepEval Faithfulness, AnswerRelevancy, Hallucination scores returned in every response |
| **Observability** | LangSmith full trace (optional) + per-node timing logs |
| **Small-talk fast path** | Regex pattern matching for greetings — bypasses LLM (~1ms response) |
