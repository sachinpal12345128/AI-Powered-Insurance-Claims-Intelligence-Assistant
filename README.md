# AI-Powered Insurance Claims Intelligence Assistant

A Retrieval-Augmented Generation (RAG) system for intelligent insurance claims analysis and fraud detection. Built with LangGraph multi-agent orchestration, hybrid semantic + keyword search, and a React dashboard.

---

## Architecture Overview

```
fraud_oracle.csv
      │
      ▼
┌─────────────────────────────────┐
│  ETL Pipeline (offline)         │
│  Loader → Preprocessor →        │
│  Chunker → Embedder → ChromaDB  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────┐
│  Query Pipeline (online)                            │
│  Input Guardrails → Cache →                         │
│  LangGraph Orchestrator:                            │
│    retrieval_agent  (hybrid search + compression)   │
│    fraud_agent      (risk scoring)                  │
│    recommendation_agent (A2A escalation >0.75)      │
│    compliance_agent (policy checks)                 │
│  → Output Guardrails → Response                     │
└─────────────────────────────────────────────────────┘
             │
             ▼
     FastAPI  ←→  React Frontend
```

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Node.js 18+
- A `.env` file in the project root (see below)

### 2. Environment Setup

Copy the template and fill in your keys:

```bash
cp .env.example .env
```

`.env` variables:

```env
# OpenAI / Org Gateway
OPENAI_API_KEY=learner027
OPENAI_BASE_URL=https://keygateway.arshnivlabs.com
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Fallback LLMs (optional)
GROQ_API_KEY=
GROQ_MODEL=llama3-8b-8192
HF_API_KEY=
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=insurance_claims

# Data
RAW_DATA_PATH=./data/raw/fraud_oracle.csv

# LangSmith observability (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # for Presidio PII guardrails
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
```

---

## Data Ingestion

Run the ETL pipeline to load `fraud_oracle.csv` into ChromaDB:

```bash
python -m backend.ingestion.pipeline
```

Expected output:
```
[pipeline] Loaded 15419 rows from fraud_oracle.csv
[pipeline] Preprocessed 15419 records
[pipeline] Chunked into 15419 documents
[pipeline] Embedding + storing in ChromaDB...
[pipeline] Ingestion complete. 15419 documents stored.
```

---

## Running the Application

### Backend (FastAPI)

```bash
python -m backend.main
```

Server starts at `http://localhost:8000`.

### Frontend (React + Vite)

```bash
cd frontend
npm run dev
```

App available at `http://localhost:5173`.

---

## API Reference

### Health Check
```
GET /health
→ {"status": "ok", "version": "1.0.0"}
```

### Query (RAG + Multi-Agent)
```
POST /query
Content-Type: application/json

{
  "query": "Was claim C-1001 fraudulent?",
  "top_k": 5,
  "filters": {"policy_type": "Vehicle"}
}

→ {
  "answer": "...",
  "fraud_score": 0.82,
  "risk_level": "HIGH",
  "matched_claim_ids": ["C-1001"],
  "sources": [...],
  "cached": false
}
```

### Analytics
```
GET /analytics
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
GET /claim/{claim_id}
→ {"id": "...", "text": "...", "metadata": {...}}
```

### Trigger Ingestion
```
POST /ingest
→ {"status": "started", "message": "Ingestion pipeline triggered."}
```

---

## Sample Queries

| Query | Expected Behaviour |
|---|---|
| `"Show me high-risk vehicle claims in the Northeast"` | Retrieves claims filtered by region + policy type |
| `"Was claim C-5042 fraudulent?"` | Direct claim lookup + fraud scoring |
| `"What patterns indicate fraud in home insurance?"` | Semantic search across all home claims |
| `"Compare fraud rates by region"` | Returns analytics breakdown |
| `"List claims with missing police reports"` | Keyword + semantic hybrid retrieval |

---

## Evaluation

Run DeepEval metrics (Faithfulness, AnswerRelevancy, ContextualPrecision):

```bash
python -m backend.evaluation.evaluator --samples 3
```

Save results to JSON:

```bash
python -m backend.evaluation.evaluator --samples 10 --output eval_results.json
```

---

## Project Structure

```
AI-Powered-Insurance-Claims-Intelligence-Assistant/
├── backend/
│   ├── agents/           # LangGraph agents (fraud, compliance, recommendation, retrieval)
│   ├── config/           # Settings, LLM factory, logging
│   ├── evaluation/       # DeepEval metrics runner
│   ├── guardrails/       # Input PII + injection; output risk bounds + hallucination
│   ├── ingestion/        # ETL: loader, preprocessor, chunker, embedder, pipeline
│   ├── retrieval/        # ChromaDB vector store, BM25, hybrid search, reranker, compressor
│   ├── routers/          # FastAPI routers: query, ingest, analytics, health
│   ├── cache.py          # In-memory QueryCache (MD5 + cosine similarity, TTL 600s)
│   └── main.py           # FastAPI app entry point
├── frontend/
│   ├── src/
│   │   ├── pages/        # Dashboard, Query, Claims, Analytics, Settings
│   │   ├── components/   # Shared UI components
│   │   └── api/          # Axios API client
│   └── package.json
├── data/
│   ├── raw/              # fraud_oracle.csv (source data)
│   └── chroma/           # ChromaDB persistent store (auto-created)
├── docs/                 # Instruction documents
├── .env                  # Local secrets (never commit)
├── .env.example          # Template
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

- **Hybrid search**: ChromaDB semantic (weight 0.6) + BM25 keyword (weight 0.4), reranked by CrossEncoder
- **Contextual compression**: Each retrieved chunk trimmed to query-relevant excerpt via LLM before passing to agents
- **A2A escalation**: `fraud_agent` automatically escalates to `recommendation_agent` when fraud score > 0.75
- **Fallback chain**: OpenAI → Groq → HuggingFace for LLM; OpenAI → MiniLM for embeddings
- **Cache**: MD5 exact match first; cosine similarity fallback (threshold 0.92, TTL 600s)
- **Guardrails**: Presidio PII anonymisation + prompt injection detection on input; risk bound + hallucination check on output
