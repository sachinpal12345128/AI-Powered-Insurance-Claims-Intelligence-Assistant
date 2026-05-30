# Project State

## Structure (restructured to match reference project)
- `backend/` — FastAPI app, agents, retrieval, cache, guardrails, config
- `etl/` — extract, transform, load pipeline (replaces old src/ingestion/)
- `datasets/` — fraud_oracle.csv source data
- `frontend/` — Vite + React, 6 pages
- `data/chroma_db/` — ChromaDB persistence (created at runtime)
- `data/bm25_index/bm25.pkl` — BM25 index (created at runtime)
- `src/` — OLD structure, superseded by backend/ + etl/. Ignore.

## Completed
- backend/config/settings.py — pydantic-settings, all env vars
- backend/config/llm.py — LLM + embedding factory (OpenAI → Groq → HuggingFace fallback)
- etl/extract.py, etl/transform.py, etl/load.py, etl/run_etl.py
- backend/retrieval/ — vector_store, bm25_search, reranker, hybrid_search
- backend/cache/query_cache.py — in-memory cache (MD5 + cosine)
- backend/guardrails/ — input (Presidio + injection) + output (hallucination + bounds)
- backend/agents/ — state, retrieval, fraud, policy, recommendation, orchestrator (LangGraph)
- backend/routers/ — query, ingest, analytics, health (split routers)
- backend/main.py — FastAPI app, mounts all routers
- frontend/ — Dashboard, Investigate, ClaimDetail, Analytics, Ingestion, SystemFlow pages
- frontend/src/api.js — flat axios wrapper (queryAPI, ingestAPI, analyticsAPI, healthAPI)

## How to run
1. Copy `.env.example` to `.env` and fill API keys
2. Backend: `pip install -r requirements.txt` → `python -m backend.main`
3. Frontend: `cd frontend` → `npm install` → `npm run dev`
4. Ingest: http://localhost:3000/ingestion → Start ingestion
5. Query: http://localhost:3000/investigate

## Key config (.env)
- OPENAI_API_KEY=learner027
- OPENAI_BASE_URL=https://keygateway.arshnivlabs.com
- CHROMA_PERSIST_DIR=./data/chroma_db
- BM25_INDEX_PATH=./data/bm25_index/bm25.pkl
- CSV_PATH=./datasets/fraud_oracle.csv
- LANGCHAIN_TRACING_V2=true (LangSmith)

## Next steps
- Add .env.example with placeholder values
- Test backend startup: python -m backend.main
- Run ingestion end-to-end and verify ChromaDB populated
- Test /investigate page with a sample query
