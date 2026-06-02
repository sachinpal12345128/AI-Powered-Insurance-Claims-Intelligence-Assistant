# Project State — AI-Powered Insurance Claims Intelligence Assistant

## Status: COMPLETE ✅
All 18 tasks done. All 3 instruction-doc gaps closed. Ready for local testing.

---

## Project Structure

```
AI-Powered-Insurance-Claims-Intelligence-Assistant/
├── backend/
│   ├── agents/         # LangGraph: retrieval, fraud, policy, recommendation, orchestrator
│   ├── cache/          # In-memory QueryCache (MD5 exact + cosine similarity, TTL 600s)
│   ├── config/         # settings.py (pydantic-settings), llm.py (LLM+embedding factory)
│   ├── evaluation/     # evaluator.py — DeepEval metrics runner
│   ├── guardrails/     # input (Presidio PII + injection), output (hallucination + bounds)
│   ├── retrieval/      # vector_store, bm25_search, hybrid_search, reranker, compressor
│   ├── routers/        # query, ingest, analytics, health
│   ├── main.py         # FastAPI entry point
│   └── models.py       # Pydantic request/response models
├── etl/                # extract.py, transform.py, load.py, run_etl.py
├── frontend/           # Vite + React (Dashboard, Investigate, ClaimDetail, Analytics, Ingestion, SystemFlow)
├── data/
│   ├── raw/            # fraud_oracle.csv (15,419 rows, 33 cols, ~6% fraud)
│   ├── chroma_db/      # ChromaDB persistence (created at runtime)
│   └── bm25_index/     # BM25 index (created at runtime)
├── docs/               # Instruction documents
├── .env                # Secrets — never commit
├── requirements.txt
└── README.md           # Full setup + API reference
```

---

## Key Design

| Component | Detail |
|---|---|
| LLM fallback | OpenAI (gpt-4o-mini) → Groq → HuggingFace |
| Embedding fallback | OpenAI text-embedding-3-small → MiniLM local |
| Retrieval | Hybrid: ChromaDB semantic (0.6) + BM25 keyword (0.4), reranked by CrossEncoder |
| Contextual compression | LLM trims each chunk to query-relevant excerpt before agents see it |
| Multi-agent | LangGraph StateGraph: retrieval → fraud → (A2A escalation at score >0.75) → recommendation → policy |
| Cache | MD5 exact match first; cosine similarity fallback (threshold 0.92, TTL 600s) |
| Guardrails | Input: Presidio PII anonymisation + prompt injection. Output: risk bounds + hallucination check |
| Evaluation | DeepEval: Faithfulness, AnswerRelevancy, ContextualPrecision |

---

## Org Gateway Config (.env)
```env
OPENAI_API_KEY=learner027
OPENAI_BASE_URL=https://keygateway.arshnivlabs.com
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIR=./data/chroma_db
CHROMA_COLLECTION=insurance_claims
BM25_INDEX_PATH=./data/bm25_index/bm25.pkl
```

---

## How to Run

```bash
# Backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m backend.main          # → http://localhost:8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev   # → http://localhost:5173

# Ingest data (after backend is up)
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"csv_path": "./data/raw/fraud_oracle.csv"}'

# Run evaluation
python -m backend.evaluation.evaluator --samples 3
```

---

## API Endpoints
- `GET  /health`
- `POST /query`       — body: `{"query": "...", "top_k": 5, "filters": {}}`
- `POST /ingest`      — body: `{"csv_path": "..."}`
- `POST /ingest/upload` — multipart file upload
- `GET  /analytics`
- `GET  /claim/{claim_id}`

---

## Critical Notes for Next Chat
- **Never hardcode API keys** — always read from `.env` via `settings`
- **File writes**: Use bash heredoc (`cat > file << 'PYEOF'`) NOT Edit/Write tools — mounted filesystem truncates last ~3 lines with those tools
- **ChromaDB in sandbox**: Can't write to mounted path — use `CHROMA_PERSIST_DIR=/tmp/chroma_test` for testing in Claude's shell
- **torch not installable in sandbox** (2GB, 45s timeout) — sentence-transformers won't load there; works fine locally
- **Old `src/` folder**: Superseded by `backend/` + `etl/`. Ignore it.
- **Frontend port**: Vite runs on `5173`, not `3000`
- **Ingestion router**: at `backend/routers/ingest.py`, calls `etl.run_etl.run_etl()`

---

## Completed Tasks
1. Project scaffold — .env, requirements.txt, settings.py
2. Ingestion ETL pipeline (etl/)
3. Retrieval — vector store, BM25, hybrid search, reranker
4. Cache + guardrails
5. LangGraph multi-agent system
6. FastAPI routers
7. React frontend (5 pages)
8-10. Restructure + audit + cleanup
11-15. Smoke tests (imports, backend, ETL, API, frontend)
16. ContextualCompression → backend/retrieval/compressor.py
17. DeepEval evaluation → backend/evaluation/evaluator.py
18. README.md (258 lines)
