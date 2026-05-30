"""
Hybrid search: weighted combination of semantic (ChromaDB) + keyword (BM25),
followed by CrossEncoder / Cohere reranking.
"""
import logging
from backend.retrieval.vector_store import semantic_search
from backend.retrieval.bm25_search import bm25_search
from backend.retrieval.reranker import rerank

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.6
BM25_WEIGHT = 0.4


def hybrid_search(
    query: str,
    n_retrieve: int = 20,
    top_k: int = 5,
    filters: dict = None,
) -> list[dict]:
    """
    1. Semantic search via ChromaDB
    2. BM25 keyword search
    3. Score fusion (weighted)
    4. Rerank top candidates
    Returns top_k results with merged scores.
    """
    semantic_results = semantic_search(query, n_results=n_retrieve, where=filters)
    bm25_results = bm25_search(query, n_results=n_retrieve)

    # Normalize scores to [0, 1]
    sem_scores = {r["id"]: r["score"] for r in semantic_results}
    bm25_max = max((r["score"] for r in bm25_results), default=1)
    bm25_scores = {r["id"]: r["score"] / max(bm25_max, 1e-9) for r in bm25_results}

    # Build unified candidate pool keyed by parent_id (strip _chunk_N suffix)
    pool: dict[str, dict] = {}

    for r in semantic_results:
        pid = r["metadata"].get("parent_id", r["id"])
        if pid not in pool or pool[pid]["hybrid_score"] < r["score"] * SEMANTIC_WEIGHT:
            pool[pid] = {**r, "hybrid_score": round(r["score"] * SEMANTIC_WEIGHT, 4)}

    for r in bm25_results:
        pid = r["id"]
        bm25_contrib = bm25_scores.get(pid, 0) * BM25_WEIGHT
        if pid in pool:
            pool[pid]["hybrid_score"] = round(pool[pid]["hybrid_score"] + bm25_contrib, 4)
        else:
            pool[pid] = {**r, "hybrid_score": round(bm25_contrib, 4)}

    candidates = sorted(pool.values(), key=lambda x: x["hybrid_score"], reverse=True)[:n_retrieve]

    # Rerank
    reranked = rerank(query, candidates, top_k=top_k)

    logger.info(f"hybrid_search: {len(semantic_results)} semantic + {len(bm25_results)} BM25 → {len(reranked)} reranked results")
    return reranked
