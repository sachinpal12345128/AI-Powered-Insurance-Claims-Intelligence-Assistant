"""
Reranker — cosine similarity (primary, no torch required) + Cohere (optional).
Pattern adopted from AI-Powered-Smart-Grid-Energy-Intelligence-Assistant reference project.
"""
from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _cosine_rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Rerank using cosine similarity between query embedding and doc embeddings."""
    from backend.config.llm import embed_query as _embed_query
    q_emb = _embed_query(query)
    for doc in candidates:
        doc_emb = doc.get("embedding")
        if doc_emb:
            doc["rerank_score"] = _cosine(q_emb, doc_emb)
        else:
            # Fall back to existing hybrid/semantic score
            doc["rerank_score"] = float(doc.get("hybrid_score", doc.get("score", 0.0)))
    return sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)[:top_k]


def _cohere_rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    import cohere
    co = cohere.Client(settings.cohere_api_key)
    docs = [c["text"] for c in candidates]
    response = co.rerank(query=query, documents=docs, top_n=top_k, model="rerank-english-v3.0")
    reranked = []
    for r in response.results:
        item = candidates[r.index].copy()
        item["rerank_score"] = round(r.relevance_score, 4)
        reranked.append(item)
    return reranked


def _passthrough(candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Sort by existing score, set rerank_score, return top_k."""
    sorted_c = sorted(
        candidates,
        key=lambda c: c.get("hybrid_score", c.get("score", 0.0)),
        reverse=True,
    )
    for item in sorted_c:
        item.setdefault("rerank_score", item.get("hybrid_score", item.get("score", 0.0)))
    return sorted_c[:top_k]


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank candidates. Priority:
      1. Cohere API (if cohere_api_key set)
      2. Cosine similarity via fastembed (no torch)
      3. Passthrough — sort by existing score
    """
    if not candidates:
        return []

    # 1. Cohere
    if settings.cohere_api_key:
        try:
            return _cohere_rerank(query, candidates, top_k)
        except Exception as e:
            logger.warning(f"[reranker] Cohere failed ({e!r}), falling back to cosine.")

    # 2. Cosine similarity
    try:
        return _cosine_rerank(query, candidates, top_k)
    except Exception as e:
        logger.warning(f"[reranker] Cosine rerank failed ({e!r}), using passthrough.")

    # 3. Passthrough
    return _passthrough(candidates, top_k)
