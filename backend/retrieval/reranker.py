"""
Reranker with fallback:
Primary  — Cohere Rerank API
Fallback — sentence-transformers CrossEncoder (local)
"""
import logging
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info("CrossEncoder loaded.")
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank candidates using Cohere (primary) or CrossEncoder (fallback).
    Each candidate must have 'text' key.
    Returns top_k candidates with updated 'rerank_score'.
    """
    if not candidates:
        return []

    # Try Cohere first
    if settings.cohere_api_key:
        try:
            return _cohere_rerank(query, candidates, top_k)
        except Exception as e:
            logger.warning(f"Cohere rerank failed ({e}). Falling back to CrossEncoder.")

    return _crossencoder_rerank(query, candidates, top_k)


def _cohere_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
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


def _crossencoder_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    model = _get_cross_encoder()
    pairs = [(query, c["text"]) for c in candidates]
    scores = model.predict(pairs)
    scored = [(s, c) for s, c in zip(scores, candidates)]
    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, item in scored[:top_k]:
        c = item.copy()
        c["rerank_score"] = round(float(score), 4)
        results.append(c)
    return results
