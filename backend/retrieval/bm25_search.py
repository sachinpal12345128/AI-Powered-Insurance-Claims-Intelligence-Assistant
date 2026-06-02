import logging
import numpy as np
from etl.load import load_bm25_index

logger = logging.getLogger(__name__)

_index_cache = None


def _get_index():
    global _index_cache
    if _index_cache is None:
        _index_cache = load_bm25_index()
    return _index_cache


def invalidate_bm25_cache() -> None:
    """Force re-load of the BM25 pickle on next search. Call after ETL re-ingest."""
    global _index_cache
    _index_cache = None


def bm25_search(query: str, n_results: int = 10) -> list[dict]:
    """Returns top-n results by BM25 score."""
    idx = _get_index()
    bm25 = idx["bm25"]
    ids = idx["ids"]
    texts = idx["texts"]

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:n_results]

    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append({
                "id": ids[i],
                "text": texts[i],
                "score": round(float(scores[i]), 4),
                "metadata": {},  # BM25 doesn't store metadata; filled in hybrid merge
            })
    return results
