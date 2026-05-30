"""
In-memory semantic cache.
Exact-match via MD5 hash + near-duplicate detection via cosine similarity.
"""
import hashlib
import time
import logging
import numpy as np
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def _cosine(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


class QueryCache:
    def __init__(self):
        self._store: dict[str, dict] = {}   # md5 → {response, embedding, ts}
        self._embeddings_list: list[tuple[str, list[float]]] = []  # (md5, embedding)

    def _md5(self, query: str) -> str:
        return hashlib.md5(query.strip().lower().encode()).hexdigest()

    def _is_expired(self, entry: dict) -> bool:
        return (time.time() - entry["ts"]) > settings.cache_ttl_seconds

    def get(self, query: str, query_embedding: list[float] = None) -> dict | None:
        key = self._md5(query)

        # Exact match
        if key in self._store:
            entry = self._store[key]
            if not self._is_expired(entry):
                logger.info(f"Cache exact hit for query hash {key[:8]}")
                return entry["response"]
            else:
                del self._store[key]

        # Semantic near-duplicate match
        if query_embedding is not None:
            for md5, emb in self._embeddings_list:
                if md5 not in self._store:
                    continue
                entry = self._store[md5]
                if self._is_expired(entry):
                    continue
                sim = _cosine(query_embedding, emb)
                if sim >= settings.cache_similarity_threshold:
                    logger.info(f"Cache semantic hit (similarity={sim:.3f})")
                    return entry["response"]

        return None

    def set(self, query: str, response: dict, query_embedding: list[float] = None):
        key = self._md5(query)
        self._store[key] = {"response": response, "embedding": query_embedding, "ts": time.time()}
        if query_embedding is not None:
            self._embeddings_list.append((key, query_embedding))
        logger.debug(f"Cached response for query hash {key[:8]}")

    def size(self) -> int:
        return len(self._store)

    def clear(self):
        self._store.clear()
        self._embeddings_list.clear()


# Singleton
_cache = QueryCache()

def get_cache() -> QueryCache:
    return _cache
