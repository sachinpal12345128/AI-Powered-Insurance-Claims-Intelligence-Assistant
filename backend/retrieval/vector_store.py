import logging
from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback, EMBED_DIM

logger = logging.getLogger(__name__)
settings = get_settings()

_index = None
_embeddings = None
_embeddings_ready = False


def _get_index():
    global _index
    if _index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key, ssl_verify=False)
        _index = pc.Index(settings.pinecone_index_name)
        logger.info(f"Pinecone index '{settings.pinecone_index_name}' connected.")
    return _index


def _init():
    global _embeddings, _embeddings_ready
    _get_index()
    if not _embeddings_ready:
        _embeddings, _ = get_embeddings_with_fallback()
        _embeddings_ready = True


def invalidate_vector_store_cache() -> None:
    global _index, _embeddings, _embeddings_ready
    _index = None
    _embeddings = None
    _embeddings_ready = False


def semantic_search(query: str, n_results: int = 10, where: dict = None) -> list:
    """Returns list of {id, text, metadata, score} sorted by cosine similarity."""
    _init()
    query_embedding = _embeddings.embed_query(query)
    kwargs = {
        "vector": query_embedding,
        "top_k": n_results,
        "include_metadata": True,
    }
    if where:
        kwargs["filter"] = where
    results = _get_index().query(**kwargs)
    return [
        {
            "id": match.id,
            "text": match.metadata.get("text", ""),
            "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
            "score": round(match.score, 4),
        }
        for match in results.matches
    ]


def get_claim_by_id(claim_id: str):
    """Fetch a single claim by claim_id metadata field."""
    _get_index()
    results = _get_index().query(
        vector=[0.0] * EMBED_DIM,
        top_k=1,
        filter={"claim_id": {"$eq": claim_id}},
        include_metadata=True,
    )
    if not results.matches:
        return None
    match = results.matches[0]
    return {
        "id": match.id,
        "text": match.metadata.get("text", ""),
        "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
    }
