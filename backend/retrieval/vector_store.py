import logging
import chromadb
from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback
from backend.config.paths import chroma_dir

logger = logging.getLogger(__name__)
settings = get_settings()

_client = None
_collection = None
_embeddings = None
_embeddings_ready = False


def _init_collection():
    """Init ChromaDB client + collection (no embeddings needed)."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=chroma_dir())
        _collection = _client.get_collection(settings.chroma_collection)
        logger.info("ChromaDB collection loaded.")


def _init():
    """Init collection + embedding model for semantic search."""
    global _embeddings, _embeddings_ready
    _init_collection()
    if not _embeddings_ready:
        _embeddings, _ = get_embeddings_with_fallback()
        _embeddings_ready = True


def invalidate_vector_store_cache() -> None:
    """Force re-load of the Chroma collection on next query. Call after ETL re-ingest."""
    global _client, _collection, _embeddings, _embeddings_ready
    _client = None
    _collection = None
    _embeddings = None
    _embeddings_ready = False


def semantic_search(query: str, n_results: int = 10, where: dict = None) -> list:
    """Returns list of {id, text, metadata, score} sorted by cosine similarity."""
    _init()
    if _embeddings is None:
        raise RuntimeError("Embedding model unavailable.")
    query_embedding = _embeddings.embed_query(query)
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where
    results = _collection.query(**kwargs)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]
    return [
        {"id": ids[i], "text": docs[i], "metadata": metas[i], "score": round(1 - distances[i], 4)}
        for i in range(len(docs))
    ]


def get_claim_by_id(claim_id: str):
    """Fetch a single claim by claim_id metadata field (no embeddings needed)."""
    _init_collection()
    results = _collection.get(
        where={"claim_id": claim_id},
        include=["documents", "metadatas"],
        limit=1,
    )
    if not results["ids"]:
        return None
    return {
        "id": results["ids"][0],
        "text": results["documents"][0],
        "metadata": results["metadatas"][0],
    }
