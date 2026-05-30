import logging
import chromadb
from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback

logger = logging.getLogger(__name__)
settings = get_settings()

_client = None
_collection = None
_embeddings = None


def _init():
    global _client, _collection, _embeddings
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_collection(settings.chroma_collection)
        _embeddings, _ = get_embeddings_with_fallback()
        logger.info("ChromaDB collection loaded.")


def semantic_search(
    query: str,
    n_results: int = 10,
    where: dict = None,
) -> list[dict]:
    """
    Returns list of {id, text, metadata, score} sorted by cosine similarity.
    score is distance (lower = more similar). We convert to similarity = 1 - distance.
    """
    _init()
    query_embedding = _embeddings.embed_query(query)
    kwargs = {"query_embeddings": [query_embedding], "n_results": n_results, "include": ["documents", "metadatas", "distances"]}
    if where:
        kwargs["where"] = where

    results = _collection.query(**kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    return [
        {
            "id": ids[i],
            "text": docs[i],
            "metadata": metas[i],
            "score": round(1 - distances[i], 4),  # cosine similarity
        }
        for i in range(len(docs))
    ]


def get_claim_by_id(claim_id: str) -> dict | None:
    """Fetch a single claim by claim_id metadata field."""
    _init()
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
