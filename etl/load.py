"""Load — embed into ChromaDB and build BM25 index."""
import pickle
import logging
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback

logger = logging.getLogger(__name__)
settings = get_settings()


# ── ChromaDB ──────────────────────────────────────────────────────────────────

def get_chroma_client() -> chromadb.PersistentClient:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=settings.chroma_persist_dir)


def embed_and_store(
    texts: list[str],
    metadatas: list[dict],
    ids: list[str],
    on_progress=None,
) -> dict:
    embeddings_model, model_name = get_embeddings_with_fallback()
    client = get_chroma_client()

    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.chroma_collection,
        metadata={"embedding_model": model_name, "hnsw:space": "cosine"},
    )

    batch_size = settings.embed_batch_size
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_embeddings = embeddings_model.embed_documents(texts[start:end])
        collection.add(
            embeddings=batch_embeddings,
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        if on_progress:
            on_progress(end, total)
        logger.debug(f"Stored batch {start}–{end}")

    logger.info(f"Stored {total} vectors in ChromaDB '{settings.chroma_collection}'.")
    return {"collection": settings.chroma_collection, "total_vectors": total, "embedding_model": model_name}


# ── BM25 ──────────────────────────────────────────────────────────────────────

def build_bm25_index(texts: list[str], ids: list[str]) -> None:
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    index_path = Path(settings.bm25_index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts}, f)
    logger.info(f"BM25 index built → {index_path}")


def load_bm25_index() -> dict:
    index_path = Path(settings.bm25_index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run ETL first.")
    with open(index_path, "rb") as f:
        return pickle.load(f)
