"""Load - embed into ChromaDB and build BM25 index."""
import pickle
import logging
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback
from backend.config.paths import chroma_dir, bm25_path

logger = logging.getLogger(__name__)
settings = get_settings()


def get_chroma_client() -> chromadb.PersistentClient:
    p = chroma_dir()
    Path(p).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=p)


def embed_and_store(texts, metadatas, ids, on_progress=None) -> dict:
    from backend.config.llm import embed_texts as _embed_texts
    _, model_name = get_embeddings_with_fallback()
    client = get_chroma_client()

    try:
        client.delete_collection(settings.chroma_collection)
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.chroma_collection,
        metadata={
            "embedding_model": model_name,
            "hnsw:space": "cosine",
            "hnsw:batch_size": 4096,
            "hnsw:sync_threshold": 8000,
        },
    )

    total = len(texts)
    t_start = time.time()

    # Embed entire dataset in one vectorized call (fast for numpy hash)
    logger.info(f"Embedding {total} chunks...")
    t = time.time()
    all_embeddings = _embed_texts(texts)
    logger.info(f"Embedding done in {time.time()-t:.1f}s")

    # Insert into ChromaDB in batches (Chroma has a ~5000 item limit per add call)
    batch_size = 1000
    n_batches = (total + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, total, batch_size), 1):
        end = min(start + batch_size, total)
        t = time.time()
        collection.add(
            embeddings=all_embeddings[start:end],
            documents=texts[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )
        logger.info(
            f"  stored batch {i}/{n_batches}  rows {start}-{end}  "
            f"store={time.time()-t:.1f}s  elapsed={time.time()-t_start:.1f}s"
        )
        if on_progress:
            on_progress(end, total)

    logger.info(f"Stored {total} vectors in ChromaDB '{settings.chroma_collection}' in {time.time()-t_start:.1f}s.")
    return {"collection": settings.chroma_collection, "total_vectors": total, "embedding_model": model_name}


def build_bm25_index(texts, ids) -> None:
    t = time.time()
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    index_path = Path(bm25_path())
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "texts": texts}, f)
    logger.info(f"BM25 index built ({len(texts)} docs) in {time.time()-t:.1f}s -> {index_path}")


def load_bm25_index() -> dict:
    index_path = Path(bm25_path())
    if not index_path.exists():
        raise FileNotFoundError(f"BM25 index not found at {index_path}. Run ETL first.")
    with open(index_path, "rb") as f:
        return pickle.load(f)
