"""Load - embed into Pinecone and build BM25 index."""
import pickle
import logging
import time
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.config.settings import get_settings
from backend.config.llm import get_embeddings_with_fallback, EMBED_DIM
from backend.config.paths import bm25_path

logger = logging.getLogger(__name__)
settings = get_settings()


def _get_or_create_index():
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=settings.pinecone_api_key, ssl_verify=False)
    existing = {idx.name for idx in pc.list_indexes()}

    if settings.pinecone_index_name in existing:
        info = pc.describe_index(settings.pinecone_index_name)
        if info.dimension != EMBED_DIM:
            logger.info(f"Index dim mismatch ({info.dimension} vs {EMBED_DIM}). Deleting and recreating...")
            pc.delete_index(settings.pinecone_index_name)
            existing.discard(settings.pinecone_index_name)

    if settings.pinecone_index_name not in existing:
        logger.info(f"Creating Pinecone index '{settings.pinecone_index_name}' (dim={EMBED_DIM})...")
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
        logger.info("Pinecone index ready.")

    return pc.Index(settings.pinecone_index_name)


def embed_and_store(texts, metadatas, ids, on_progress=None) -> dict:
    from backend.config.llm import embed_texts as _embed_texts
    _, model_name = get_embeddings_with_fallback()

    index = _get_or_create_index()

    # Delete existing vectors so re-ingest starts clean
    try:
        index.delete(delete_all=True)
        logger.info("Cleared existing Pinecone vectors.")
    except Exception:
        pass

    total = len(texts)
    t_start = time.time()

    logger.info(f"Embedding {total} chunks...")
    t = time.time()
    all_embeddings = _embed_texts(texts)
    logger.info(f"Embedding done in {time.time()-t:.1f}s")

    # Pinecone recommends batches of 100 vectors
    batch_size = 100
    n_batches = (total + batch_size - 1) // batch_size
    for i, start in enumerate(range(0, total, batch_size), 1):
        end = min(start + batch_size, total)
        vectors = [
            {
                "id": ids[j],
                "values": all_embeddings[j],
                "metadata": {**metadatas[j], "text": texts[j]},
            }
            for j in range(start, end)
        ]
        t = time.time()
        index.upsert(vectors=vectors)
        logger.info(
            f"  stored batch {i}/{n_batches}  rows {start}-{end}  "
            f"store={time.time()-t:.1f}s  elapsed={time.time()-t_start:.1f}s"
        )
        if on_progress:
            on_progress(end, total)

    logger.info(f"Stored {total} vectors in Pinecone '{settings.pinecone_index_name}' in {time.time()-t_start:.1f}s.")
    return {"collection": settings.pinecone_index_name, "total_vectors": total, "embedding_model": model_name}


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
