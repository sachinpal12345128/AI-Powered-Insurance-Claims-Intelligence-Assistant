from fastapi import APIRouter
from backend.models import HealthResponse
from backend.cache.query_cache import get_cache
from backend.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health():
    chroma_ok = False
    bm25_ok = False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.get_collection(settings.chroma_collection)
        chroma_ok = True
    except Exception:
        pass
    try:
        from pathlib import Path
        bm25_ok = Path(settings.bm25_index_path).exists()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        chroma_ok=chroma_ok,
        bm25_ok=bm25_ok,
        llm_model=settings.openai_model,
        cache_size=get_cache().size(),
    )
