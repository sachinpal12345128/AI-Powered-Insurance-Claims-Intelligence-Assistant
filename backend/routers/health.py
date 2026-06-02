import os
from fastapi import APIRouter
from backend.models import HealthResponse
from backend.cache.query_cache import get_cache
from backend.config.settings import get_settings
from backend.config.paths import bm25_path

router = APIRouter()
settings = get_settings()


def _active_llm_model() -> str:
    primary = os.getenv("LLM_PRIMARY", "openai").lower()
    if primary == "groq" and settings.groq_api_key:
        return f"groq:{settings.groq_model}"
    if primary == "openai" and settings.openai_api_key:
        return f"openai:{settings.openai_model}"
    if settings.groq_api_key:
        return f"groq:{settings.groq_model} (fallback)"
    if settings.openai_api_key:
        return f"openai:{settings.openai_model} (fallback)"
    return "none"


@router.get("/health", response_model=HealthResponse)
async def health():
    pinecone_ok = False
    bm25_ok = False
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.pinecone_api_key, ssl_verify=False)
        index = pc.Index(settings.pinecone_index_name)
        index.describe_index_stats()
        pinecone_ok = True
    except Exception:
        pass
    try:
        from pathlib import Path
        bm25_ok = Path(bm25_path()).exists()
    except Exception:
        pass

    return HealthResponse(
        status="ok",
        pinecone_ok=pinecone_ok,
        bm25_ok=bm25_ok,
        llm_model=_active_llm_model(),
        cache_size=get_cache().size(),
    )
