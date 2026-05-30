"""
LLM + Embedding client factory with fallback chain.
Primary: OpenAI gateway -> Fallback 1: Groq -> Fallback 2: HuggingFace
"""
import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_primary_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
    )


def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )


def get_hf_llm():
    from langchain_community.llms import HuggingFaceHub
    return HuggingFaceHub(
        repo_id=settings.hf_model,
        huggingfacehub_api_token=settings.hf_api_key,
    )


def get_llm_with_fallback():
    """Returns LLM with automatic fallback chain."""
    primary = get_primary_llm()
    fallbacks = []
    if settings.groq_api_key:
        try:
            fallbacks.append(get_groq_llm())
        except Exception:
            pass
    if settings.hf_api_key:
        try:
            fallbacks.append(get_hf_llm())
        except Exception:
            pass
    if fallbacks:
        return primary.with_fallbacks(fallbacks)
    return primary


def get_primary_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_local_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_embeddings_with_fallback():
    """Returns embedding model; falls back to local sentence-transformers."""
    try:
        emb = get_primary_embeddings()
        emb.embed_query("test")
        logger.info("Using OpenAI embeddings via org gateway.")
        return emb, "openai"
    except Exception as e:
        logger.warning(f"OpenAI embeddings unavailable ({e}). Falling back to MiniLM.")
        return get_local_embeddings(), "minilm"
