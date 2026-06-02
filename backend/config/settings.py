from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Primary LLM — org gateway
    openai_api_key: str = Field("learner027", alias="OPENAI_API_KEY")
    openai_base_url: str = Field("https://keygateway.arshnivlabs.com", alias="OPENAI_BASE_URL")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")

    # Fallback — Groq
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Fallback — HuggingFace
    hf_api_key: str = Field("", alias="HF_API_KEY")
    hf_model: str = Field("mistralai/Mistral-7B-Instruct-v0.2", alias="HF_MODEL")

    # Reranker (optional)
    cohere_api_key: str = Field("", alias="COHERE_API_KEY")

    # Embeddings
    fastembed_model: str = Field("BAAI/bge-small-en-v1.5", alias="FASTEMBED_MODEL")

    # Vector store
    chroma_persist_dir: str = Field("./data/chroma_db", alias="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("insurance_claims", alias="CHROMA_COLLECTION")

    # BM25
    bm25_index_path: str = Field("./data/bm25_index/bm25.pkl", alias="BM25_INDEX_PATH")

    # Ingestion
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")
    embed_batch_size: int = Field(256, alias="EMBED_BATCH_SIZE")

    # Retrieval
    dense_top_k: int = Field(20, alias="DENSE_TOP_K")
    sparse_top_k: int = Field(20, alias="SPARSE_TOP_K")
    rerank_top_k: int = Field(5, alias="RERANK_TOP_K")
    crag_relevance_threshold: float = Field(0.6, alias="CRAG_RELEVANCE_THRESHOLD")

    # Guardrails
    enable_pii_detection: bool = Field(False, alias="ENABLE_PII_DETECTION")
    enable_contextual_compression: bool = Field(False, alias="ENABLE_CONTEXTUAL_COMPRESSION")

    # Cache
    cache_ttl_seconds: int = Field(600, alias="CACHE_TTL_SECONDS")
    cache_similarity_threshold: float = Field(0.92, alias="CACHE_SIMILARITY_THRESHOLD")

    # Observability
    langchain_tracing_v2: str = Field("false", alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field("", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field("insurance-claims-assistant", alias="LANGCHAIN_PROJECT")

    # App
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def root_dir(self) -> Path:
        return ROOT_DIR

    def resolve(self, path_str: str) -> Path:
        p = Path(path_str)
        return p if p.is_absolute() else (ROOT_DIR / p).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
