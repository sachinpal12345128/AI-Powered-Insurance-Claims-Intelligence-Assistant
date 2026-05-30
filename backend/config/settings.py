from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Primary LLM — org gateway
    openai_api_key: str = Field("learner027", env="OPENAI_API_KEY")
    openai_base_url: str = Field("https://keygateway.arshnivlabs.com", env="OPENAI_BASE_URL")
    openai_model: str = Field("gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field("text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")

    # Fallback 1 — Groq
    groq_api_key: str = Field("", env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-70b-versatile", env="GROQ_MODEL")

    # Fallback 2 — HuggingFace
    hf_api_key: str = Field("", env="HF_API_KEY")
    hf_model: str = Field("mistralai/Mistral-7B-Instruct-v0.2", env="HF_MODEL")

    # Reranker
    cohere_api_key: str = Field("", env="COHERE_API_KEY")

    # Vector store
    chroma_persist_dir: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection: str = Field("insurance_claims", env="CHROMA_COLLECTION")

    # BM25
    bm25_index_path: str = Field("./data/bm25_index/bm25.pkl", env="BM25_INDEX_PATH")

    # Ingestion
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    embed_batch_size: int = Field(100, env="EMBED_BATCH_SIZE")

    # Cache
    cache_ttl_seconds: int = Field(600, env="CACHE_TTL_SECONDS")
    cache_similarity_threshold: float = Field(0.92, env="CACHE_SIMILARITY_THRESHOLD")

    # Observability
    langchain_tracing_v2: str = Field("false", env="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field("", env="LANGCHAIN_API_KEY")
    langchain_project: str = Field("insurance-claims-assistant", env="LANGCHAIN_PROJECT")

    # App
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
