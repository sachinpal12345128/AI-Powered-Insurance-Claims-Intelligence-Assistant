"""
LLM + Embedding factory.
LLM:       OpenAI gateway (8s) → Groq fallback
Embedding: OpenAI gateway → HuggingFace API → fastembed → numpy hash (guaranteed fallback)
"""
from __future__ import annotations
import hashlib
import logging
import math
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_openai_client = None
_fastembed_model = None

EMBED_DIM = 1536  # text-embedding-3-small full dimension (gateway ignores Matryoshka truncation)

# ── LLM ───────────────────────────────────────────────────────────────────────

def get_primary_llm() -> ChatOpenAI:
    import httpx
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0, timeout=8, max_retries=1,
        http_client=httpx.Client(verify=False),
        http_async_client=httpx.AsyncClient(verify=False),
    )

def get_groq_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0, timeout=20, max_retries=2,
    )

def get_llm_with_fallback():
    primary = get_primary_llm()
    fallbacks = []
    if settings.groq_api_key:
        try:
            fallbacks.append(get_groq_llm())
            logger.info("[llm] Groq added to fallback chain.")
        except Exception:
            pass
    return primary.with_fallbacks(fallbacks) if fallbacks else primary

# ── Embedding backends ────────────────────────────────────────────────────────

def _embed_openai(texts: List[str]) -> List[List[float]]:
    global _openai_client
    if _openai_client is None:
        import httpx
        from openai import OpenAI
        _openai_client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=30, max_retries=1,
            http_client=httpx.Client(verify=False),
        )
    results = []
    batch_size = 32  # small batches to stay under gateway payload limit
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = _openai_client.embeddings.create(
            model=settings.openai_embedding_model, input=batch
        )
        results.extend([e.embedding for e in resp.data])
    return results


def _embed_hf_api(texts: List[str]) -> List[List[float]]:
    import urllib.request, json as _json
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {"Content-Type": "application/json"}
    if settings.hf_api_key:
        headers["Authorization"] = f"Bearer {settings.hf_api_key}"
    payload = _json.dumps({"inputs": texts, "options": {"wait_for_model": True}}).encode()
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        result = _json.loads(r.read())
    if result and isinstance(result[0], list) and isinstance(result[0][0], float):
        return result
    import numpy as np
    return [list(np.mean(r, axis=0)) for r in result]


def _embed_fastembed(texts: List[str]) -> List[List[float]]:
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding
        _fastembed_model = TextEmbedding(model_name=settings.fastembed_model)
        logger.info(f"fastembed loaded ({settings.fastembed_model}).")
    return [list(v) for v in _fastembed_model.embed(texts)]


def _embed_numpy(texts: List[str]) -> List[List[float]]:
    """
    Fast vectorized numpy hash embedding — no external deps.
    Deterministic 384-dim unit vectors using hash trick.
    """
    import numpy as np

    STOP = {"a","an","the","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","of","in","on","at","to","for","with",
            "by","from","and","or","but","not","it","its","this","that"}

    results = np.zeros((len(texts), EMBED_DIM), dtype=np.float32)

    for i, text in enumerate(texts):
        words = [w.strip(".,;:!\"'()") for w in text.lower().split()]
        words = [w for w in words if w and w not in STOP and len(w) > 1]
        if not words:
            continue
        # vectorized hash positions
        h1 = np.array([int(hashlib.md5(w.encode()).hexdigest(), 16) % EMBED_DIM for w in words])
        h2 = np.array([int(hashlib.sha1(w.encode()).hexdigest(), 16) % EMBED_DIM for w in words])
        np.add.at(results[i], h1, 1.0 / len(words))
        np.add.at(results[i], h2, 0.5 / len(words))

    # normalize rows
    norms = np.linalg.norm(results, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    results = results / norms
    return results.tolist()


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Try OpenAI → HF API → fastembed → numpy hash (always works)."""
    if settings.openai_api_key:
        try:
            return _embed_openai(texts)
        except Exception as e:
            logger.warning(f"[embed] OpenAI failed ({e!r}), trying HF API.")
    try:
        result = _embed_hf_api(texts)
        logger.info("[embed] Using HuggingFace Inference API.")
        return result
    except Exception as e:
        logger.warning(f"[embed] HF API failed ({e!r}), trying fastembed.")
    try:
        return _embed_fastembed(texts)
    except Exception as e:
        logger.warning(f"[embed] fastembed failed ({e!r}), using numpy hash embedding.")
    result = _embed_numpy(texts)
    logger.info(f"[embed] Using numpy hash embedding (dim={EMBED_DIM}).")
    return result


def embed_query(query: str) -> List[float]:
    return embed_texts([query])[0]


class InsuranceEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return embed_texts(texts)
    def embed_query(self, text: str) -> List[float]:
        return embed_query(text)


def get_primary_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


def get_embeddings_with_fallback():
    for probe, name in [(_embed_openai, "openai"), (_embed_hf_api, "hf-api"),
                        (_embed_fastembed, "fastembed")]:
        try:
            probe(["test"])
            logger.info(f"[embed] Active backend: {name}")
            return InsuranceEmbeddings(), name
        except Exception:
            continue
    logger.info("[embed] Active backend: numpy-hash")
    return InsuranceEmbeddings(), "numpy-hash"
