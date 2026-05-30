import logging
from backend.agents.state import ClaimsState
from backend.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)


def retrieval_agent(state: ClaimsState) -> ClaimsState:
    """Hybrid search + reranking → populate retrieved_claims."""
    query = state["query"]
    filters = state.get("filters") or {}

    logger.info(f"[retrieval_agent] Query: {query[:80]}")

    try:
        results = hybrid_search(query=query, n_retrieve=20, top_k=5, filters=filters or None)
        claim_ids = [r.get("metadata", {}).get("claim_id", r["id"]) for r in results]

        return {
            **state,
            "retrieved_claims": results,
            "retrieved_ids": claim_ids,
            "messages": [{"role": "retrieval_agent", "content": f"Retrieved {len(results)} claims."}],
        }
    except Exception as e:
        logger.error(f"[retrieval_agent] Error: {e}")
        return {**state, "retrieved_claims": [], "retrieved_ids": [], "error": str(e)}
