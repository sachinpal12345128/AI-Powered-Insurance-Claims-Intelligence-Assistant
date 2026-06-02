"""
LangGraph orchestrator - wires all 4 agents into a stateful graph with per-node timing.
Flow: retrieval -> fraud -> policy -> recommendation (or A2A escalation).
"""
import logging
import time
from langgraph.graph import StateGraph, END
from backend.agents.state import ClaimsState
from backend.agents.retrieval_agent import retrieval_agent as _retrieval
from backend.agents.fraud_agent import fraud_agent as _fraud
from backend.agents.policy_agent import policy_agent as _policy
from backend.agents.recommendation_agent import recommendation_agent as _recommendation

logger = logging.getLogger(__name__)


def _timed(name, fn):
    def wrapper(state):
        t = time.time()
        try:
            out = fn(state)
            logger.info(f"[orchestrator] {name} done in {(time.time()-t)*1000:.0f}ms")
            return out
        except Exception as e:
            logger.error(f"[orchestrator] {name} FAILED after {(time.time()-t)*1000:.0f}ms: {e}")
            raise
    return wrapper


def _should_escalate(state: ClaimsState) -> str:
    if state.get("escalate"):
        logger.info("[orchestrator] A2A escalation -> jumping to recommendation_agent")
        return "recommendation"
    return "policy"


def build_graph() -> StateGraph:
    graph = StateGraph(ClaimsState)
    graph.add_node("retrieval",       _timed("retrieval", _retrieval))
    graph.add_node("fraud",           _timed("fraud", _fraud))
    graph.add_node("policy",          _timed("policy", _policy))
    graph.add_node("recommendation",  _timed("recommendation", _recommendation))
    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "fraud")
    graph.add_conditional_edges("fraud", _should_escalate, {
        "policy": "policy",
        "recommendation": "recommendation",
    })
    graph.add_edge("policy", "recommendation")
    graph.add_edge("recommendation", END)
    return graph.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(query: str, filters: dict = None) -> dict:
    graph = get_graph()
    initial_state: ClaimsState = {
        "query": query,
        "filters": filters or {},
        "retrieved_claims": [], "retrieved_ids": [],
        "fraud_signals": [], "risk_score": 0.0, "fraud_cluster": "none", "escalate": False,
        "policy_violations": [], "policy_compliance": "compliant",
        "recommendations": [], "answer": "",
        "model_used": "openai", "messages": [], "error": None,
    }
    t = time.time()
    final_state = graph.invoke(initial_state)
    logger.info(f"[orchestrator] full pipeline done in {(time.time()-t)*1000:.0f}ms")
    return {
        "query": query,
        "answer": final_state.get("answer", ""),
        "risk_score": final_state.get("risk_score", 0.0),
        "fraud_signals": final_state.get("fraud_signals", []),
        "fraud_cluster": final_state.get("fraud_cluster", "none"),
        "policy_violations": final_state.get("policy_violations", []),
        "policy_compliance": final_state.get("policy_compliance", "compliant"),
        "recommendations": final_state.get("recommendations", []),
        "matched_claims": final_state.get("retrieved_claims", []),
        "matched_claim_ids": final_state.get("retrieved_ids", []),
        "model_used": final_state.get("model_used", "openai"),
        "escalated": final_state.get("escalate", False),
        "error": final_state.get("error"),
    }
