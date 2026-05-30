"""
LangGraph orchestrator — wires all 4 agents into a stateful graph.
Flow: retrieval → fraud_analysis → policy_validation → (conditional) recommendation
A2A escalation: fraud_agent sets escalate=True → recommendation_agent runs immediately.
"""
import logging
from langgraph.graph import StateGraph, END
from backend.agents.state import ClaimsState
from backend.agents.retrieval_agent import retrieval_agent
from backend.agents.fraud_agent import fraud_agent
from backend.agents.policy_agent import policy_agent
from backend.agents.recommendation_agent import recommendation_agent

logger = logging.getLogger(__name__)


def _should_escalate(state: ClaimsState) -> str:
    """Conditional edge: if escalate=True, skip policy check and go straight to recommendation."""
    if state.get("escalate"):
        logger.info("[orchestrator] A2A escalation triggered → jumping to recommendation_agent")
        return "recommendation"
    return "policy"


def build_graph() -> StateGraph:
    graph = StateGraph(ClaimsState)

    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("fraud", fraud_agent)
    graph.add_node("policy", policy_agent)
    graph.add_node("recommendation", recommendation_agent)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "fraud")
    graph.add_conditional_edges("fraud", _should_escalate, {
        "policy": "policy",
        "recommendation": "recommendation",
    })
    graph.add_edge("policy", "recommendation")
    graph.add_edge("recommendation", END)

    return graph.compile()


# Compile once at import time
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_pipeline(query: str, filters: dict = None) -> dict:
    """
    Run the full multi-agent pipeline.
    Returns structured response dict.
    """
    graph = get_graph()

    initial_state: ClaimsState = {
        "query": query,
        "filters": filters or {},
        "retrieved_claims": [],
        "retrieved_ids": [],
        "fraud_signals": [],
        "risk_score": 0.0,
        "fraud_cluster": "none",
        "escalate": False,
        "policy_violations": [],
        "policy_compliance": "compliant",
        "recommendations": [],
        "answer": "",
        "model_used": "openai",
        "messages": [],
        "error": None,
    }

    final_state = graph.invoke(initial_state)

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
