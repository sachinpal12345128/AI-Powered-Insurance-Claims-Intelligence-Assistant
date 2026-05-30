import logging
from backend.agents.state import ClaimsState
from backend.config.llm import get_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

FRAUD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an insurance fraud analysis expert.
Analyze the retrieved claims and identify fraud signals.
Return a JSON object with:
- fraud_signals: list of objects with {signal, severity (high/medium/low), description}
- risk_score: float 0.0-1.0 (probability of fraud)
- fraud_cluster: short label for the detected fraud pattern (or "none")
- escalate: true if risk_score >= 0.75, false otherwise
- summary: one sentence explanation

Known fraud signals to look for:
- No police report filed
- External agent involvement
- Multiple supplements (3+)
- Address change near claim date
- No witness present
- New policy (< 30 days before accident)
- Third-party fault with no police report
- High vehicle price with missing documentation

Respond ONLY with valid JSON."""),
    ("human", "Query: {query}\n\nRetrieved claims context:\n{context}"),
])

SIGNAL_WEIGHTS = {
    "police_report_filed": ("No", 0.25),
    "agent_type": ("External", 0.20),
    "witness_present": ("No", 0.12),
}


def _rule_based_risk(claims: list[dict]) -> tuple[float, list[dict]]:
    """Fast rule-based scoring as baseline."""
    if not claims:
        return 0.0, []
    signals = []
    score = 0.0
    meta = claims[0].get("metadata", {})
    if meta.get("police_report_filed") == "No":
        score += 0.25; signals.append({"signal": "No police report", "severity": "high", "description": "Claim filed without police report"})
    if meta.get("agent_type") == "External":
        score += 0.20; signals.append({"signal": "External agent", "severity": "high", "description": "External agent involved"})
    if meta.get("witness_present") == "No":
        score += 0.12; signals.append({"signal": "No witness", "severity": "medium", "description": "No witness present"})
    return min(score, 1.0), signals


def fraud_agent(state: ClaimsState) -> ClaimsState:
    """Analyze retrieved claims for fraud signals and compute risk score."""
    claims = state.get("retrieved_claims", [])
    query = state["query"]

    if not claims:
        return {**state, "fraud_signals": [], "risk_score": 0.0, "fraud_cluster": "none", "escalate": False}

    context = "\n\n".join([f"Claim {i+1}: {c['text'][:400]}" for i, c in enumerate(claims[:3])])

    try:
        llm = get_llm_with_fallback()
        chain = FRAUD_PROMPT | llm
        response = chain.invoke({"query": query, "context": context})

        import json, re
        raw = response.content
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON in LLM response")

        risk_score = float(data.get("risk_score", 0.0))
        fraud_signals = data.get("fraud_signals", [])
        fraud_cluster = data.get("fraud_cluster", "none")
        escalate = risk_score >= 0.75
        model_used = "openai"

    except Exception as e:
        logger.warning(f"[fraud_agent] LLM failed ({e}). Using rule-based fallback.")
        risk_score, fraud_signals = _rule_based_risk(claims)
        fraud_cluster = "rule-based"
        escalate = risk_score >= 0.75
        model_used = "rule-based"

    logger.info(f"[fraud_agent] risk_score={risk_score:.2f}, escalate={escalate}")

    return {
        **state,
        "fraud_signals": fraud_signals,
        "risk_score": risk_score,
        "fraud_cluster": fraud_cluster,
        "escalate": escalate,
        "model_used": model_used,
        "messages": [{"role": "fraud_agent", "content": f"Risk score: {risk_score:.2f}. Escalate: {escalate}."}],
    }
