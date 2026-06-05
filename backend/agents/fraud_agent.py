import logging
from backend.agents.state import ClaimsState
from backend.config.llm import get_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

FRAUD_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an insurance fraud analysis expert.
The user is describing a specific insurance claim. Analyze the characteristics mentioned in the claim description (the query) and assess the fraud risk for THAT claim.
Use the retrieved similar historical claims only as reference context for comparison — do NOT base the risk score on them.

Return a JSON object with:
- fraud_signals: list of objects with {{signal, severity (high/medium/low), description}}
- risk_score: float 0.0-1.0 (probability of fraud for the described claim)
- fraud_cluster: short label for the detected fraud pattern (or "none")
- escalate: true if risk_score >= 0.75, false otherwise
- summary: one sentence explanation

Known fraud signals to look for in the described claim:
- No police report filed (+high risk)
- External agent involvement (+high risk)
- Multiple supplements (3+) (+medium risk)
- Address change near claim date (+medium risk)
- No witness present (+medium risk)
- New policy (< 30 days before accident) (+high risk)
- Third-party fault with no police report (+high risk)
- High vehicle price with missing documentation (+medium risk)
- Claim filed same day as accident (+low risk, slightly suspicious)

Protective factors that LOWER risk:
- Police report available / filed
- Witness statement provided
- Supporting documentation complete

Respond ONLY with valid JSON."""),
    ("human", "Claim description (analyze this for fraud risk): {query}\n\nSimilar historical claims for reference:\n{context}"),
])

SIGNAL_WEIGHTS = {
    "police_report_filed": ("No", 0.25),
    "agent_type": ("External", 0.20),
    "witness_present": ("No", 0.12),
}


def _rule_based_risk(claims: list[dict], query: str = "") -> tuple[float, list[dict]]:
    """Rule-based scoring: parse query text first, fall back to retrieved claim metadata."""
    signals = []
    score = 0.0
    q = query.lower()

    # Parse explicit protective / risk factors from the query text
    has_police = any(p in q for p in ["police report available", "police report filed", "police report present", "police report provided"])
    no_police = any(p in q for p in ["no police report", "without police report", "police report not filed"])
    has_witness = any(p in q for p in ["witness statement", "witness present", "witness provided", "witness available"])
    no_witness = any(p in q for p in ["no witness", "without witness", "witness absent"])

    # Police report check — prefer query text over metadata
    if no_police:
        score += 0.25
        signals.append({"signal": "No police report", "severity": "high", "description": "Claim filed without police report"})
    elif not has_police and claims:
        meta = claims[0].get("metadata", {})
        if meta.get("police_report_filed") == "No":
            score += 0.25
            signals.append({"signal": "No police report", "severity": "high", "description": "Claim filed without police report"})

    # Witness check — prefer query text over metadata
    if no_witness:
        score += 0.12
        signals.append({"signal": "No witness", "severity": "medium", "description": "No witness present"})
    elif not has_witness and claims:
        meta = claims[0].get("metadata", {})
        if meta.get("witness_present") == "No":
            score += 0.12
            signals.append({"signal": "No witness", "severity": "medium", "description": "No witness present"})

    # External agent — metadata only
    if claims:
        meta = claims[0].get("metadata", {})
        if meta.get("agent_type") == "External":
            score += 0.20
            signals.append({"signal": "External agent", "severity": "high", "description": "External agent involved"})

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
        risk_score, fraud_signals = _rule_based_risk(claims, query)
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
