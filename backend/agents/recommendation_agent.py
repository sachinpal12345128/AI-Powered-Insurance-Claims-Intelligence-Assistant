import logging
from backend.agents.state import ClaimsState
from backend.config.llm import get_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

REC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior insurance fraud investigator.
Based on the analysis results below, generate a clear, actionable investigation report.

Your response must be a JSON object with:
- recommendations: list of 3-5 specific investigation actions (strings)
- answer: 2-3 sentence summary for the claims analyst
- priority: "immediate" | "high" | "standard"

Be specific. Reference actual claim details when possible.
Respond ONLY with valid JSON."""),
    ("human", """Query: {query}

Risk score: {risk_score}
Fraud signals: {fraud_signals}
Fraud cluster: {fraud_cluster}
Policy violations: {policy_violations}
Top claim summary: {claim_summary}
Escalated: {escalate}"""),
])

FALLBACK_RECS = [
    "Request police report if not already filed",
    "Verify witness statements independently",
    "Cross-check vehicle VIN history",
    "Review supplement documentation",
    "Confirm agent credentials and history",
]


def recommendation_agent(state: ClaimsState) -> ClaimsState:
    """Generate investigation recommendations and final answer."""
    claims = state.get("retrieved_claims", [])
    claim_summary = claims[0]["text"][:300] if claims else "No claims retrieved."

    try:
        llm = get_llm_with_fallback()
        chain = REC_PROMPT | llm
        response = chain.invoke({
            "query": state["query"],
            "risk_score": state.get("risk_score", 0.0),
            "fraud_signals": state.get("fraud_signals", []),
            "fraud_cluster": state.get("fraud_cluster", "none"),
            "policy_violations": state.get("policy_violations", []),
            "claim_summary": claim_summary,
            "escalate": state.get("escalate", False),
        })

        import json, re
        raw = response.content
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON in response")

        recommendations = data.get("recommendations", FALLBACK_RECS)
        answer = data.get("answer", "Analysis complete.")
        priority = data.get("priority", "standard")

    except Exception as e:
        logger.warning(f"[recommendation_agent] LLM failed ({e}). Using fallback recommendations.")
        risk = state.get("risk_score", 0.0)
        recommendations = FALLBACK_RECS
        answer = f"Risk score: {risk:.2f}. {len(state.get('fraud_signals', []))} fraud signal(s) detected. Manual review recommended."
        priority = "immediate" if risk >= 0.75 else "high" if risk >= 0.5 else "standard"

    logger.info(f"[recommendation_agent] Generated {len(recommendations)} recommendations.")

    return {
        **state,
        "recommendations": recommendations,
        "answer": answer,
        "messages": [{"role": "recommendation_agent", "content": answer}],
    }
