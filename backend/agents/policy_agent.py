import logging
import json, re
from backend.agents.state import ClaimsState
from backend.config.llm import get_llm_with_fallback
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

POLICY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an insurance policy compliance expert.
Review the claim details and check for policy violations.
Return a JSON object with:
- violations: list of objects with {rule, severity (high/medium/low), description}
- compliance: "compliant" or "violations_found"
- summary: one sentence

Check these rules:
- Claim must be filed within 30 days of incident
- Police report required for collision claims
- Supporting documents must be complete
- Policy must be active at time of incident
- Coverage type must match claim type

Respond ONLY with valid JSON."""),
    ("human", "Claim context:\n{context}"),
])


def policy_agent(state: ClaimsState) -> ClaimsState:
    """Check policy compliance for top retrieved claim."""
    claims = state.get("retrieved_claims", [])
    if not claims:
        return {**state, "policy_violations": [], "policy_compliance": "compliant"}

    context = claims[0]["text"][:600] if claims else ""

    try:
        llm = get_llm_with_fallback()
        chain = POLICY_PROMPT | llm
        response = chain.invoke({"context": context})

        raw = response.content
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            raise ValueError("No JSON in response")

        violations = data.get("violations", [])
        compliance = data.get("compliance", "compliant")

    except Exception as e:
        logger.warning(f"[policy_agent] LLM failed ({e}). Using empty violations.")
        violations = []
        compliance = "compliant"

    logger.info(f"[policy_agent] compliance={compliance}, violations={len(violations)}")

    return {
        **state,
        "policy_violations": violations,
        "policy_compliance": compliance,
        "messages": [{"role": "policy_agent", "content": f"Compliance: {compliance}. Violations: {len(violations)}."}],
    }
