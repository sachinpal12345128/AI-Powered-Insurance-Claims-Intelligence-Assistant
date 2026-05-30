"""
Output guardrails:
1. Risk score bounds check (0–1)
2. Hallucinated claim_id check (must exist in retrieved context)
3. Basic format validation
"""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SAFE_FALLBACK = {
    "answer": "I was unable to generate a reliable response. Please rephrase your query.",
    "risk_score": None,
    "fraud_signals": [],
    "recommendations": [],
    "guardrail_triggered": True,
}


@dataclass
class OutputGuardrailResult:
    passed: bool
    response: dict
    issues: list[str]


def check_output(response: dict, retrieved_ids: list[str]) -> OutputGuardrailResult:
    issues = []

    # 1. Risk score bounds
    risk = response.get("risk_score")
    if risk is not None:
        try:
            risk = float(risk)
            if not (0.0 <= risk <= 1.0):
                issues.append(f"Risk score {risk} out of bounds [0,1]")
                response["risk_score"] = max(0.0, min(1.0, risk))
        except (TypeError, ValueError):
            issues.append("Invalid risk score type")
            response["risk_score"] = None

    # 2. Claim ID hallucination check
    cited_ids = response.get("matched_claim_ids", [])
    hallucinated = [cid for cid in cited_ids if cid not in retrieved_ids]
    if hallucinated:
        issues.append(f"Hallucinated claim IDs removed: {hallucinated}")
        response["matched_claim_ids"] = [cid for cid in cited_ids if cid in retrieved_ids]
        logger.warning(f"Output guardrail: removed hallucinated IDs {hallucinated}")

    # 3. Answer must not be empty
    if not response.get("answer", "").strip():
        issues.append("Empty answer generated")
        return OutputGuardrailResult(False, SAFE_FALLBACK, issues)

    passed = len([i for i in issues if "hallucinated" in i or "out of bounds" in i]) == 0
    return OutputGuardrailResult(passed=passed, response=response, issues=issues)
