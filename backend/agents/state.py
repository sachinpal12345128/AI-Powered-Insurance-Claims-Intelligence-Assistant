from typing import TypedDict, Annotated, Optional
import operator


class ClaimsState(TypedDict):
    # Input
    query: str
    filters: dict

    # Retrieval agent output
    retrieved_claims: list[dict]
    retrieved_ids: list[str]

    # Fraud analysis agent output
    fraud_signals: list[dict]
    risk_score: float
    fraud_cluster: str
    escalate: bool          # True → trigger A2A escalation to recommendation agent

    # Policy validation agent output
    policy_violations: list[dict]
    policy_compliance: str  # "compliant" | "violations_found"

    # Recommendation agent output
    recommendations: list[str]
    answer: str

    # Meta
    model_used: str
    messages: Annotated[list, operator.add]
    error: Optional[str]
