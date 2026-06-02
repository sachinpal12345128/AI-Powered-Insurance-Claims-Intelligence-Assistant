from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    filters: Optional[dict] = None


class IngestRequest(BaseModel):
    csv_path: str = Field(default="./data/source_data/fraud_oracle.csv")


class QueryResponse(BaseModel):
    query: str = ""
    answer: str = ""
    risk_score: Optional[float] = None
    fraud_signals: list = []
    fraud_cluster: str = "none"
    policy_violations: list = []
    policy_compliance: str = "compliant"
    recommendations: list = []
    matched_claims: list = []
    matched_claim_ids: list = []
    model_used: str = "openai"
    escalated: bool = False
    cache_hit: bool = False
    guardrail_violations: list = []
    eval_scores: dict = {}
    error: Optional[str] = None

    class Config:
        extra = "ignore"


class IngestResponse(BaseModel):
    success: bool
    rows_loaded: int
    chunks_created: int
    vectors_stored: int
    embedding_model: str
    elapsed_seconds: float
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    pinecone_ok: bool
    bm25_ok: bool
    llm_model: str
    cache_size: int
