import pandas as pd
from fastapi import APIRouter, HTTPException
from backend.config.settings import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/analytics")
async def analytics():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        col = client.get_collection(settings.chroma_collection)
        results = col.get(limit=5000, include=["metadatas"])
        df = pd.DataFrame(results["metadatas"])

        fraud_count = int(df["fraud_label"].astype(int).sum()) if "fraud_label" in df.columns else 0
        total = len(df)
        by_region = (
            df.groupby("customer_region")["fraud_label"]
            .apply(lambda x: int(x.astype(int).sum())).to_dict()
            if "customer_region" in df.columns else {}
        )
        by_policy = (
            df.groupby("policy_type")["fraud_label"]
            .apply(lambda x: int(x.astype(int).sum())).to_dict()
            if "policy_type" in df.columns else {}
        )
        return {
            "total_claims": total,
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / max(total, 1), 3),
            "by_region": by_region,
            "by_policy_type": by_policy,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/claim/{claim_id}")
async def get_claim(claim_id: str):
    from backend.retrieval.vector_store import get_claim_by_id
    claim = get_claim_by_id(claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
    return claim
