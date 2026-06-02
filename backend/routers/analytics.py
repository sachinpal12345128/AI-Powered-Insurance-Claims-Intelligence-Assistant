import pandas as pd
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.config.settings import get_settings

router = APIRouter()
settings = get_settings()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "source_data" / "fraud_oracle.csv"


def _load_df():
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH)
    # Normalise column names to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    return df


@router.get("/analytics")
async def analytics():
    try:
        df = _load_df()
        if df is None or df.empty:
            return {
                "total_claims": 0, "fraud_count": 0, "fraud_rate": 0.0,
                "by_region": {}, "by_policy_type": {},
                "message": "No data found. Check data/raw/fraud_oracle.csv.",
            }

        fraud_col = next((c for c in df.columns if "fraud" in c), None)
        region_col = next((c for c in df.columns if "region" in c or "area" in c), None)
        policy_col = next((c for c in df.columns if "policy" in c and "type" in c), None)

        fraud_count = int(df[fraud_col].astype(int).sum()) if fraud_col else 0
        total = len(df)

        by_region = (
            df.groupby(region_col)[fraud_col]
            .apply(lambda x: int(x.astype(int).sum())).to_dict()
            if region_col and fraud_col else {}
        )
        by_policy = (
            df.groupby(policy_col)[fraud_col]
            .apply(lambda x: int(x.astype(int).sum())).to_dict()
            if policy_col and fraud_col else {}
        )
        return {
            "total_claims": total,
            "fraud_count": fraud_count,
            "fraud_rate": round(fraud_count / max(total, 1), 3),
            "by_region": by_region,
            "by_policy_type": by_policy,
        }
    except Exception as e:
        return {
            "total_claims": 0, "fraud_count": 0, "fraud_rate": 0.0,
            "by_region": {}, "by_policy_type": {},
            "message": f"Analytics error: {e}",
        }


@router.get("/claim/{claim_id}")
async def get_claim(claim_id: str):
    try:
        from backend.retrieval.vector_store import get_claim_by_id
        claim = get_claim_by_id(claim_id)
        if not claim:
            raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found")
        return claim
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found (store not ready)")
