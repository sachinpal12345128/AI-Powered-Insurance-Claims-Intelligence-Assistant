from fastapi import APIRouter, HTTPException
from backend.models import QueryRequest, QueryResponse
from backend.agents.orchestrator import run_pipeline
from backend.guardrails.input_guardrails import check_input
from backend.guardrails.output_guardrails import check_output
from backend.cache.query_cache import get_cache

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    guard = check_input(req.query)
    if not guard.passed:
        raise HTTPException(status_code=400, detail=f"Input rejected: {guard.violations}")

    sanitized = guard.sanitized_query
    cache = get_cache()

    cached = cache.get(sanitized)
    if cached:
        cached["cache_hit"] = True
        cached["guardrail_violations"] = guard.violations
        return QueryResponse(**cached)

    try:
        result = run_pipeline(query=sanitized, filters=req.filters)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    out_guard = check_output(result, result.get("matched_claim_ids", []))
    result = out_guard.response
    result["guardrail_violations"] = guard.violations + out_guard.issues
    result["cache_hit"] = False

    cache.set(sanitized, result)
    return QueryResponse(**result)
