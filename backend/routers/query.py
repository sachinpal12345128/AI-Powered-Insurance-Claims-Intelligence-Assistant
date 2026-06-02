import re
import asyncio
import logging
import traceback
from fastapi import APIRouter, HTTPException
from backend.models import QueryRequest, QueryResponse
from backend.agents.orchestrator import run_pipeline
from backend.guardrails.input_guardrails import check_input
from backend.guardrails.output_guardrails import check_output
from backend.cache.query_cache import get_cache

logger = logging.getLogger(__name__)
router = APIRouter()


# Small-talk fast path -----------------------------------------------------
_GREETING_TOKENS = (
    r"hi|hii+|hello+|hey+|yo|hola|namaste|"
    r"good\s*(?:morning|afternoon|evening|day|night)|"
    r"how\s*are\s*you|how'?s\s*it\s*going|what'?s\s*up|sup|"
    r"thanks?|thank\s*you|thx|ty|"
    r"ok(?:ay)?|cool|nice|great|"
    r"bye|goodbye|see\s*ya|cya"
)
_GREETING_RE = re.compile(
    rf"^\s*(?:{_GREETING_TOKENS})\b[\s\w!?\.,'-]{{0,30}}$",
    re.IGNORECASE,
)

_GREETING_REPLY = (
    "Hello! I'm the Insurance Claims Intelligence Assistant. "
    "Ask me about claim patterns, fraud signals, policy compliance, "
    "or investigation recommendations. For example: "
    "\"show claims with no police report and external agent\" or "
    "\"what are common fraud signals in urban areas\"."
)

# Hard ceiling for the full multi-agent pipeline.
PIPELINE_BUDGET_SECONDS = 90


def _is_small_talk(q: str) -> bool:
    if _GREETING_RE.match(q):
        return True
    tokens = [t for t in re.findall(r"\w+", q) if len(t) > 1]
    return len(tokens) < 2


def _empty_response(query: str, *, answer: str, model_used: str = "fallback",
                    error: str = None, violations: list = None) -> dict:
    return {
        "query": query,
        "answer": answer,
        "risk_score": None,
        "fraud_signals": [],
        "fraud_cluster": "none",
        "policy_violations": [],
        "policy_compliance": "compliant",
        "recommendations": [],
        "matched_claims": [],
        "matched_claim_ids": [],
        "model_used": model_used,
        "escalated": False,
        "cache_hit": False,
        "guardrail_violations": violations or [],
        "error": error,
    }


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    # 1. Small-talk fast path - runs BEFORE any heavy guardrail / model load.
    #    This makes "hi" / "hello" reply in ~1ms even if Presidio/spaCy is stuck.
    if _is_small_talk(q):
        logger.info(f"[query] small-talk fast path: {q!r}")
        return QueryResponse(**_empty_response(q, answer=_GREETING_REPLY, model_used="small-talk"))

    # 2. Heavy input guardrails (injection regex + optional PII)
    guard = check_input(q)
    if not guard.passed:
        raise HTTPException(status_code=400, detail=f"Input rejected: {guard.violations}")

    sanitized = guard.sanitized_query
    cache = get_cache()

    # 3. Cache hit
    cached = cache.get(sanitized)
    if cached:
        cached["cache_hit"] = True
        cached["guardrail_violations"] = guard.violations
        return QueryResponse(**cached)

    # 4. Multi-agent pipeline with hard time budget
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(run_pipeline, sanitized, req.filters),
            timeout=PIPELINE_BUDGET_SECONDS,
        )
    except asyncio.TimeoutError:
        msg = f"Pipeline exceeded {PIPELINE_BUDGET_SECONDS}s budget."
        logger.error(msg)
        return QueryResponse(**_empty_response(
            sanitized,
            answer="The analysis took too long and was aborted. The LLM gateway may be unreachable.",
            model_used="timeout",
            error=msg,
            violations=guard.violations,
        ))
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Pipeline crashed: {e}\n{tb}")
        return QueryResponse(**_empty_response(
            sanitized,
            answer="Something went wrong while running the analysis. Check the backend logs.",
            model_used="error",
            error=f"{type(e).__name__}: {e}",
            violations=guard.violations,
        ))

    out_guard = check_output(result, result.get("matched_claim_ids", []))
    result = out_guard.response
    result["guardrail_violations"] = guard.violations + out_guard.issues
    result["cache_hit"] = False

    # 5. DeepEval inline scoring
    eval_scores = {}
    try:
        from backend.evaluation.evaluator import EvalSample, run_evaluation
        retrieval_context = [
            c.get("text", "") for c in result.get("matched_claims", []) if c.get("text")
        ]
        if retrieval_context and result.get("answer"):
            sample = EvalSample(
                query=sanitized,
                actual_output=result["answer"],
                expected_output=sanitized,
                retrieval_context=retrieval_context,
            )
            eval_results = await asyncio.to_thread(run_evaluation, [sample])
            eval_scores = {r.metric: round(r.score, 3) for r in eval_results}
            logger.info(f"[eval] scores: {eval_scores}")
    except Exception as eval_err:
        logger.warning(f"[eval] DeepEval skipped: {eval_err}")

    result["eval_scores"] = eval_scores
    cache.set(sanitized, result)
    return QueryResponse(**result)
