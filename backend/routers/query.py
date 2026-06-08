import re
import asyncio
import logging
import traceback
from pathlib import Path
from fastapi import APIRouter, HTTPException
from backend.models import QueryRequest, QueryResponse
from backend.agents.orchestrator import run_pipeline
from backend.guardrails.input_guardrails import check_input
from backend.guardrails.output_guardrails import check_output
from backend.cache.query_cache import get_cache

# ---------------------------------------------------------------------------
# Analytics / aggregate query handling
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

# Matches count queries AND rate/statistical/regional queries
_AGG_RE = re.compile(
    r"\b("
    r"how\s+many|count\s+(of\s+)?claims?|total\s+(number\s+of\s+)?claims?|"
    r"number\s+of\s+claims?|claims?\s+filed|claims?\s+submitted|"
    r"how\s+much|how\s+often|how\s+frequently|"
    r"fraud\s+rate|fraud\s+percentage|fraud\s+ratio|fraud\s+statistic|"
    r"what\s+is\s+the\s+fraud|fraud\s+level|rate\s+of\s+fraud|"
    r"fraud\s+(by|in|for|across|per)\s+\w+|"
    r"claims?\s+(by|in|for|across|per)\s+\w+|"
    r"breakdown|distribution\s+of|statistics\s+(for|on|about)|"
    r"(total|overall)\s+fraud"
    r")\b",
    re.IGNORECASE,
)

_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "source_data" / "fraud_oracle.csv"


def _is_aggregate_query(q: str) -> bool:
    return bool(_AGG_RE.search(q))


def _fmt_breakdown(grp_df, label_col: str, fraud_col: str) -> str:
    """Format a grouped breakdown table as a readable string."""
    lines = []
    for _, row in grp_df.iterrows():
        lines.append(
            f"  • {row[label_col]}: {int(row['total'])} claims, "
            f"{int(row['fraud'])} fraud ({row['rate']}%)"
        )
    return "\n".join(lines)


def _safe_analytics_response(q: str, answer: str) -> dict:
    """Return a well-formed analytics response with a custom answer."""
    return {
        "query": q,
        "answer": answer,
        "risk_score": None,
        "fraud_signals": [],
        "fraud_cluster": "none",
        "policy_violations": [],
        "policy_compliance": "compliant",
        "recommendations": [
            "Use the Analytics page for interactive charts by region and policy type.",
            "Use the Investigate page to analyse specific claim fraud signals.",
        ],
        "matched_claims": [],
        "matched_claim_ids": [],
        "model_used": "analytics-direct",
        "escalated": False,
        "cache_hit": False,
        "guardrail_violations": [],
        "eval_scores": {},
        "error": None,
    }


def _handle_analytics_query(q: str) -> dict:
    """Answer count/rate/regional aggregate queries directly from the full CSV dataset."""
    _log = logging.getLogger(__name__)

    try:
        import pandas as pd
    except ImportError:
        _log.warning(
            "[analytics] pandas not installed — cannot serve analytics query. "
            "Run: pip install pandas"
        )
        return _safe_analytics_response(
            q,
            "Analytics queries require pandas, which is not installed on this server. "
            "Please install it with: pip install pandas",
        )

    if not _CSV_PATH.exists():
        _log.warning(
            "[analytics] CSV not found at %s — aggregate fast path skipped. "
            "Run the ETL pipeline or check data/source_data/fraud_oracle.csv.",
            _CSV_PATH,
        )
        return _safe_analytics_response(
            q,
            f"The source data file was not found at the expected path "
            f"({_CSV_PATH}). Please run the ETL pipeline or upload the CSV "
            f"via the Data Ingestion page before querying analytics.",
        )

    try:
        df = pd.read_csv(_CSV_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        total = len(df)

        fraud_col = next((c for c in df.columns if "fraud" in c), None)
        area_col  = next((c for c in df.columns if "region" in c or "area" in c), None)
        policy_col = next((c for c in df.columns if "policy" in c and "type" in c), None)

        fraud_count = int(df[fraud_col].astype(int).sum()) if fraud_col else 0
        overall_rate = round(fraud_count / max(total, 1) * 100, 1)

        q_lower = q.lower()
        answer_parts = []
        recommendations = []

        # ── Helper: compute grouped stats ─────────────────────────────────
        def _group_stats(col):
            g = df.groupby(col).agg(
                total=(fraud_col, "count"),
                fraud=(fraud_col, lambda x: int(x.astype(int).sum())),
            ).reset_index()
            g["rate"] = (g["fraud"] / g["total"] * 100).round(1)
            return g

        # ── Detect rate / statistical intent ──────────────────────────────
        is_rate_query = bool(re.search(
            r"\b(rate|percentage|ratio|statistic|level|breakdown|distribution)\b",
            q_lower,
        ))

        # ── Region / area filter ───────────────────────────────────────────
        requested_region = None
        geo_mentioned = None
        if area_col:
            available_areas = [v.lower() for v in df[area_col].dropna().unique()]
            for area in available_areas:
                if area in q_lower:
                    requested_region = area
                    break
            geo_regions = ["northeast", "northwest", "southeast", "southwest",
                           "north", "south", "east", "west", "central", "midwest"]
            geo_mentioned = next((r for r in geo_regions if r in q_lower), None)

        # ── Policy type filter ─────────────────────────────────────────────
        requested_policy = None
        if policy_col:
            available_policies = [v.lower() for v in df[policy_col].dropna().unique()]
            for pol in available_policies:
                if any(word in q_lower for word in pol.split()):
                    requested_policy = pol
                    break

        # ── Build the answer ───────────────────────────────────────────────

        # CASE 1: Known area rate query (Urban / Rural)
        if is_rate_query and requested_region and area_col and fraud_col:
            sub = df[df[area_col].str.lower() == requested_region]
            sub_total = len(sub)
            sub_fraud = int(sub[fraud_col].astype(int).sum())
            sub_rate = round(sub_fraud / max(sub_total, 1) * 100, 1)
            answer_parts.append(
                f"The fraud rate in {requested_region.capitalize()} areas is "
                f"{sub_rate}% ({sub_fraud} fraudulent out of {sub_total} claims)."
            )
            recommendations.append(
                f"Drill into {requested_region.capitalize()} claims on the Analytics page."
            )

        # CASE 2: Geographic region NOT in data (e.g. "Northeast")
        elif is_rate_query and geo_mentioned and area_col and fraud_col:
            answer_parts.append(
                f"The dataset does not contain a '{geo_mentioned.capitalize()}' region. "
                f"Claims are categorised by Accident Area: Urban and Rural."
            )
            g = _group_stats(area_col)
            answer_parts.append(
                f"\nOverall fraud rate across all {total} claims: {overall_rate}%.\n"
                f"Breakdown by Accident Area:\n" + _fmt_breakdown(g, area_col, fraud_col)
            )
            recommendations.append(
                "The Analytics page shows fraud stats by Accident Area and Policy Type."
            )

        # CASE 3: Policy type rate query
        elif is_rate_query and requested_policy and policy_col and fraud_col:
            sub = df[df[policy_col].str.lower() == requested_policy]
            sub_total = len(sub)
            sub_fraud = int(sub[fraud_col].astype(int).sum())
            sub_rate = round(sub_fraud / max(sub_total, 1) * 100, 1)
            answer_parts.append(
                f"The fraud rate for '{requested_policy.title()}' policies is "
                f"{sub_rate}% ({sub_fraud} fraudulent out of {sub_total} claims)."
            )

        # CASE 4: General fraud rate query
        elif is_rate_query and "fraud" in q_lower and fraud_col:
            answer_parts.append(
                f"Overall fraud rate: {overall_rate}% "
                f"({fraud_count} fraudulent claims out of {total} total)."
            )
            if area_col:
                g = _group_stats(area_col)
                answer_parts.append(
                    f"\nBreakdown by Accident Area:\n" + _fmt_breakdown(g, area_col, fraud_col)
                )
            if policy_col:
                g2 = _group_stats(policy_col)
                answer_parts.append(
                    f"\nBreakdown by Policy Type:\n" + _fmt_breakdown(g2, policy_col, fraud_col)
                )
            recommendations.append("Visit the Analytics page for interactive charts.")

        # CASE 5: Date-range count query ("last N months", "in January", etc.)
        else:
            use_claimed = any(w in q_lower for w in ["filed", "submitted", "claimed", "reported"])
            date_col = "monthclaimed" if (use_claimed and "monthclaimed" in df.columns) else "month"
            year_col = "year" if "year" in df.columns else None

            month_match = re.search(r"last\s+(\d+)\s*months?", q_lower)
            named_month = next((m for m in _MONTH_MAP if re.search(rf"\b{m}\b", q_lower)), None)
            filtered_count = None
            sub = df

            if month_match and year_col:
                n = int(month_match.group(1))
                df["_month_num"] = (
                    df[date_col].str[:3].str.lower().map(_MONTH_MAP).fillna(0).astype(int)
                )
                df["_date_int"] = df[year_col].astype(int) * 12 + df["_month_num"]
                max_d = df["_date_int"].max()
                sub = df[df["_date_int"] > max_d - n]
                filtered_count = len(sub)
                max_row = df[df["_date_int"] == max_d].iloc[0]
                max_label = f"{max_row[date_col]} {int(max_row[year_col])}"
                answer_parts.append(
                    f"{filtered_count} claims were "
                    f"{'filed' if use_claimed else 'reported'} "
                    f"in the last {n} month{'s' if n > 1 else ''} of available data "
                    f"(ending {max_label})."
                )
            elif named_month:
                m_num = _MONTH_MAP[named_month]
                df["_month_num"] = (
                    df[date_col].str[:3].str.lower().map(_MONTH_MAP).fillna(0).astype(int)
                )
                sub = df[df["_month_num"] == m_num]
                filtered_count = len(sub)
                answer_parts.append(
                    f"{filtered_count} claims were "
                    f"{'filed' if use_claimed else 'reported'} "
                    f"in {named_month.capitalize()} across all available years."
                )

            if "fraud" in q_lower and filtered_count is not None and fraud_col:
                sub_fraud = sub[sub[fraud_col].astype(int) == 1]
                answer_parts.append(
                    f"Of those, {len(sub_fraud)} "
                    f"({round(len(sub_fraud)/max(filtered_count,1)*100,1)}%) "
                    f"were flagged as potentially fraudulent."
                )
            elif "fraud" in q_lower and filtered_count is None and fraud_col:
                answer_parts.append(
                    f"Out of {total} total claims, {fraud_count} "
                    f"({overall_rate}%) are flagged as potentially fraudulent."
                )

            if not answer_parts:
                answer_parts.append(
                    f"The dataset contains {total} total insurance claims, "
                    f"of which {fraud_count} ({overall_rate}%) are flagged as "
                    f"potentially fraudulent."
                )
                if area_col:
                    g = _group_stats(area_col)
                    answer_parts.append(
                        f"\nBreakdown by Accident Area:\n"
                        + _fmt_breakdown(g, area_col, fraud_col)
                    )

        answer = "\n".join(answer_parts)
        answer += (
            "\n\nNote: data spans historical records (1994–1996). "
            "Visit the Analytics page for interactive breakdowns."
        )

        if not recommendations:
            recommendations = [
                "Use the Analytics page for interactive charts by region and policy type.",
                "Use the Investigate page to analyse specific claim fraud signals.",
            ]

        return {
            "query": q,
            "answer": answer,
            "risk_score": None,
            "fraud_signals": [],
            "fraud_cluster": "none",
            "policy_violations": [],
            "policy_compliance": "compliant",
            "recommendations": recommendations,
            "matched_claims": [],
            "matched_claim_ids": [],
            "model_used": "analytics-direct",
            "escalated": False,
            "cache_hit": False,
            "guardrail_violations": [],
            "eval_scores": {},
            "error": None,
        }

    except Exception as exc:
        _log.warning(
            "[analytics] Unexpected error processing analytics query (%s): %s — "
            "falling back to safe response.",
            type(exc).__name__, exc,
        )
        return _safe_analytics_response(
            q,
            "An error occurred while computing analytics from the dataset. "
            "Please check the backend logs or use the Analytics page directly.",
        )

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

    # 3. Aggregate / count query fast path — answer directly from the full CSV
    if _is_aggregate_query(sanitized):
        logger.info(f"[query] aggregate fast path: {sanitized!r}")
        agg_result = _handle_analytics_query(sanitized)
        if agg_result:
            agg_result["guardrail_violations"] = guard.violations
            return QueryResponse(**agg_result)

    # 4. Cache hit
    cached = cache.get(sanitized)
    if cached:
        cached["cache_hit"] = True
        cached["guardrail_violations"] = guard.violations
        return QueryResponse(**cached)

    # 5. Multi-agent pipeline with hard time budget
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
