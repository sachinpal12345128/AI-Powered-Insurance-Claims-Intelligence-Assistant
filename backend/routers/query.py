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

# Matches count, rate, regional AND date/latest/earliest lookup queries
_AGG_RE = re.compile(
    r"\b("
    r"how\s+many|count\s+(of\s+)?claims?|total\s+(number\s+of\s+)?claims?|"
    r"number\s+of\s+claims?|claims?\s+filed\s+(in|by|within|after|more|over|during|last)|claims?\s+submitted\s+(in|by|within|after|more|over|during|last)|"
    r"how\s+much|how\s+often|how\s+frequently|"
    r"fraud\s+rate|fraud\s+percentage|fraud\s+ratio|fraud\s+statistic|"
    r"what\s+is\s+the\s+fraud|fraud\s+level|rate\s+of\s+fraud|"
    r"fraud\s+(by|in|for|across|per)\s+\w+|"
    r"claims?\s+(by|in|for|across|per)\s+\w+|"
    r"breakdown|distribution\s+of|statistics\s+(for|on|about)|"
    r"(total|overall)\s+fraud|"
    r"(latest|earliest|most\s+recent|first|last|oldest)\s+(claim|submission|date|record)|"
    r"(date|when)\s+(of\s+)?(the\s+)?(latest|earliest|most\s+recent|first|last|oldest)|"
    r"claim\s+submission\s+date|submission\s+date|date\s+range|"
    r"when\s+(was|were|is|are)\s+(the\s+)?(first|last|latest|earliest|most\s+recent)|"
    r"(filed|submitted|reported)\s+(more\s+than|after|beyond|outside|late|later)\s+\d*\s*(day|month|week)|"
    r"(more\s+than|after|beyond|outside)\s+\d+\s*(day|month|week)s?\s+(after|from|since|of)\s+.*(incident|accident|claim|event)|"
    r"late\s+(filing|submission|claim|claims?)|"
    r"(outside|beyond|exceed|violat).*(30.day|filing\s+window|policy\s+window|time\s+limit)|"
    r"30.day\s+(rule|window|limit|policy|violation|filing)|"
    r"(filing|submission)\s+(window|deadline|violation|delay|limit)|"
    r"(common|top|major|frequent|typical|key|main|most\s+common)\s+(fraud\s+)?(signal|pattern|indicator|factor|trigger|flag)s?|"
    r"(fraud\s+)?(signal|pattern|indicator|factor)s?\s+(in|for|at|across)\s+(urban|rural|\w+\s+area)|"
    r"what\s+(are|were|is)\s+(the\s+)?(common|top|major|frequent|typical|key|main)?\s*(fraud|risk)\s*(signal|pattern|indicator)|"
    r"(compare|show|list|summarize|analyse|analyze)\s+(fraud\s+)?(signal|pattern|indicator|rate|statistic)s?\s*(by|in|for|across)?\s*(area|region|urban|rural|policy|type)?|"
    r"multiple\s+supplements?|supplements?.*(address|change)|address.*(change|supplements?)|"
    r"claims?\s+with\s+(multiple|several|many|3\+|3\s*or\s*more|high)\s+supplements?|"
    r"claims?\s+with.*(address\s+change|changed\s+address)|"
    r"(supplements?|address\s+change).*(fraud|risk|pattern|signal)"
    r")\b",
    re.IGNORECASE,
)

_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "source_data" / "fraud_oracle.csv"

# ── Dataset statistics cache (computed once per server lifetime) ──────────────
_dataset_stats_cache: dict | None = None


def _get_dataset_stats() -> dict | None:
    """Pre-compute comprehensive dataset statistics from the CSV (cached)."""
    global _dataset_stats_cache
    if _dataset_stats_cache is not None:
        return _dataset_stats_cache
    try:
        import pandas as pd
        import json as _json
    except ImportError:
        return None
    if not _CSV_PATH.exists():
        return None
    try:
        df = pd.read_csv(_CSV_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        total = len(df)
        fraud_col  = next((c for c in df.columns if "fraud" in c), None)
        area_col   = next((c for c in df.columns if "area"  in c or "region" in c), None)
        policy_col = next((c for c in df.columns if "policy" in c and "type" in c), None)

        fraud_count  = int(df[fraud_col].astype(int).sum()) if fraud_col else 0
        fraud_df     = df[df[fraud_col].astype(int) == 1] if fraud_col else df

        MONTH_MAP = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
        }

        # Date range
        df["_mnum"]     = df["monthclaimed"].str[:3].str.lower().map(MONTH_MAP).fillna(0).astype(int)
        df["_date_int"] = df["year"].astype(int) * 12 + df["_mnum"]
        max_d = df["_date_int"].max()
        min_d = df[df["_mnum"] > 0]["_date_int"].min()
        max_row = df[df["_date_int"] == max_d].iloc[0]
        _valid   = df[df["_mnum"] > 0]
        min_row  = _valid[_valid["_date_int"] == min_d].iloc[0]
        latest_lbl   = f"{max_row['monthclaimed']} {int(max_row['year'])}"
        earliest_lbl = f"{min_row['monthclaimed']} {int(min_row['year'])}"

        # Last 2 months
        last2 = df[df["_date_int"] > max_d - 2]
        jan   = df[df["_mnum"] == 1]

        # By area
        by_area = {}
        if area_col:
            for area in df[area_col].dropna().unique():
                sub = df[df[area_col] == area]
                sf  = int(sub[fraud_col].astype(int).sum()) if fraud_col else 0
                by_area[area] = {
                    "total": len(sub),
                    "fraud": sf,
                    "fraud_rate_pct": round(sf / max(len(sub), 1) * 100, 1),
                }

        # By policy type
        by_policy = {}
        if policy_col:
            for pol in df[policy_col].dropna().unique():
                sub = df[df[policy_col] == pol]
                sf  = int(sub[fraud_col].astype(int).sum()) if fraud_col else 0
                by_policy[pol] = {
                    "total": len(sub),
                    "fraud": sf,
                    "fraud_rate_pct": round(sf / max(len(sub), 1) * 100, 1),
                }

        # Fraud signals (in fraud claims)
        def sig_pct(col, val):
            if col not in fraud_df.columns:
                return 0
            n = len(fraud_df[fraud_df[col].astype(str).str.strip() == val])
            return {"count": n, "pct_of_fraud_claims": round(n / max(fraud_count, 1) * 100, 1)}

        # Late filings
        df["_acc_mnum"]   = df["month"].str[:3].str.lower().map(MONTH_MAP).fillna(0).astype(int)
        df["_acc_date"]   = df["year"].astype(int) * 12 + df["_acc_mnum"]
        df["_month_diff"] = df["_date_int"] - df["_acc_date"]
        late = df[df["_month_diff"] > 1]
        late_fraud = int(late[fraud_col].astype(int).sum()) if fraud_col else 0

        # Multi supplements + address change
        both = df[
            df["numberofsuppliments"].isin(["3 to 5", "more than 5"]) &
            (df["addresschange_claim"] != "no change")
        ] if "numberofsuppliments" in df.columns and "addresschange_claim" in df.columns else df.head(0)
        both_fraud = int(both[fraud_col].astype(int).sum()) if fraud_col else 0

        stats = {
            "overview": {
                "total_claims": total,
                "total_fraud_claims": fraud_count,
                "fraud_rate_pct": round(fraud_count / max(total, 1) * 100, 1),
                "date_range": f"{earliest_lbl} to {latest_lbl}",
                "years_covered": int(max_row["year"]) - int(min_row["year"]) + 1,
            },
            "date_info": {
                "latest_claim_date": latest_lbl,
                "earliest_claim_date": earliest_lbl,
                "claims_in_last_2_months": len(last2),
                "claims_in_january_all_years": len(jan),
            },
            "by_accident_area": by_area,
            "by_policy_type": by_policy,
            "fraud_signals_in_fraud_claims": {
                "no_police_report":    sig_pct("policereportfiled", "No"),
                "no_witness_present":  sig_pct("witnesspresent", "No"),
                "external_agent":      sig_pct("agenttype", "External"),
                "policy_holder_fault": sig_pct("fault", "Policy Holder"),
                "multiple_supplements_3_plus": {
                    "count": int(fraud_df["numberofsuppliments"].isin(["3 to 5","more than 5"]).sum())
                    if "numberofsuppliments" in fraud_df.columns else 0,
                },
            },
            "late_filings": {
                "claims_filed_more_than_1_month_after_accident": len(late),
                "pct_of_total": round(len(late) / max(total, 1) * 100, 1),
                "fraud_count": late_fraud,
                "fraud_rate_pct": round(late_fraud / max(len(late), 1) * 100, 1),
            },
            "combined_signals": {
                "multiple_supplements_and_address_change": {
                    "count": len(both),
                    "pct_of_total": round(len(both) / max(total, 1) * 100, 1),
                    "fraud_count": both_fraud,
                    "fraud_rate_pct": round(both_fraud / max(len(both), 1) * 100, 1),
                },
            },
            "driver_rating": (lambda dr_col="driverrating": {
                "scale": "1 (worst) to 4 (best)",
                "average_all_claims":       round(float(df[dr_col].mean()), 2)       if dr_col in df.columns else None,
                "average_fraud_claims":     round(float(fraud_df[dr_col].mean()), 2) if dr_col in df.columns else None,
                "average_non_fraud_claims": round(float(df[df[fraud_col].astype(int)==0][dr_col].mean()), 2)
                                            if dr_col in df.columns and fraud_col else None,
                "distribution_in_fraud_claims": {
                    str(k): int(v)
                    for k, v in fraud_df[dr_col].value_counts().sort_index().items()
                } if dr_col in df.columns else {},
                "distribution_all_claims": {
                    str(k): int(v)
                    for k, v in df[dr_col].value_counts().sort_index().items()
                } if dr_col in df.columns else {},
                "note": "Driver rating shows almost no difference between fraud and non-fraud claims — it is NOT a strong fraud predictor.",
            })(),
            "claim_amounts": (lambda amt_col=next((c for c in df.columns if "amount" in c or "price" in c), None): {
                "vehicle_price_categories": df[amt_col].value_counts().to_dict()
                if amt_col else {},
            })(),
            "demographics": {
                "age_groups_in_fraud": fraud_df["age"].describe().round(1).to_dict()
                                       if "age" in fraud_df.columns else {},
                "fault_distribution_all": df["fault"].value_counts().to_dict()
                                          if "fault" in df.columns else {},
                "fault_distribution_fraud": fraud_df["fault"].value_counts().to_dict()
                                            if "fault" in fraud_df.columns else {},
                "vehicle_category_fraud": fraud_df["vehiclecategory"].value_counts().to_dict()
                                          if "vehiclecategory" in fraud_df.columns else {},
            },
        }
        _dataset_stats_cache = stats
        return stats
    except Exception as exc:
        logging.getLogger(__name__).warning("[stats] Failed to build dataset stats: %s", exc)
        return None


def _smart_analytics_handler(q: str) -> dict | None:
    """
    Single LLM call that:
      1. Classifies query intent — ANALYTICS or INVESTIGATIVE.
      2. For ANALYTICS: formulates a precise answer from pre-computed dataset stats.
      3. For INVESTIGATIVE (or any ambiguity): returns None → falls through to RAG.
    Eliminates the need to add new regex patterns for every analytics query type.
    """
    _log = logging.getLogger(__name__)
    stats = _get_dataset_stats()
    if stats is None:
        return None

    # Queries that clearly need specific claim retrieval must go to RAG.
    # Guard against the LLM misclassifying investigative filter queries.
    _investigative_hints = re.compile(
        r"\b(show\s+me|find|retrieve|list|get|display|search|look\s+up)\b"
        r"|\bno\s+police\s+report\b"
        r"|\bexternal\s+agent\b"
        r"|\bwithout\s+(police|witness)\b"
        r"|\bcollision\s+policy\b"
        r"|\bclaim\s+id\b|\bpol-\d+\b",
        re.IGNORECASE,
    )
    if _investigative_hints.search(q):
        _log.info("[smart_router] investigative hint detected → RAG pipeline: %r", q)
        return None

    try:
        import json as _json
        from backend.config.llm import get_llm_with_fallback
        llm = get_llm_with_fallback()
        prompt = (
            "You are an insurance analytics assistant with access to pre-computed "
            "dataset statistics.\n\n"
            "TASK: Classify the query and respond accordingly.\n\n"
            "ANALYTICS queries ask for: counts, rates, percentages, totals, averages, "
            "date ranges, summaries, or breakdowns across the WHOLE dataset "
            "(e.g. 'how many', 'what is the rate', 'which month', 'what percentage').\n\n"
            "INVESTIGATIVE queries ask to: find/show/retrieve specific claims, investigate "
            "individual claims, or analyse fraud signals for specific claim characteristics.\n\n"
            "RULES:\n"
            "- If INVESTIGATIVE → respond with exactly the single word: INVESTIGATIVE\n"
            "- If ANALYTICS → respond with a clear factual answer using ONLY the numbers "
            "from the dataset statistics below. Do NOT just say the word ANALYTICS. "
            "Write a proper answer sentence with actual numbers.\n"
            "- If unsure → respond with: INVESTIGATIVE\n\n"
            f"Query: {q}\n\n"
            f"Dataset Statistics:\n{_json.dumps(stats, indent=2)}"
        )
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Route to RAG if: LLM says INVESTIGATIVE, answer is too short,
        # or answer is just a classification word with no actual data.
        _bad_response = (
            answer.upper() == "INVESTIGATIVE"
            or answer.upper().startswith("INVESTIGATIVE")
            or answer.upper() == "ANALYTICS"
            or len(answer.split()) < 8          # too short to be a real answer
        )
        if _bad_response:
            _log.info("[smart_router] routed to RAG (response: %r)", answer[:60])
            return None

        _log.info("[smart_router] analytics answer generated for %r", q[:60])
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
            "model_used": "analytics-llm",
            "escalated": False,
            "cache_hit": False,
            "guardrail_violations": [],
            "eval_scores": {},
            "error": None,
        }
    except Exception as exc:
        _log.warning("[smart_router] failed (%s) — falling through to RAG", exc)
        return None


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

        # CASE 5: Date lookup — latest / earliest / most-recent claim date
        elif re.search(
            r"\b(latest|earliest|most\s+recent|first|last|oldest)\b.*\b(claim|submission|date|record)\b"
            r"|\b(date|when)\b.*\b(latest|earliest|most\s+recent|first|last|oldest)\b"
            r"|\bclaim\s+submission\s+date\b"
            r"|\bdate\s+of\s+the\s+(latest|earliest|most\s+recent|first|last|oldest)\b"
            r"|\bwhen\s+(was|were|is|are)\b",
            q_lower,
        ):
            year_col = "year" if "year" in df.columns else None
            if year_col and "monthclaimed" in df.columns:
                df["_mnum"]     = df["monthclaimed"].str[:3].str.lower().map(_MONTH_MAP).fillna(0).astype(int)
                df["_date_int"] = df[year_col].astype(int) * 12 + df["_mnum"]

                # Latest
                max_d      = df["_date_int"].max()
                max_row    = df[df["_date_int"] == max_d].iloc[0]
                latest_lbl = f"{max_row['monthclaimed']} {int(max_row[year_col])}"
                latest_cnt = len(df[df["_date_int"] == max_d])

                # Earliest (exclude month=0 / unknown)
                valid      = df[df["_mnum"] > 0]
                min_d      = valid["_date_int"].min()
                min_row    = valid[valid["_date_int"] == min_d].iloc[0]
                earliest_lbl = f"{min_row['monthclaimed']} {int(min_row[year_col])}"
                earliest_cnt = len(valid[valid["_date_int"] == min_d])

                is_earliest = bool(re.search(r"\b(earliest|first|oldest)\b", q_lower))
                if is_earliest:
                    answer_parts.append(
                        f"The earliest claim submission in the dataset is "
                        f"{earliest_lbl} ({earliest_cnt} claims filed in that month)."
                    )
                else:
                    answer_parts.append(
                        f"The latest claim submission in the dataset is "
                        f"{latest_lbl} ({latest_cnt} claims filed in that month)."
                    )
                answer_parts.append(
                    f"The dataset spans from {earliest_lbl} to {latest_lbl} "
                    f"covering {int(max_row[year_col]) - int(min_row[year_col]) + 1} years "
                    f"with {total} total claims."
                )
                recommendations.append(
                    "Use the Analytics page for a full monthly breakdown of claim submissions."
                )
            else:
                answer_parts.append(
                    f"The dataset contains {total} claims but date columns "
                    f"(monthclaimed / year) could not be found."
                )

        # CASE 6: Fraud signals / patterns by area or general
        elif re.search(
            r"(common|top|major|frequent|typical|key|main|most\s+common)\s+(fraud\s+)?(signal|pattern|indicator|factor|trigger|flag)s?"
            r"|(fraud\s+)?(signal|pattern|indicator|factor)s?\s+(in|for|at|across)\s+\w+"
            r"|what\s+(are|were|is)\s+(the\s+)?(common|top|major|frequent|typical|key|main)?\s*(fraud|risk)\s*(signal|pattern|indicator)"
            r"|(compare|show|list|summarize|analyse|analyze)\s+(fraud\s+)?(signal|pattern|indicator|rate|statistic)s?",
            q_lower,
        ):
            # Detect area filter (urban / rural) from query
            target_area = None
            if area_col:
                available_areas = [v.lower() for v in df[area_col].dropna().unique()]
                target_area = next((a for a in available_areas if a in q_lower), None)

            base = df[df[fraud_col].astype(int) == 1] if fraud_col else df
            if target_area and area_col:
                subset = base[base[area_col].str.lower() == target_area]
                area_label = target_area.capitalize()
            else:
                subset = base
                area_label = "Overall"

            subset_total = len(subset)
            all_claims_in_area = (
                df[df[area_col].str.lower() == target_area] if target_area and area_col else df
            )
            area_total = len(all_claims_in_area)

            if subset_total == 0:
                answer_parts.append(
                    f"No fraud claims found for the specified area in the dataset."
                )
            else:
                answer_parts.append(
                    f"Analysis of {subset_total} fraud claims in {area_label} areas "
                    f"({round(subset_total/max(area_total,1)*100,1)}% fraud rate "
                    f"out of {area_total} total {area_label} claims):\n"
                )

                # Build signal stats
                signal_stats = []

                if "policereportfiled" in subset.columns:
                    no_pol = len(subset[subset["policereportfiled"].astype(str).str.strip() == "No"])
                    signal_stats.append((
                        "No police report filed",
                        no_pol, subset_total, "HIGH"
                    ))

                if "witnesspresent" in subset.columns:
                    no_wit = len(subset[subset["witnesspresent"].astype(str).str.strip() == "No"])
                    signal_stats.append((
                        "No witness present",
                        no_wit, subset_total, "MEDIUM"
                    ))

                if "agenttype" in subset.columns:
                    ext_agt = len(subset[subset["agenttype"].astype(str).str.strip() == "External"])
                    signal_stats.append((
                        "External agent involvement",
                        ext_agt, subset_total, "HIGH"
                    ))

                if "numberofsuppliments" in subset.columns:
                    multi_sup = len(subset[subset["numberofsuppliments"].astype(str).str.strip().isin(["3 to 5", "more than 5"])])
                    signal_stats.append((
                        "Multiple supplements (3+)",
                        multi_sup, subset_total, "MEDIUM"
                    ))

                if "fault" in subset.columns:
                    ph_fault = len(subset[subset["fault"].astype(str).str.strip() == "Policy Holder"])
                    signal_stats.append((
                        "Policy Holder at fault",
                        ph_fault, subset_total, "INFO"
                    ))

                for sig_name, sig_count, sig_base, severity in signal_stats:
                    pct = round(sig_count / max(sig_base, 1) * 100, 1)
                    answer_parts.append(
                        f"  • {sig_name} [{severity}]: {sig_count} of {sig_base} "
                        f"fraud claims ({pct}%)"
                    )

                # Top policy types
                if policy_col and len(subset) > 0:
                    top_pol = (
                        subset.groupby(policy_col).size()
                        .sort_values(ascending=False).head(3)
                    )
                    pol_str = ", ".join(
                        f"{p} ({n})" for p, n in top_pol.items()
                    )
                    answer_parts.append(f"\nTop policy types in {area_label} fraud: {pol_str}")

                recommendations.extend([
                    f"Focus investigations on the 'No police report + External agent' "
                    f"combination — it appears in the vast majority of {area_label} fraud cases.",
                    "Request police report and independent witness verification for all flagged claims.",
                    "Use the Analytics page for interactive fraud signal charts.",
                ])

        # CASE 7: Multi-condition attribute filter — supplements / address change
        elif re.search(
            r"multiple\s+supplements?|supplements?.*(address|change)|address.*(change|supplements?)"
            r"|claims?\s+with\s+(multiple|several|many|3\+|3\s*or\s*more|high)\s+supplements?"
            r"|claims?\s+with.*(address\s+change|changed\s+address)"
            r"|(supplements?|address\s+change).*(fraud|risk|pattern|signal)",
            q_lower,
        ):
            want_supplements   = bool(re.search(r"supplement", q_lower))
            want_addr_change   = bool(re.search(r"address.*(change|chang)|changed.*address", q_lower))

            mask = pd.Series([True] * len(df), index=df.index)

            # Multiple supplements = 3 to 5 OR more than 5
            if want_supplements and "numberofsuppliments" in df.columns:
                mask &= df["numberofsuppliments"].isin(["3 to 5", "more than 5"])

            # Address change = any value other than "no change"
            if want_addr_change and "addresschange_claim" in df.columns:
                mask &= df["addresschange_claim"] != "no change"

            subset      = df[mask]
            subset_total = len(subset)
            sub_fraud   = int(subset[fraud_col].astype(int).sum()) if fraud_col else 0
            sub_rate    = round(sub_fraud / max(subset_total, 1) * 100, 1)

            # Build human-readable condition label
            conds = []
            if want_supplements:   conds.append("multiple supplements (3+)")
            if want_addr_change:   conds.append("address change near claim")
            cond_label = " AND ".join(conds) if conds else "specified conditions"

            answer_parts.append(
                f"{subset_total} claims ({round(subset_total/max(total,1)*100,1)}% of dataset) "
                f"match the condition: {cond_label}."
            )
            answer_parts.append(
                f"Fraud rate in these claims: {sub_rate}% ({sub_fraud} of {subset_total}) "
                f"— compared to the overall dataset fraud rate of {overall_rate}%."
            )

            # Supplements breakdown
            if want_supplements and "numberofsuppliments" in subset.columns and subset_total > 0:
                sup_counts = subset["numberofsuppliments"].value_counts()
                sup_lines = "  " + "\n  ".join(
                    f"• {k}: {v} claims" for k, v in sup_counts.items()
                )
                answer_parts.append(f"\nSupplements breakdown:\n{sup_lines}")

            # Address change timing breakdown
            if want_addr_change and "addresschange_claim" in subset.columns and subset_total > 0:
                addr_counts = subset["addresschange_claim"].value_counts()
                addr_lines = "  " + "\n  ".join(
                    f"• {k}: {v} claims" for k, v in addr_counts.items()
                )
                answer_parts.append(f"\nAddress change timing:\n{addr_lines}")

            # Co-occurring fraud signals
            if subset_total > 0:
                co_signals = []
                if "policereportfiled" in subset.columns:
                    n = len(subset[subset["policereportfiled"] == "No"])
                    co_signals.append(f"No police report: {n}/{subset_total} ({round(n/subset_total*100,1)}%)")
                if "witnesspresent" in subset.columns:
                    n = len(subset[subset["witnesspresent"] == "No"])
                    co_signals.append(f"No witness: {n}/{subset_total} ({round(n/subset_total*100,1)}%)")
                if "agenttype" in subset.columns:
                    n = len(subset[subset["agenttype"] == "External"])
                    co_signals.append(f"External agent: {n}/{subset_total} ({round(n/subset_total*100,1)}%)")
                if co_signals:
                    answer_parts.append(
                        "\nCo-occurring fraud signals:\n  " + "\n  ".join(f"• {s}" for s in co_signals)
                    )

            # Area breakdown
            if area_col and subset_total > 0:
                area_counts = subset[area_col].value_counts()
                answer_parts.append(
                    "By Accident Area: " +
                    "  ".join(f"{a}: {n}" for a, n in area_counts.items())
                )

            recommendations.extend([
                f"All {subset_total} claims matching '{cond_label}' should be queued for manual review.",
                f"The {sub_rate}% fraud rate in this group "
                + ("exceeds" if sub_rate > overall_rate else "is comparable to")
                + f" the overall {overall_rate}% rate.",
                "Cross-reference supplement invoices with repair shop records to verify legitimacy.",
                "Verify address change timing against policy documents for suspicious patterns.",
            ])

        # CASE 8: Late-filing / 30-day policy window queries
        elif re.search(
            r"(filed|submitted|reported)\s+(more\s+than|after|beyond|outside|late|later)\s*\d*\s*(day|month|week)"
            r"|(more\s+than|after|beyond)\s+\d+\s*(day|month|week)s?\s+(after|from|since)"
            r"|late\s+(filing|submission|claim)"
            r"|(outside|beyond|exceed|violat).*(30.day|filing\s+window|time\s+limit)"
            r"|30.day\s+(rule|window|limit|policy|violation|filing)"
            r"|(filing|submission)\s+(window|deadline|violation|delay|limit)",
            q_lower,
        ):
            if "month" in df.columns and "monthclaimed" in df.columns and "year" in df.columns:
                df["_acc_mnum"]   = df["month"].str[:3].str.lower().map(_MONTH_MAP).fillna(0).astype(int)
                df["_claim_mnum"] = df["monthclaimed"].str[:3].str.lower().map(_MONTH_MAP).fillna(0).astype(int)
                df["_acc_date"]   = df["year"].astype(int) * 12 + df["_acc_mnum"]
                df["_claim_date"] = df["year"].astype(int) * 12 + df["_claim_mnum"]
                df["_month_diff"] = df["_claim_date"] - df["_acc_date"]

                # >1 month gap between accident and claim ≈ >30 days
                late = df[df["_month_diff"] > 1]
                late_count = len(late)
                late_pct   = round(late_count / max(total, 1) * 100, 1)

                # Fraud rate in late claims vs overall
                late_fraud = int(late[fraud_col].astype(int).sum()) if fraud_col else 0
                late_fraud_rate = round(late_fraud / max(late_count, 1) * 100, 1)

                answer_parts.append(
                    f"{late_count} claims ({late_pct}% of dataset) were filed "
                    f"more than 30 days after the incident."
                )
                answer_parts.append(
                    f"Fraud rate in late-filed claims: {late_fraud_rate}% "
                    f"({late_fraud} of {late_count}) — compared to the overall "
                    f"dataset fraud rate of {overall_rate}%."
                )

                # Breakdown by policy type
                if policy_col and late_count > 0:
                    top_policies = (
                        late.groupby(policy_col).size()
                        .sort_values(ascending=False)
                        .head(5)
                    )
                    policy_lines = "  " + "\n  ".join(
                        f"• {pol}: {cnt} claims"
                        for pol, cnt in top_policies.items()
                    )
                    answer_parts.append(f"\nBreakdown by Policy Type:\n{policy_lines}")

                # Breakdown by area
                if area_col and late_count > 0:
                    area_split = late.groupby(area_col).size().sort_values(ascending=False)
                    area_lines = "  " + "  ".join(
                        f"{a}: {n}" for a, n in area_split.items()
                    )
                    answer_parts.append(f"By Accident Area: {area_lines}")

                recommendations.extend([
                    "Flag late-filed claims for mandatory policy compliance review.",
                    "Cross-reference late claims with the 30-day filing window policy rule.",
                    f"The {late_fraud_rate}% fraud rate in late claims exceeds the {overall_rate}% "
                    f"overall rate — prioritise these for investigation.",
                ])
            else:
                answer_parts.append(
                    "Could not compute late-filing statistics: "
                    "date columns (month / monthclaimed / year) not found in dataset."
                )

        # CASE 9: Date-range count query ("last N months", "in January", etc.)
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

    # 3a. Regex fast path — known analytics patterns (zero extra latency)
    if _is_aggregate_query(sanitized):
        logger.info(f"[query] regex analytics fast path: {sanitized!r}")
        agg_result = _handle_analytics_query(sanitized)
        if agg_result:
            agg_result["guardrail_violations"] = guard.violations
            return QueryResponse(**agg_result)

    # 3b. Smart router — LLM classifies unknown queries as analytics or investigative.
    #     This removes the need to add regex patterns for every new analytics query type.
    #     If the LLM says "analytics" it answers from pre-computed dataset stats.
    #     If the LLM says "investigative" (or fails) it falls through to the RAG pipeline.
    else:
        logger.info(f"[query] smart router: classifying {sanitized!r}")
        smart_result = _smart_analytics_handler(sanitized)
        if smart_result:
            smart_result["guardrail_violations"] = guard.violations
            return QueryResponse(**smart_result)

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
