"""
Input guardrails:
1. Empty / length check
2. Prompt injection pattern detection
3. Harmful / unsafe content detection
4. Domain relevance check (insurance-claims domain)
5. Optional Presidio PII anonymisation (disabled by default — requires spaCy model)
   Enable via ENABLE_PII_DETECTION=true in .env
Pattern adopted from AI-Powered-Smart-Grid-Energy-Intelligence-Assistant reference project.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"forget (everything|your instructions)",
    r"system prompt",
    r"<\|.*?\|>",
    r"###\s*instruction",
    r"disregard (all|previous|your)",
]

# Hard-block patterns for dangerous / harmful requests
HARMFUL_PATTERNS = [
    # Explosives / weapons
    r"\b(make|create|build|assemble|fabricate|construct)\s+(a\s+)?(bomb|explosive|grenade|ied|improvised\s+explosive)",
    r"\bbomb\s+(making|creation|instructions|recipe|formula)",
    r"\b(make|manufacture|3d\s*print)\s+(a\s+)?(gun|firearm|pistol|rifle|weapon)",
    # Drugs / controlled substances
    r"\b(make|cook|synthesize|manufacture|produce)\s+(meth(?:amphetamine)?|heroin|cocaine|fentanyl|crack|lsd|mdma)",
    # Violence / harm to people
    r"\bhow\s+to\s+(kill|murder|assassinate|poison|harm)\s+(a\s+)?(person|someone|people|human)",
    r"\b(kill|murder|assassinate)\s+(instructions|guide|method|tutorial)",
    # Malware / destructive cyberattacks
    r"\b(create|write|build|code)\s+(a\s+)?(virus|malware|ransomware|trojan|keylogger|rootkit)",
    r"\bhow\s+to\s+(hack|breach|attack)\s+(a\s+)?(bank|hospital|government|power\s+grid|infrastructure)",
    # Self-harm
    r"\bhow\s+to\s+(commit\s+suicide|kill\s+myself|end\s+my\s+life)",
]

# Keywords that indicate the query is relevant to the insurance-claims domain
_DOMAIN_KEYWORDS = {
    "claim", "claims", "policy", "policies", "coverage", "insured", "insurer",
    "insurance", "premium", "deductible", "fraud", "adjuster", "settlement",
    "liability", "accident", "injury", "damage", "loss", "payout", "risk",
    "investigation", "compliance", "beneficiary", "underwriting", "vehicle",
    "auto", "property", "medical", "health", "report", "incident", "claimant",
    "evidence", "approval", "denial", "appeal", "reimbursement", "assessment",
    "evaluation", "score", "signal", "pattern", "agent", "broker",
    "case", "file", "document", "hospital", "repair", "estimate",
    "witness", "police", "theft", "fire", "flood", "collision", "crash",
}

# Only apply domain check for queries longer than this many unique word tokens
_DOMAIN_MIN_WORDS = 3

_analyzer = None
_anonymizer = None
_presidio_unavailable = False


def _get_presidio():
    global _analyzer, _anonymizer, _presidio_unavailable
    if _presidio_unavailable:
        return None, None
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
            logger.info("Presidio PII engine loaded.")
        except Exception as e:
            logger.warning(f"Presidio unavailable ({e!r}). PII anonymisation disabled.")
            _presidio_unavailable = True
            return None, None
    return _analyzer, _anonymizer


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_query: str
    violations: List[str] = field(default_factory=list)
    fast_reply: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.passed


def _is_harmful(query: str) -> bool:
    q_lower = query.lower()
    for pattern in HARMFUL_PATTERNS:
        if re.search(pattern, q_lower):
            return True
    return False


def _is_off_topic(query: str) -> bool:
    tokens = set(re.findall(r"\b[a-z]+\b", query.lower()))
    if len(tokens) < _DOMAIN_MIN_WORDS:
        return False  # too short to judge domain relevance
    return tokens.isdisjoint(_DOMAIN_KEYWORDS)


def check_input(query: str) -> GuardrailResult:
    # 1. Empty / length
    if not query or not query.strip():
        return GuardrailResult(passed=False, sanitized_query=query, violations=["Empty query"])

    if len(query) > 2000:
        return GuardrailResult(
            passed=False, sanitized_query=query, violations=["Query too long (max 2000 chars)"]
        )

    q_lower = query.lower()

    # 2. Prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return GuardrailResult(
                passed=False,
                sanitized_query=query,
                violations=["Potential prompt injection detected"],
            )

    # 3. Harmful / unsafe content — hard block
    if _is_harmful(query):
        logger.warning(f"[guardrail] Harmful content blocked: {query!r}")
        return GuardrailResult(
            passed=False,
            sanitized_query=query,
            violations=["Harmful or unsafe content detected. This assistant only handles insurance claims queries."],
        )

    # 4. Domain relevance — reject clearly off-topic queries
    if _is_off_topic(query):
        logger.info(f"[guardrail] Off-topic query rejected: {query!r}")
        return GuardrailResult(
            passed=False,
            sanitized_query=query,
            violations=["Query is outside the insurance claims domain. Please ask about claims, fraud, policies, or related topics."],
        )

    sanitized = query
    violations = []

    # 5. PII detection — only if explicitly enabled
    if getattr(settings, "enable_pii_detection", False):
        analyzer, anonymizer = _get_presidio()
        if analyzer:
            try:
                results = analyzer.analyze(text=query, language="en")
                if results:
                    pii_types = list({r.entity_type for r in results})
                    anon = anonymizer.anonymize(text=query, analyzer_results=results)
                    sanitized = anon.text
                    violations.append(f"PII detected and anonymized: {pii_types}")
            except Exception as e:
                logger.warning(f"Presidio analysis failed: {e!r}")

    return GuardrailResult(passed=True, sanitized_query=sanitized, violations=violations)
