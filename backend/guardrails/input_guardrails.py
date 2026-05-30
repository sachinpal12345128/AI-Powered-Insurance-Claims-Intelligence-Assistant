"""
Input guardrails:
1. PII detection + anonymization via Presidio
2. Prompt injection detection
3. Empty / irrelevant input check
"""
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

INJECTION_PATTERNS = [
    r"ignore (previous|all|above) instructions",
    r"you are now",
    r"forget (everything|your instructions)",
    r"system prompt",
    r"<\|.*?\|>",
    r"###\s*instruction",
]

_analyzer = None
_anonymizer = None


def _get_presidio():
    global _analyzer, _anonymizer
    if _analyzer is None:
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
        except Exception as e:
            logger.warning(f"Presidio not available: {e}. PII check disabled.")
    return _analyzer, _anonymizer


@dataclass
class GuardrailResult:
    passed: bool
    sanitized_query: str
    violations: list[str]


def check_input(query: str) -> GuardrailResult:
    violations = []

    # 1. Empty check
    if not query or not query.strip():
        return GuardrailResult(False, query, ["Empty query"])

    # 2. Length check
    if len(query) > 2000:
        violations.append("Query too long (max 2000 chars)")
        return GuardrailResult(False, query, violations)

    # 3. Injection check
    q_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            violations.append(f"Potential prompt injection detected")
            return GuardrailResult(False, query, violations)

    # 4. PII anonymization
    sanitized = query
    analyzer, anonymizer = _get_presidio()
    if analyzer:
        try:
            results = analyzer.analyze(text=query, language="en")
            if results:
                pii_types = list({r.entity_type for r in results})
                anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
                sanitized = anonymized.text
                violations.append(f"PII detected and anonymized: {pii_types}")
                logger.info(f"PII anonymized: {pii_types}")
        except Exception as e:
            logger.warning(f"Presidio analysis failed: {e}")

    return GuardrailResult(passed=True, sanitized_query=sanitized, violations=violations)
