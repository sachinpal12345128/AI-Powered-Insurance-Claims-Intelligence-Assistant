"""
Input guardrails:
1. Empty / length check
2. Prompt injection pattern detection
3. Optional Presidio PII anonymisation (disabled by default — requires spaCy model)
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


def check_input(query: str) -> GuardrailResult:
    if not query or not query.strip():
        return GuardrailResult(passed=False, sanitized_query=query, violations=["Empty query"])

    if len(query) > 2000:
        return GuardrailResult(
            passed=False, sanitized_query=query, violations=["Query too long (max 2000 chars)"]
        )

    q_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return GuardrailResult(
                passed=False,
                sanitized_query=query,
                violations=["Potential prompt injection detected"],
            )

    sanitized = query
    violations = []

    # PII detection — only if explicitly enabled
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
