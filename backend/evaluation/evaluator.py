"""
RAG Evaluation using DeepEval.
Metrics: Faithfulness, AnswerRelevancy, ContextualPrecision.

Usage:
    python -m backend.evaluation.evaluator --samples 20
"""
import json
import logging
import argparse
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    query: str
    actual_output: str
    expected_output: str
    retrieval_context: list[str]


@dataclass
class EvalResult:
    metric: str
    score: float
    passed: bool
    reason: str = ""


def build_test_cases(samples: list[EvalSample]):
    """Convert EvalSamples to DeepEval LLMTestCase objects."""
    from deepeval.test_case import LLMTestCase
    return [
        LLMTestCase(
            input=s.query,
            actual_output=s.actual_output,
            expected_output=s.expected_output,
            retrieval_context=s.retrieval_context,
        )
        for s in samples
    ]


def run_evaluation(samples: list[EvalSample]) -> list[EvalResult]:
    """
    Run DeepEval metrics on the provided samples.
    Returns list of EvalResult per metric.
    """
    try:
        from deepeval.metrics import (
            FaithfulnessMetric,
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
        )
        from deepeval import evaluate
    except ImportError as e:
        logger.error(f"DeepEval not installed: {e}")
        return []

    test_cases = build_test_cases(samples)

    metrics = [
        FaithfulnessMetric(threshold=0.5, verbose_mode=False),
        AnswerRelevancyMetric(threshold=0.5, verbose_mode=False),
        ContextualPrecisionMetric(threshold=0.5, verbose_mode=False),
    ]

    results = evaluate(test_cases=test_cases, metrics=metrics, print_results=False)

    eval_results = []
    for tc in results.test_results:
        for mr in tc.metrics_data:
            eval_results.append(EvalResult(
                metric=mr.name,
                score=round(mr.score, 4) if mr.score is not None else 0.0,
                passed=mr.success,
                reason=mr.reason or "",
            ))

    return eval_results


def load_sample_queries() -> list[EvalSample]:
    """Generate synthetic evaluation samples for smoke testing."""
    return [
        EvalSample(
            query="Was claim C-1001 fraudulent?",
            actual_output="Based on the retrieved records, claim C-1001 shows indicators of fraud: inflated repair costs and inconsistent witness statements.",
            expected_output="Claim C-1001 was flagged as potentially fraudulent.",
            retrieval_context=[
                "Claim C-1001: Auto collision. Repair estimate $18,000. Witness statements inconsistent.",
                "Fraud indicators: inflated estimates, missing police report.",
            ],
        ),
        EvalSample(
            query="What is the fraud rate for vehicle claims in the Northeast?",
            actual_output="Vehicle claims in the Northeast region show a fraud rate of approximately 8.2% based on historical data.",
            expected_output="The fraud rate for Northeast vehicle claims is around 8%.",
            retrieval_context=[
                "Region: Northeast. Policy type: Vehicle. Fraud label: 1. Count: 82 of 1000.",
            ],
        ),
        EvalSample(
            query="Explain the claim for policy type Home in region West.",
            actual_output="Home policy claims in the West region show lower fraud rates compared to other regions, averaging 4.1%.",
            expected_output="West region home claims have a ~4% fraud rate.",
            retrieval_context=[
                "Region: West. Policy type: Home. Fraud label: 0. Claim amount: $12,000.",
                "West region home claims: 41 fraud cases out of 1000 total.",
            ],
        ),
    ]


def print_summary(results: list[EvalResult]):
    by_metric: dict[str, list[float]] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r.score)

    print("\n=== DeepEval Evaluation Summary ===")
    for metric, scores in by_metric.items():
        avg = sum(scores) / len(scores)
        print(f"  {metric}: avg={avg:.4f} over {len(scores)} samples")
    print("===================================\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run DeepEval on the RAG pipeline.")
    parser.add_argument("--samples", type=int, default=3, help="Number of synthetic samples to evaluate")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    samples = load_sample_queries()[: args.samples]
    logger.info(f"Running evaluation on {len(samples)} samples...")

    results = run_evaluation(samples)

    if results:
        print_summary(results)
        if args.output:
            with open(args.output, "w") as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            logger.info(f"Results saved to {args.output}")
    else:
        logger.warning("No results returned. Check DeepEval installation and API keys.")
