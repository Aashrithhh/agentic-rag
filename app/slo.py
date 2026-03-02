"""Service Level Objectives — hard targets for latency, errors, cost.

Defines measurable SLO thresholds and provides enforcement utilities
for CI/CD pipelines and runtime alerting.

Targets:
  - p95 end-to-end latency  ≤ 30 s
  - p95 per-node latency    ≤ 10 s  (retriever, grader, generator, etc.)
  - Error rate              ≤  2 %  (pipeline failures / total queries)
  - Timeout rate            ≤  1 %  (queries exceeding 60 s hard cap)
  - Cost per query          ≤ $0.15 (LLM tokens + embedding calls)
  - Faithfulness score      ≥ 0.70  (LLM-as-judge, nightly eval)
  - Relevance score         ≥ 0.75  (LLM-as-judge, nightly eval)
  - Recall@k                ≥ 0.60  (retrieval quality, gold eval set)
  - MRR                     ≥ 0.50  (retrieval ranking quality)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── SLO threshold definitions ────────────────────────────────────────


@dataclass(frozen=True)
class SLOThresholds:
    """Hard SLO targets. Fail CI/CD if load tests exceed these."""

    # Latency (seconds)
    p95_latency_seconds: float = 30.0
    p95_node_latency_seconds: float = 10.0
    p99_latency_seconds: float = 45.0
    hard_timeout_seconds: float = 60.0

    # Error rates (fractions, 0.0–1.0)
    max_error_rate: float = 0.02          # 2%
    max_timeout_rate: float = 0.01        # 1%

    # Cost
    max_cost_per_query_usd: float = 0.15

    # Quality (from nightly eval)
    min_faithfulness: float = 0.70
    min_relevance: float = 0.75
    min_recall_at_k: float = 0.60
    min_mrr: float = 0.50

    # Cache
    min_cache_hit_rate: float = 0.30      # 30% minimum cache hit rate


# Module-level default thresholds
SLO = SLOThresholds()


# ── SLO check result ─────────────────────────────────────────────────


@dataclass
class SLOViolation:
    """A single SLO violation with details."""
    metric: str
    threshold: float
    actual: float
    severity: Literal["warning", "critical"] = "critical"
    message: str = ""

    def __post_init__(self):
        if not self.message:
            self.message = (
                f"SLO violation: {self.metric} = {self.actual:.4f} "
                f"(threshold: {self.threshold:.4f})"
            )


@dataclass
class SLOReport:
    """Aggregate SLO check result."""
    passed: bool = True
    violations: list[SLOViolation] = field(default_factory=list)
    metrics_checked: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""
    test_run_id: str = ""

    def add_violation(self, violation: SLOViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "critical":
            self.passed = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": [asdict(v) for v in self.violations],
            "metrics_checked": self.metrics_checked,
            "timestamp": self.timestamp,
            "test_run_id": self.test_run_id,
            "num_violations": len(self.violations),
            "num_critical": sum(1 for v in self.violations if v.severity == "critical"),
        }

    def summary(self) -> str:
        status = "PASS ✅" if self.passed else "FAIL ❌"
        lines = [f"SLO Report: {status}"]
        for v in self.violations:
            icon = "🔴" if v.severity == "critical" else "🟡"
            lines.append(f"  {icon} {v.message}")
        return "\n".join(lines)


# ── SLO checker ──────────────────────────────────────────────────────


def check_latency_slos(
    histogram_data: dict[str, dict[str, float]],
    thresholds: SLOThresholds = SLO,
) -> list[SLOViolation]:
    """Check latency histograms against SLO thresholds."""
    violations: list[SLOViolation] = []

    # End-to-end p95 latency
    for name, stats in histogram_data.items():
        if "http_request_seconds" in name and "query" in name:
            p95 = stats.get("p95", 0)
            if p95 > thresholds.p95_latency_seconds:
                violations.append(SLOViolation(
                    metric=f"{name}.p95",
                    threshold=thresholds.p95_latency_seconds,
                    actual=p95,
                ))
            p99 = stats.get("p99", 0)
            if p99 > thresholds.p99_latency_seconds:
                violations.append(SLOViolation(
                    metric=f"{name}.p99",
                    threshold=thresholds.p99_latency_seconds,
                    actual=p99,
                ))

    # Per-node latency
    for name, stats in histogram_data.items():
        if "node_latency" in name:
            p95 = stats.get("p95", 0)
            if p95 > thresholds.p95_node_latency_seconds:
                violations.append(SLOViolation(
                    metric=f"{name}.p95",
                    threshold=thresholds.p95_node_latency_seconds,
                    actual=p95,
                ))

    return violations


def check_error_rate_slos(
    counters: dict[str, int],
    thresholds: SLOThresholds = SLO,
) -> list[SLOViolation]:
    """Check error and timeout rates against SLO thresholds."""
    violations: list[SLOViolation] = []

    total_queries = sum(v for k, v in counters.items() if "node_invocations.retriever" in k)
    total_errors = sum(v for k, v in counters.items()
                       if "llm_call_failure" in k or "db_call_failure" in k)
    total_timeouts = sum(v for k, v in counters.items() if "timeout" in k.lower())

    if total_queries > 0:
        error_rate = total_errors / total_queries
        if error_rate > thresholds.max_error_rate:
            violations.append(SLOViolation(
                metric="error_rate",
                threshold=thresholds.max_error_rate,
                actual=error_rate,
            ))

        timeout_rate = total_timeouts / total_queries
        if timeout_rate > thresholds.max_timeout_rate:
            violations.append(SLOViolation(
                metric="timeout_rate",
                threshold=thresholds.max_timeout_rate,
                actual=timeout_rate,
            ))

    return violations


def check_quality_slos(
    eval_report: dict[str, Any],
    thresholds: SLOThresholds = SLO,
) -> list[SLOViolation]:
    """Check evaluation quality metrics against SLO thresholds."""
    violations: list[SLOViolation] = []

    aq = eval_report.get("answer_quality", {})
    rm = eval_report.get("retrieval_metrics", {})

    faithfulness = aq.get("avg_faithfulness", 0)
    if faithfulness < thresholds.min_faithfulness:
        violations.append(SLOViolation(
            metric="avg_faithfulness",
            threshold=thresholds.min_faithfulness,
            actual=faithfulness,
        ))

    relevance = aq.get("avg_relevance", 0)
    if relevance < thresholds.min_relevance:
        violations.append(SLOViolation(
            metric="avg_relevance",
            threshold=thresholds.min_relevance,
            actual=relevance,
        ))

    recall = rm.get("avg_recall_at_k")
    if isinstance(recall, (int, float)) and recall < thresholds.min_recall_at_k:
        violations.append(SLOViolation(
            metric="avg_recall_at_k",
            threshold=thresholds.min_recall_at_k,
            actual=recall,
        ))

    mrr = rm.get("avg_mrr")
    if isinstance(mrr, (int, float)) and mrr < thresholds.min_mrr:
        violations.append(SLOViolation(
            metric="avg_mrr",
            threshold=thresholds.min_mrr,
            actual=mrr,
        ))

    return violations


def check_cost_slo(
    total_cost_usd: float,
    num_queries: int,
    thresholds: SLOThresholds = SLO,
) -> list[SLOViolation]:
    """Check cost-per-query against SLO threshold."""
    if num_queries == 0:
        return []
    cost_per_query = total_cost_usd / num_queries
    if cost_per_query > thresholds.max_cost_per_query_usd:
        return [SLOViolation(
            metric="cost_per_query_usd",
            threshold=thresholds.max_cost_per_query_usd,
            actual=cost_per_query,
        )]
    return []


def full_slo_check(
    metrics_snapshot: dict[str, Any],
    eval_report: dict[str, Any] | None = None,
    total_cost_usd: float = 0.0,
    num_queries: int = 0,
    thresholds: SLOThresholds = SLO,
) -> SLOReport:
    """Run all SLO checks and produce a unified report."""
    import datetime

    report = SLOReport(
        timestamp=datetime.datetime.utcnow().isoformat(),
        test_run_id=f"slo-{int(time.time())}",
    )

    counters = metrics_snapshot.get("counters", {})
    histograms = metrics_snapshot.get("histograms", {})

    # Latency SLOs
    latency_violations = check_latency_slos(histograms, thresholds)
    for v in latency_violations:
        report.add_violation(v)

    # Error / timeout SLOs
    error_violations = check_error_rate_slos(counters, thresholds)
    for v in error_violations:
        report.add_violation(v)

    # Cost SLO
    cost_violations = check_cost_slo(total_cost_usd, num_queries, thresholds)
    for v in cost_violations:
        report.add_violation(v)

    # Quality SLOs (only if eval report available)
    if eval_report:
        quality_violations = check_quality_slos(eval_report, thresholds)
        for v in quality_violations:
            report.add_violation(v)

    # Track checked metrics
    report.metrics_checked = {
        "total_queries": sum(v for k, v in counters.items()
                            if "node_invocations.retriever" in k),
        "total_errors": sum(v for k, v in counters.items()
                           if "failure" in k),
        "histograms_checked": len(histograms),
        "eval_available": eval_report is not None,
    }

    return report


def save_slo_report(report: SLOReport, path: str | Path) -> None:
    """Persist SLO report as JSON for CI artifacts."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info("SLO report saved to %s", p)
