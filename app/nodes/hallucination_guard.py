"""Hallucination Guard Node — LangGraph node that scores answer risk."""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.metrics import metrics
from app.state import AgentState

logger = logging.getLogger(__name__)

_SAFE_DEFAULTS = {
    "hallucination_score": 0.0,
    "hallucination_decision": "pass",
    "hallucination_flags": [],
}


def hallucination_guard_node(state: AgentState) -> dict:
    """LangGraph node: evaluate hallucination risk and apply policy."""
    metrics.inc("node_invocations.hallucination_guard")
    t0 = time.perf_counter()
    try:
        return _hallucination_guard_inner(state)
    finally:
        metrics.observe("node_latency.hallucination_guard", time.perf_counter() - t0)


def _hallucination_guard_inner(state: AgentState) -> dict:
    if not settings.hallucination_guard_enabled:
        logger.debug("Hallucination guard disabled — returning defaults")
        return dict(_SAFE_DEFAULTS)

    from app.hallucination_guard import run_hallucination_guard

    citation_report = state.get("citation_report", {})
    post_check_result = state.get("post_check_result", {})
    critical_fact_assessments = state.get("critical_fact_assessments", [])
    answer = state.get("answer", "")

    result = run_hallucination_guard(
        citation_report=citation_report,
        post_check_result=post_check_result,
        critical_fact_assessments=critical_fact_assessments,
        answer=answer,
    )

    decision = result.get("hallucination_decision", "pass")
    metrics.inc(f"hallucination_guard.{decision}")
    metrics.observe("hallucination_guard.score", result.get("hallucination_score", 0.0))

    return result
