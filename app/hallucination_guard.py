"""Hallucination Guard — scores answer risk using existing grounding signals.

Uses citation validation results, unsupported-claim post-check results, and
critical fact assessments to produce a composite hallucination risk score and
a pass | warn | block decision.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_SEVERITY_WEIGHTS = {"critical": 1.0, "moderate": 0.6, "minor": 0.3}


def compute_hallucination_score(
    citation_report: dict[str, Any],
    post_check_result: dict[str, Any],
    critical_fact_assessments: list[dict[str, Any]],
) -> tuple[float, list[dict[str, str]]]:
    """Compute a 0..1 hallucination risk score from grounding signals.

    Returns (score, flags) where *flags* is a list of human-readable reason
    dicts like ``{"type": "citation", "detail": "3/8 invalid citations"}``.
    """
    flags: list[dict[str, str]] = []

    # ── Invalid citation ratio ──────────────────────────────────────
    total_citations = citation_report.get("total_citations", 0)
    invalid_citations = citation_report.get("invalid_citations", 0)
    citation_ratio = invalid_citations / max(total_citations, 1)
    if invalid_citations > 0:
        flags.append({
            "type": "citation",
            "detail": f"{invalid_citations}/{total_citations} invalid citations",
        })

    # ── Unsupported claims severity ─────────────────────────────────
    unsupported_claims = post_check_result.get("unsupported_claims", [])
    if unsupported_claims:
        severity_sum = sum(
            _SEVERITY_WEIGHTS.get(c.get("severity", "minor"), 0.3)
            for c in unsupported_claims
        )
        max_possible = len(unsupported_claims) * 1.0
        claims_score = severity_sum / max(max_possible, 1.0)
        claims_score = min(claims_score, 1.0)

        critical_count = sum(1 for c in unsupported_claims if c.get("severity") == "critical")
        if critical_count:
            flags.append({
                "type": "unsupported_claim",
                "detail": f"{critical_count} critical unsupported claims",
            })
        non_critical = len(unsupported_claims) - critical_count
        if non_critical and not critical_count:
            flags.append({
                "type": "unsupported_claim",
                "detail": f"{len(unsupported_claims)} unsupported claims",
            })
    else:
        claims_score = 0.0

    # ── Weak critical facts ratio ───────────────────────────────────
    total_critical = len(critical_fact_assessments)
    weak_count = sum(
        1 for a in critical_fact_assessments
        if not a.get("is_sufficiently_supported", True)
    )
    critical_fact_ratio = weak_count / max(total_critical, 1)
    if weak_count > 0:
        flags.append({
            "type": "critical_fact",
            "detail": f"{weak_count}/{total_critical} weak critical facts",
        })

    # ── Weighted sum ────────────────────────────────────────────────
    raw_score = (
        settings.hallucination_invalid_citation_weight * citation_ratio
        + settings.hallucination_unsupported_claim_weight * claims_score
        + settings.hallucination_critical_fact_weight * critical_fact_ratio
    )
    score = max(0.0, min(1.0, raw_score))

    return score, flags


def decide_hallucination(score: float) -> str:
    """Return ``"pass"``, ``"warn"``, or ``"block"`` based on threshold config."""
    if score >= settings.hallucination_block_threshold:
        return "block"
    if score >= settings.hallucination_warn_threshold:
        return "warn"
    return "pass"


def run_hallucination_guard(
    citation_report: dict[str, Any],
    post_check_result: dict[str, Any],
    critical_fact_assessments: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """Run the full guard: score, decide, and optionally modify the answer.

    Returns a dict suitable for merging into AgentState.
    """
    score, flags = compute_hallucination_score(
        citation_report, post_check_result, critical_fact_assessments,
    )
    decision = decide_hallucination(score)

    result: dict[str, Any] = {
        "hallucination_score": round(score, 4),
        "hallucination_decision": decision,
        "hallucination_flags": flags,
    }

    if decision == "block":
        result["answer"] = (
            "**Insufficient evidence to provide a reliable answer.**\n\n"
            "The system detected a high risk of hallucination based on citation "
            "validation, unsupported claims, and critical fact assessments. "
            "Please refine your query or consult the source documents directly."
        )
    elif decision == "warn":
        warning_section = (
            "> **Hallucination Warning** (risk score: "
            f"{score:.2f})\n>\n"
        )
        for f in flags:
            warning_section += f"> - {f['detail']}\n"
        warning_section += "\n"
        result["answer"] = warning_section + answer

    logger.info(
        "Hallucination guard — score=%.4f decision=%s flags=%d",
        score, decision, len(flags),
    )
    return result
