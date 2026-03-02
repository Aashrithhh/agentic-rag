"""LangGraph state definitions for the Self-Correction Audit Agent."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


class ValidationResult(TypedDict):
    """Single fact-check result returned by the Validator node."""
    claim: str
    endpoint: str
    status_code: int
    api_response: dict
    is_valid: bool


class AgentState(TypedDict, total=False):
    """Root state flowing through every LangGraph node."""
    query: str
    case_id: str                                     # selects isolated case context
    metadata_filter: dict[str, str]                   # auto-set from case_id
    rewritten_query: str
    retrieved_docs: list[Document]
    grader_score: Literal["yes", "no"]
    grader_reasoning: str
    validation_results: list[ValidationResult]
    validation_status: Literal["pass", "partial", "fail"]
    answer: str
    loop_count: int
    error: str | None
    thinking: str

    # ── Added by production-hardening modules ─────────────
    intent: str                                      # fact_lookup | summary | timeline | comparison | exploratory
    intent_confidence: float
    reranked: bool                                   # whether docs were reranked
    citation_report: dict                            # from citation_validator
    post_check_result: dict                          # from answer post-checker
    quality_downgrade: bool                          # set when fallback model used
    hitl_review_id: str                              # set when flagged for human review
