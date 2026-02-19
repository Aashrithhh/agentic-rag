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
    metadata_filter: dict
    thinking: str
