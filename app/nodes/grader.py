"""Grader Node — binary relevance check with structured output."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.cache import cache, key_for
from app.config import settings
from app.llm import get_chat_llm
from app.state import AgentState

logger = logging.getLogger(__name__)


class GradeResult(BaseModel):
    """Binary relevance grade for retrieved documents."""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the retrieved documents contain relevant info; 'no' otherwise."
    )
    reasoning: str = Field(description="Brief explanation of the relevance verdict.")


_SYSTEM = """\
You are a compliance-document relevance grader.
Given a USER QUERY and RETRIEVED DOCUMENTS, decide whether the documents
contain enough information to meaningfully answer the query.
Be strict: tangentially related topics should get "no".
Think step-by-step before providing your grade.\
"""

_HUMAN = """\
<query>{query}</query>
<documents>{documents}</documents>
Grade the relevance of the documents to the query.\
"""

_prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def grader_node(state: AgentState) -> dict:
    """LangGraph node: grade retrieved-doc relevance."""
    import time as _time
    from app.metrics import metrics
    metrics.inc("node_invocations.grader")
    _t0 = _time.perf_counter()
    try:
        return _grader_node_inner(state)
    finally:
        metrics.observe("node_latency.grader", _time.perf_counter() - _t0)


def _grader_node_inner(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    case_id = state.get("case_id", "")

    if not docs:
        return {"grader_score": "no", "grader_reasoning": "No documents were retrieved."}

    doc_texts = "\n---\n".join(
        f"[Source: {d.metadata.get('source', '?')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    # ── Cache check (key: query + doc snippets + model) ────
    if settings.cache_llm_enabled:
        doc_snippets = [
            {"source": d.metadata.get("source", "?"),
             "page": d.metadata.get("page"),
             "content_hash": d.page_content[:200]}
            for d in docs
        ]
        ck = key_for(
            "llm",
            {"node": "grader", "case_id": case_id, "query": query,
             "docs": doc_snippets, "model": settings.active_llm_model},
            prefix=f"grader:{case_id}",
        )
        hit, cached = cache.get("llm", ck)
        if hit:
            logger.info("Grader cache HIT — score=%s", cached["grader_score"])
            return cached

    llm = get_chat_llm(temperature=0, max_tokens=1024)
    structured_llm = llm.with_structured_output(GradeResult)
    chain = _prompt | structured_llm
    from app.resilience import invoke_with_retry
    result: GradeResult = invoke_with_retry(
        chain, {"query": query, "documents": doc_texts}, node_name="grader",
    )

    output = {"grader_score": result.score, "grader_reasoning": result.reasoning}

    # ── Cache store ────────────────────────────────────────
    if settings.cache_llm_enabled:
        cache.set("llm", ck, output)
        logger.debug("Grader cache MISS — stored result")

    logger.info("Grader — score=%s  reasoning=%s", result.score, result.reasoning[:120])
    return output
