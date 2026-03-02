"""Query Rewriter Node — self-correction loop component."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.cache import cache, key_for
from app.config import settings
from app.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a search-query optimiser for a compliance document retrieval system.
The previous query did NOT return relevant documents. Rewrite the query so
that a hybrid (semantic + keyword) search engine is more likely to find the
correct passages. Strategies: add synonyms, expand abbreviations, broaden/narrow scope.
Return ONLY the rewritten query string — nothing else.\
"""

_HUMAN = """\
<original_query>{original_query}</original_query>
<previous_search_query>{previous_query}</previous_search_query>
<grader_feedback>{grader_reasoning}</grader_feedback>
<retrieved_doc_summaries>{doc_summaries}</retrieved_doc_summaries>
Produce an improved search query.\
"""

_prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def rewriter_node(state: AgentState) -> dict:
    """LangGraph node: rewrite the search query for the next retrieval attempt."""
    import time as _time
    from app.metrics import metrics
    metrics.inc("node_invocations.rewriter")
    _t0 = _time.perf_counter()
    try:
        return _rewriter_node_inner(state)
    finally:
        metrics.observe("node_latency.rewriter", _time.perf_counter() - _t0)


def _rewriter_node_inner(state: AgentState) -> dict:
    original_query = state["query"]
    previous_query = state.get("rewritten_query") or original_query
    grader_reasoning = state.get("grader_reasoning", "")
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)
    case_id = state.get("case_id", "")

    doc_summaries = "\n".join(
        f"- [{d.metadata.get('source', '?')}] {d.page_content[:200]}…"
        for d in docs[:5]
    )

    # ── Cache check ────────────────────────────────────────
    if settings.cache_llm_enabled:
        ck = key_for(
            "llm",
            {"node": "rewriter", "case_id": case_id,
             "original_query": original_query,
             "previous_query": previous_query,
             "grader_reasoning": grader_reasoning,
             "doc_summaries": doc_summaries or "(none)",
             "model": settings.openai_model},
            prefix=f"rewriter:{case_id}",
        )
        hit, cached = cache.get("llm", ck)
        if hit:
            logger.info("Rewriter cache HIT — returning '%s'", cached["rewritten_query"][:80])
            return {"rewritten_query": cached["rewritten_query"], "loop_count": loop + 1}

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0.3,
        max_tokens=512,
    )

    chain = _prompt | llm
    from app.resilience import invoke_with_retry
    response = invoke_with_retry(
        chain,
        {"original_query": original_query, "previous_query": previous_query,
         "grader_reasoning": grader_reasoning, "doc_summaries": doc_summaries or "(none)"},
        node_name="rewriter",
    )

    new_query = response.content.strip().strip('"')

    # ── Cache store (store only rewritten_query; loop_count is computed) ──
    if settings.cache_llm_enabled:
        cache.set("llm", ck, {"rewritten_query": new_query})
        logger.debug("Rewriter cache MISS — stored rewritten query")

    logger.info("Rewriter — loop %d | '%s' → '%s'", loop + 1, previous_query[:80], new_query[:80])
    return {"rewritten_query": new_query, "loop_count": loop + 1}
