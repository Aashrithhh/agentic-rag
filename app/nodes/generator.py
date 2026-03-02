"""Generator Node — final answer synthesis with inline citations."""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from app.cache import cache, key_for, stable_hash
from app.config import settings
from app.state import AgentState

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a senior compliance analyst producing audit-ready answers.
RULES:
1. Ground every statement in the provided DOCUMENTS with [Source: filename.pdf, p.12] citations.
2. If VALIDATION RESULTS show a claim is invalid, flag the discrepancy for human review.
3. If validation status is "partial" or "fail", add a DISCREPANCY section.
4. Structure your answer with clear headings.
5. Think step-by-step through the compliance implications.
6. If documents are insufficient, state what is missing.\
"""

_HUMAN = """\
<query>{query}</query>
<documents>{documents}</documents>
<validation_results>{validation_results}</validation_results>
<validation_status>{validation_status}</validation_status>
Produce a comprehensive, cited answer.\
"""

_prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def _format_validation_results(results: list[dict]) -> str:
    if not results:
        return "No claims were sent for external validation."
    lines = []
    for r in results:
        icon = "✅" if r.get("is_valid") else "❌"
        lines.append(f"{icon} Claim: {r['claim']}\n   Endpoint: {r['endpoint']}\n   Valid: {r['is_valid']}")
    return "\n".join(lines)


def generator_node(state: AgentState) -> dict:
    """LangGraph node: synthesise the final cited answer."""
    import time as _time
    from app.metrics import metrics
    metrics.inc("node_invocations.generator")
    _t0 = _time.perf_counter()
    try:
        return _generator_node_inner(state)
    finally:
        metrics.observe("node_latency.generator", _time.perf_counter() - _t0)


def _generator_node_inner(state: AgentState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    v_results = state.get("validation_results", [])
    v_status = state.get("validation_status", "pass")
    case_id = state.get("case_id", "")

    doc_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    # ── Cache check ────────────────────────────────────────
    if settings.cache_llm_enabled:
        ck = key_for(
            "llm",
            {"node": "generator", "case_id": case_id, "query": query,
             "docs_digest": stable_hash(doc_text),
             "validation_results": v_results,
             "validation_status": v_status,
             "model": settings.openai_model},
            prefix=f"generator:{case_id}",
        )
        hit, cached = cache.get("llm", ck)
        if hit:
            logger.info("Generator cache HIT — returning %d-char cached answer.", len(cached["answer"]))
            return cached

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0, max_tokens=4096,
    )

    chain = _prompt | llm
    from app.resilience import invoke_with_retry
    response = invoke_with_retry(
        chain,
        {"query": query, "documents": doc_text or "(No documents retrieved.)",
         "validation_results": _format_validation_results(v_results),
         "validation_status": v_status},
        node_name="generator",
    )

    output = {"answer": response.content}

    # ── Citation validation ────────────────────────────────
    if settings.enable_citation_validation:
        try:
            from app.citation_validator import enforce_grounding
            grounding_result = enforce_grounding(response.content, docs)
            output["answer"] = grounding_result["answer"]
            output["citation_report"] = grounding_result.get("citation_report", {})
            output["post_check_result"] = grounding_result.get("post_check_result", {})
        except Exception as exc:
            logger.warning("Citation validation failed (returning raw answer): %s", exc)

    # ── HITL flagging (low-confidence answers) ─────────────
    if settings.hitl_enabled:
        try:
            from app.hitl import hitl_queue
            # Estimate confidence from grader + validation status
            confidence = 0.85
            v_status_val = v_status
            if v_status_val == "fail":
                confidence = 0.3
            elif v_status_val == "partial":
                confidence = 0.55
            if state.get("quality_downgrade"):
                confidence -= 0.15

            if hitl_queue.should_flag(confidence):
                review = hitl_queue.flag_for_review(
                    case_id=case_id,
                    query=query,
                    answer=output["answer"],
                    sources=[d.metadata.get("source", "unknown") for d in docs],
                    confidence=confidence,
                )
                output["hitl_review_id"] = review.id
                logger.info("Answer flagged for HITL review: %s (confidence=%.2f)", review.id, confidence)
        except Exception as exc:
            logger.warning("HITL flagging failed: %s", exc)

    # ── Cache store ────────────────────────────────────────
    if settings.cache_llm_enabled:
        cache.set("llm", ck, output)
        logger.debug("Generator cache MISS — stored answer")

    logger.info("Generator — produced %d-char answer.", len(output["answer"]))
    return output
