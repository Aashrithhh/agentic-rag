"""Validator Node — cross-verify document claims against an external API.

Calls /api/validate-all/{table} with pagination (top, skip) and ordering
to fact-check specific data-points. Uses parallel async httpx calls.

Cache strategy (Option A): only the claim *extraction* step is cached
(the LLM call).  API verification is always live because external data
can change between requests.

Critical Fact Protection (gated by ``critical_fact_protection`` flag):
  For numeric/date claims, require at least N supporting spans in retrieved
  documents. If support is weak, flag as "insufficient evidence" rather
  than returning a confident answer.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal

import httpx
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.cache import cache, key_for
from app.config import settings
from app.state import AgentState, ValidationResult

logger = logging.getLogger(__name__)

# ── Regex for identifying critical facts (numeric values, dates, monetary) ──
_CRITICAL_VALUE_RE = re.compile(
    r'(?:'
    r'\$[\d,]+(?:\.\d+)?'                          # monetary ($1,234.56)
    r'|\d{1,3}(?:,\d{3})+(?:\.\d+)?'               # large numbers (1,234,567)
    r'|\d+(?:\.\d+)?%'                              # percentages (12.5%)
    r'|\d{4}[-/]\d{1,2}[-/]\d{1,2}'                 # dates (2024-01-15)
    r'|\d{1,2}[-/]\d{1,2}[-/]\d{4}'                 # dates (01/15/2024)
    r'|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s*\d{2,4}'
    r'|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{2,4}'
    r')',
    re.IGNORECASE,
)


class ExtractedClaim(BaseModel):
    """A single verifiable claim extracted from the retrieved documents."""
    claim: str = Field(description="The factual assertion to verify.")
    table: str = Field(description="The API table/resource to query.")
    filter_field: str = Field(description="The column to filter on.")
    filter_value: str = Field(description="The value to filter for.")
    order_by: str = Field(default="id", description="Column to order results by.")


class ClaimList(BaseModel):
    """List of claims to verify against the external API."""
    claims: list[ExtractedClaim] = Field(default_factory=list)


_SYSTEM = """\
You are a compliance fact-extraction engine.
Given a user query and retrieved document excerpts, identify discrete,
verifiable claims checkable via GET /api/validate-all/{{table}}?$top=N&$skip=M&$orderby=COL.
Available tables: transactions, entities, filings, audit_findings, controls.
Return a list of ExtractedClaim objects. If none are verifiable, return an empty list.\
"""

_HUMAN = """\
<query>{query}</query>
<documents>{documents}</documents>
Extract verifiable claims.\
"""

_prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


async def _verify_claim(
    client: httpx.AsyncClient, claim: ExtractedClaim, *, top: int = 10, skip: int = 0,
) -> ValidationResult:
    url = f"{settings.validation_api_base}/api/validate-all/{claim.table}"
    params = {
        "$top": top, "$skip": skip, "$orderby": claim.order_by,
        "$filter": f"{claim.filter_field} eq '{claim.filter_value}'",
    }
    try:
        resp = await client.get(url, params=params, timeout=15.0)
        data = resp.json() if resp.status_code == 200 else {}
        is_valid = bool(data.get("value")) if isinstance(data, dict) else bool(data)
        return ValidationResult(
            claim=claim.claim, endpoint=f"{url}?{httpx.QueryParams(params)}",
            status_code=resp.status_code, api_response=data if isinstance(data, dict) else {"raw": data},
            is_valid=is_valid,
        )
    except httpx.HTTPError as exc:
        logger.error("Validation HTTP error for claim '%s': %s", claim.claim, exc)
        return ValidationResult(
            claim=claim.claim, endpoint=url, status_code=0,
            api_response={"error": str(exc)}, is_valid=False,
        )


async def _verify_all(claims: list[ExtractedClaim]) -> list[ValidationResult]:
    async with httpx.AsyncClient() as client:
        return await asyncio.gather(*[_verify_claim(client, c) for c in claims])


def _aggregate_status(results: list[ValidationResult]) -> Literal["pass", "partial", "fail"]:
    if not results:
        return "pass"
    valid_count = sum(1 for r in results if r["is_valid"])
    if valid_count == len(results):
        return "pass"
    return "partial" if valid_count > 0 else "fail"


def validator_node(state: AgentState) -> dict:
    """LangGraph node: extract claims and verify them against the external API."""
    import time as _time
    from app.metrics import metrics
    metrics.inc("node_invocations.validator")
    _t0 = _time.perf_counter()
    try:
        return _validator_node_inner(state)
    finally:
        metrics.observe("node_latency.validator", _time.perf_counter() - _t0)


def _validator_node_inner(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    case_id = state.get("case_id", "")

    if not docs:
        return {"validation_results": [], "validation_status": "pass"}

    doc_text = "\n---\n".join(f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)

    # ── Cache check for claim EXTRACTION only (Option A) ──
    claim_list: ClaimList | None = None
    if settings.cache_llm_enabled:
        doc_snippets = [
            {"source": d.metadata.get("source", "?"),
             "content_hash": d.page_content[:200]}
            for d in docs
        ]
        ck = key_for(
            "llm",
            {"node": "validator_extract", "case_id": case_id,
             "query": query, "docs": doc_snippets,
             "model": settings.openai_model,
             "validation_api_base": settings.validation_api_base},
            prefix=f"validator:{case_id}",
        )
        hit, cached_claims = cache.get("llm", ck)
        if hit:
            logger.info("Validator extraction cache HIT — %d claims", len(cached_claims))
            claim_list = ClaimList(claims=[ExtractedClaim(**c) for c in cached_claims])

    if claim_list is None:
        llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0, max_tokens=2048,
        )
        structured_llm = llm.with_structured_output(ClaimList)
        chain = _prompt | structured_llm
        from app.resilience import invoke_with_retry
        claim_list = invoke_with_retry(
            chain, {"query": query, "documents": doc_text},
            node_name="validator", fallback=ClaimList(claims=[]),
        )

        # ── Cache the extracted claims ─────────────────────
        if settings.cache_llm_enabled:
            cache.set("llm", ck, [c.model_dump() for c in claim_list.claims])
            logger.debug("Validator extraction cache MISS — stored %d claims", len(claim_list.claims))

    if not claim_list.claims:
        return {"validation_results": [], "validation_status": "pass",
                "critical_fact_assessments": []}

    # API verification is always live (not cached)
    logger.info("Validator — verifying %d claims in parallel...", len(claim_list.claims))
    results = asyncio.run(_verify_all(claim_list.claims))
    status = _aggregate_status(results)

    # ── Critical fact protection ───────────────────────────
    critical_assessments: list[dict] = []
    if settings.critical_fact_protection:
        critical_assessments = _check_critical_fact_support(
            claim_list.claims, docs,
        )
        weak_facts = [a for a in critical_assessments if not a["is_sufficiently_supported"]]
        if weak_facts:
            logger.warning(
                "Critical fact protection: %d/%d critical claims have insufficient evidence.",
                len(weak_facts), len(critical_assessments),
            )
            # Downgrade status if critical facts lack evidence
            if status == "pass" and weak_facts:
                status = "partial"

    return {
        "validation_results": results,
        "validation_status": status,
        "critical_fact_assessments": critical_assessments,
    }


# ── Critical Fact Protection ─────────────────────────────────────────


def _check_critical_fact_support(
    claims: list[ExtractedClaim],
    docs: list[Document],
) -> list[dict]:
    """Verify that numeric/date claims have at least N supporting spans.

    For each claim containing critical values (numbers, dates, monetary
    amounts), count how many distinct retrieved documents contain matching
    values. If the count is below ``settings.critical_fact_min_spans``,
    mark the claim as insufficiently supported.

    Returns a list of assessments (only for claims with critical values).
    """
    assessments: list[dict] = []

    for claim in claims:
        # Extract critical values from the claim text
        critical_values = _CRITICAL_VALUE_RE.findall(claim.claim)
        if not critical_values:
            continue

        # Normalize values for comparison
        normalized_values = [_normalize_value(v) for v in critical_values]

        # Count distinct supporting documents
        supporting_sources: list[str] = []
        support_count = 0

        for doc in docs:
            content_lower = doc.page_content.lower()
            content_normalized = re.sub(r'[,\s]+', '', content_lower)
            source = doc.metadata.get("source", "unknown")

            matched = False
            for nv in normalized_values:
                # Check both normalized and raw matching
                if nv in content_normalized or nv in content_lower:
                    matched = True
                    break
                # Also check the raw value
                raw_lower = critical_values[normalized_values.index(nv)].lower()
                if raw_lower in content_lower:
                    matched = True
                    break

            if matched:
                support_count += 1
                supporting_sources.append(source)

        is_supported = support_count >= settings.critical_fact_min_spans
        assessments.append({
            "claim": claim.claim,
            "critical_values": critical_values,
            "support_count": support_count,
            "min_required": settings.critical_fact_min_spans,
            "is_sufficiently_supported": is_supported,
            "supporting_sources": supporting_sources,
        })

    return assessments


def _normalize_value(value: str) -> str:
    """Normalize a critical value for fuzzy matching.

    Strips currency symbols, commas, whitespace to allow matching
    "$1,234.56" against "1234.56" in document text.
    """
    v = value.lower().strip()
    v = re.sub(r'[\$,\s]', '', v)
    return v
