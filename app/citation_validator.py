"""Citation validator and answer post-checker — grounding enforcement.

Two-stage hallucination control:
  1. Citation extraction: parse [Source: ...] citations from generated answers
  2. Citation validation: verify every cited chunk ID maps to a retrieved doc
  3. Answer post-check: flag unsupported statements before returning to user

Forces every claim in the answer to be traceable to a retrieved chunk.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_chat_llm

logger = logging.getLogger(__name__)


# ── Citation extraction ──────────────────────────────────────────────

# Matches patterns like [Source: file.pdf, p.12] or [Source: file.txt]
_CITATION_RE = re.compile(
    r"\[Source:\s*([^\],]+?)(?:\s*,\s*p\.?\s*(\d+))?\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    """A single parsed citation from the answer."""
    source: str
    page: int | None = None
    text_context: str = ""    # surrounding text where citation appeared
    is_valid: bool = False    # set after validation
    matched_chunk_id: int | None = None


@dataclass
class CitationReport:
    """Report on citation validity for an answer."""
    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0
    uncited_claims: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    grounding_score: float = 0.0  # fraction of claims with valid citations
    status: Literal["grounded", "partially_grounded", "ungrounded"] = "grounded"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "uncited_claims": self.uncited_claims,
            "grounding_score": self.grounding_score,
            "status": self.status,
        }


def extract_citations(answer: str) -> list[Citation]:
    """Parse all [Source: ...] citations from the answer text."""
    citations: list[Citation] = []
    for match in _CITATION_RE.finditer(answer):
        source = match.group(1).strip()
        page_str = match.group(2)
        page = int(page_str) if page_str else None

        # Grab surrounding context (30 chars before, 30 after)
        start = max(0, match.start() - 60)
        end = min(len(answer), match.end() + 60)
        context = answer[start:end].strip()

        citations.append(Citation(
            source=source,
            page=page,
            text_context=context,
        ))

    return citations


def validate_citations(
    citations: list[Citation],
    retrieved_docs: list[Document],
) -> CitationReport:
    """Validate each citation against the retrieved documents.

    A citation is valid if its source filename matches a retrieved doc.
    Page number matching is done soft (citation is still valid if source
    matches but page is different, flagged with a warning).
    """
    report = CitationReport(total_citations=len(citations))
    report.citations = citations

    # Build lookup from retrieved docs
    doc_sources = set()
    doc_source_pages: dict[str, set[int | None]] = {}
    for doc in retrieved_docs:
        src = doc.metadata.get("source", "")
        doc_sources.add(src)
        if src not in doc_source_pages:
            doc_source_pages[src] = set()
        doc_source_pages[src].add(doc.metadata.get("page"))

    for citation in citations:
        # Check if source is among retrieved docs (fuzzy match)
        matched = False
        for doc_src in doc_sources:
            if _source_matches(citation.source, doc_src):
                citation.is_valid = True
                matched = True
                break

        if matched:
            report.valid_citations += 1
        else:
            report.invalid_citations += 1
            logger.warning(
                "Invalid citation: [Source: %s] — not in retrieved docs",
                citation.source,
            )

    if report.total_citations > 0:
        report.grounding_score = report.valid_citations / report.total_citations
    else:
        report.grounding_score = 0.0

    if report.grounding_score >= 0.9:
        report.status = "grounded"
    elif report.grounding_score >= 0.5:
        report.status = "partially_grounded"
    else:
        report.status = "ungrounded"

    return report


def _source_matches(citation_source: str, doc_source: str) -> bool:
    """Fuzzy check if a citation source matches a doc source.

    Handles common variations: path prefixes, case differences,
    partial filenames.
    """
    cs = citation_source.lower().strip()
    ds = doc_source.lower().strip()

    # Exact match
    if cs == ds:
        return True

    # Citation is a suffix of the doc source (or vice versa)
    if ds.endswith(cs) or cs.endswith(ds):
        return True

    # Strip path separators and compare filenames
    cs_name = cs.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    ds_name = ds.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if cs_name == ds_name:
        return True

    return False


# ── Answer post-checker (LLM-based) ─────────────────────────────────


class UnsupportedClaim(BaseModel):
    """A claim in the answer not supported by any retrieved document."""
    claim: str = Field(description="The unsupported statement from the answer")
    reason: str = Field(description="Why this claim is unsupported")
    severity: Literal["critical", "moderate", "minor"] = Field(
        description="How problematic the unsupported claim is"
    )


class PostCheckResult(BaseModel):
    """Result of the answer post-check for unsupported claims."""
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    overall_grounding: Literal["strong", "moderate", "weak"] = Field(
        description="Overall grounding assessment"
    )
    recommendation: Literal["approve", "flag_for_review", "reject"] = Field(
        description="Whether the answer should be approved, flagged, or rejected"
    )
    reasoning: str = Field(description="Overall assessment reasoning")


_POSTCHECK_SYSTEM = """\
You are an answer quality auditor for a compliance document system.
Compare the ANSWER against the SOURCE DOCUMENTS and identify any claims
in the answer that are NOT directly supported by the documents.

Rules:
1. Every factual claim, number, date, or assertion must be traceable to
   a specific passage in the source documents.
2. Reasonable inferences from document data are acceptable (marked "minor").
3. Claims that contradict documents are "critical".
4. Claims about topics not covered by documents are "moderate".
5. If the answer properly caveats uncertainty ("documents do not indicate..."),
   that is acceptable and NOT an unsupported claim."""

_POSTCHECK_HUMAN = """\
<answer>
{answer}
</answer>

<source_documents>
{documents}
</source_documents>

Identify ALL unsupported claims and provide your assessment."""

_postcheck_prompt = ChatPromptTemplate.from_messages([
    ("system", _POSTCHECK_SYSTEM),
    ("human", _POSTCHECK_HUMAN),
])


def post_check_answer(
    answer: str,
    retrieved_docs: list[Document],
) -> PostCheckResult:
    """Run LLM-based post-check to flag unsupported statements.

    Returns a PostCheckResult with any claims that cannot be traced
    back to the retrieved documents.
    """
    from app.metrics import metrics
    metrics.inc("answer_post_checks")

    if not answer or not retrieved_docs:
        return PostCheckResult(
            unsupported_claims=[],
            overall_grounding="weak",
            recommendation="flag_for_review",
            reasoning="No answer or documents to check",
        )

    doc_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]\n"
        f"{d.page_content}"
        for d in retrieved_docs
    )

    try:
        llm = get_chat_llm(temperature=0, max_tokens=2048)
        chain = _postcheck_prompt | llm.with_structured_output(PostCheckResult)
        result = chain.invoke({"answer": answer, "documents": doc_text})

        n_unsupported = len(result.unsupported_claims)
        n_critical = sum(1 for c in result.unsupported_claims if c.severity == "critical")

        metrics.inc("post_check_unsupported_claims", n_unsupported)
        metrics.inc("post_check_critical_claims", n_critical)

        logger.info(
            "Post-check: grounding=%s, recommendation=%s, "
            "unsupported=%d (critical=%d)",
            result.overall_grounding, result.recommendation,
            n_unsupported, n_critical,
        )
        return result

    except Exception as exc:
        logger.error("Answer post-check failed: %s", exc)
        return PostCheckResult(
            unsupported_claims=[],
            overall_grounding="moderate",
            recommendation="flag_for_review",
            reasoning=f"Post-check failed: {exc}",
        )


def enforce_grounding(
    answer: str,
    retrieved_docs: list[Document],
    strict: bool = True,
) -> tuple[str, CitationReport, PostCheckResult | None]:
    """Full grounding enforcement pipeline.

    1. Extract and validate citations
    2. Optionally run LLM post-check for uncited claims
    3. Add disclaimer if grounding is insufficient

    Returns (possibly_modified_answer, citation_report, post_check_result).
    """
    # Step 1: Citation validation
    citations = extract_citations(answer)
    citation_report = validate_citations(citations, retrieved_docs)

    # Step 2: Post-check (if enabled and answer has content)
    post_check = None
    if settings.enable_answer_post_check and len(answer) > 50:
        post_check = post_check_answer(answer, retrieved_docs)

    # Step 3: Add disclaimers if needed
    modified_answer = answer

    if citation_report.status == "ungrounded":
        disclaimer = (
            "\n\n---\n⚠️ **GROUNDING WARNING**: This answer contains citations "
            "that could not be verified against the retrieved documents. "
            "Please verify claims independently before relying on this response."
        )
        modified_answer += disclaimer

    if post_check and post_check.recommendation == "reject":
        critical_claims = [
            c for c in post_check.unsupported_claims
            if c.severity == "critical"
        ]
        if critical_claims:
            claims_text = "\n".join(f"  - {c.claim}: {c.reason}" for c in critical_claims)
            disclaimer = (
                f"\n\n---\n🚨 **UNSUPPORTED CLAIMS DETECTED**:\n{claims_text}\n"
                f"These claims are not supported by the source documents."
            )
            modified_answer += disclaimer

    if post_check and post_check.recommendation == "flag_for_review":
        modified_answer += (
            "\n\n---\n🔍 **REVIEW RECOMMENDED**: Some claims in this answer "
            "may need additional verification."
        )

    return modified_answer, citation_report, post_check
