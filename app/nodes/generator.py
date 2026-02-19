"""Generator Node — final answer synthesis with inline citations."""

from __future__ import annotations

import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    v_results = state.get("validation_results", [])
    v_status = state.get("validation_status", "pass")

    doc_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0, max_tokens=4096,
    )

    chain = _prompt | llm
    response = chain.invoke({
        "query": query,
        "documents": doc_text or "(No documents retrieved.)",
        "validation_results": _format_validation_results(v_results),
        "validation_status": v_status,
    })

    logger.info("Generator — produced %d-char answer.", len(response.content))
    return {"answer": response.content}
