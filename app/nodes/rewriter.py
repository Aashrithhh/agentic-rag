"""Query Rewriter Node — self-correction loop component."""

from __future__ import annotations

import logging

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    original_query = state["query"]
    previous_query = state.get("rewritten_query") or original_query
    grader_reasoning = state.get("grader_reasoning", "")
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    doc_summaries = "\n".join(
        f"- [{d.metadata.get('source', '?')}] {d.page_content[:200]}…"
        for d in docs[:5]
    )

    llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0.3,
        max_tokens=512,
    )

    chain = _prompt | llm
    response = chain.invoke({
        "original_query": original_query,
        "previous_query": previous_query,
        "grader_reasoning": grader_reasoning,
        "doc_summaries": doc_summaries or "(none)",
    })

    new_query = response.content.strip().strip('"')
    logger.info("Rewriter — loop %d | '%s' → '%s'", loop + 1, previous_query[:80], new_query[:80])
    return {"rewritten_query": new_query, "loop_count": loop + 1}
