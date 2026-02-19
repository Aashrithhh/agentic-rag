"""Grader Node — binary relevance check with structured output."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
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
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"grader_score": "no", "grader_reasoning": "No documents were retrieved."}

    doc_texts = "\n---\n".join(
        f"[Source: {d.metadata.get('source', '?')} | Page: {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

    llm = AzureChatOpenAI(
        azure_deployment=settings.azure_openai_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=0,
        max_tokens=1024,
    )
    structured_llm = llm.with_structured_output(GradeResult)
    chain = _prompt | structured_llm
    result: GradeResult = chain.invoke({"query": query, "documents": doc_texts})

    logger.info("Grader — score=%s  reasoning=%s", result.score, result.reasoning[:120])
    return {"grader_score": result.score, "grader_reasoning": result.reasoning}
