"""Query intent router — classify user queries and route to optimal strategy.

Intent categories:
  - ``fact_lookup``   — specific factual question (short, targeted retrieval)
  - ``summary``       — broad summarization request (retrieve more docs, longer answer)
  - ``timeline``      — chronological/event sequence (time-ordered retrieval)
  - ``comparison``    — compare entities/periods (multi-faceted retrieval)
  - ``exploratory``   — open-ended exploration (wide retrieval, structured answer)

The router adjusts retrieval parameters (top_k, search strategy) and
generator prompts based on detected intent.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.llm import get_chat_llm

logger = logging.getLogger(__name__)

QueryIntent = Literal["fact_lookup", "summary", "timeline", "comparison", "exploratory"]


class IntentClassification(BaseModel):
    """Structured intent classification result."""
    intent: QueryIntent = Field(
        description="The primary intent category of the query"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the classification (0.0–1.0)"
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was chosen"
    )
    suggested_top_k: int = Field(
        default=6,
        description="Recommended number of documents to retrieve"
    )
    suggested_search_emphasis: Literal["semantic", "keyword", "balanced"] = Field(
        default="balanced",
        description="Whether to emphasize semantic or keyword search"
    )


# ── Intent-specific retrieval parameters ─────────────────────────────

INTENT_PARAMS: dict[QueryIntent, dict[str, Any]] = {
    "fact_lookup": {
        "top_k": 4,
        "semantic_weight": 0.8,
        "keyword_weight": 0.2,
        "max_answer_tokens": 1024,
        "system_addendum": "Be concise and precise. Answer the specific question directly.",
    },
    "summary": {
        "top_k": 10,
        "semantic_weight": 0.6,
        "keyword_weight": 0.4,
        "max_answer_tokens": 4096,
        "system_addendum": "Provide a comprehensive summary covering all key aspects.",
    },
    "timeline": {
        "top_k": 8,
        "semantic_weight": 0.5,
        "keyword_weight": 0.5,
        "max_answer_tokens": 3072,
        "system_addendum": (
            "Organize your answer chronologically. Include dates and "
            "sequence of events where available."
        ),
    },
    "comparison": {
        "top_k": 8,
        "semantic_weight": 0.7,
        "keyword_weight": 0.3,
        "max_answer_tokens": 3072,
        "system_addendum": (
            "Structure your answer as a comparison. Use tables or "
            "side-by-side format where appropriate."
        ),
    },
    "exploratory": {
        "top_k": 8,
        "semantic_weight": 0.6,
        "keyword_weight": 0.4,
        "max_answer_tokens": 4096,
        "system_addendum": (
            "Provide a thorough exploration of the topic with clear "
            "structure and headings."
        ),
    },
}


# ── Classification prompt ────────────────────────────────────────────

_SYSTEM = """\
You are a query intent classifier for a compliance document retrieval system.
Classify the user's query into exactly one of these categories:

- fact_lookup: Specific factual question seeking a precise answer
  (e.g., "What was the revenue in Q4 2024?", "Who signed the contract?")

- summary: Request for a broad overview or summarization
  (e.g., "Summarize the safety audit findings", "What are the key compliance issues?")

- timeline: Request for chronological/sequential information
  (e.g., "What happened between Jan and March?", "Timeline of the investigation")

- comparison: Request to compare entities, periods, or metrics
  (e.g., "Compare Q3 vs Q4 performance", "Differences between Policy A and B")

- exploratory: Open-ended exploration or multi-faceted question
  (e.g., "What do we know about worker conditions?", "Tell me about Big Thorium")

Also suggest how many documents to retrieve (top_k) and whether to emphasize
semantic similarity or keyword matching."""

_HUMAN = """\
<query>{query}</query>

Classify the intent of this query."""

_prompt = ChatPromptTemplate.from_messages([("system", _SYSTEM), ("human", _HUMAN)])


def classify_intent(query: str) -> IntentClassification:
    """Classify the intent of a user query using LLM.

    Falls back to 'exploratory' with balanced settings on failure.
    """
    from app.metrics import metrics
    metrics.inc("intent_classifications")

    try:
        llm = get_chat_llm(temperature=0, max_tokens=512)
        chain = _prompt | llm.with_structured_output(IntentClassification)
        result = chain.invoke({"query": query})
        logger.info(
            "Intent classified: %s (confidence=%.2f) for query: %s",
            result.intent, result.confidence, query[:80],
        )
        metrics.inc(f"intent_type.{result.intent}")
        return result

    except Exception as exc:
        logger.warning("Intent classification failed: %s — defaulting to exploratory", exc)
        return IntentClassification(
            intent="exploratory",
            confidence=0.5,
            reasoning="Classification failed; using default",
            suggested_top_k=settings.top_k,
            suggested_search_emphasis="balanced",
        )


def classify_intent_fast(query: str) -> IntentClassification:
    """Heuristic-based fast intent classification (no LLM call).

    Uses keyword patterns for quick classification when LLM is
    unavailable or to save costs.
    """
    q = query.lower().strip()

    # Timeline indicators
    timeline_keywords = [
        "timeline", "chronolog", "sequence", "when did", "what happened",
        "history of", "events", "between.*and", "from.*to.*date",
    ]
    for kw in timeline_keywords:
        import re
        if re.search(kw, q):
            return IntentClassification(
                intent="timeline", confidence=0.7,
                reasoning=f"Matched timeline keyword: {kw}",
                suggested_top_k=8,
                suggested_search_emphasis="balanced",
            )

    # Comparison indicators
    comparison_keywords = [
        "compare", "difference", "versus", " vs ", "contrast",
        "how does.*differ", "similarities",
    ]
    for kw in comparison_keywords:
        import re
        if re.search(kw, q):
            return IntentClassification(
                intent="comparison", confidence=0.7,
                reasoning=f"Matched comparison keyword: {kw}",
                suggested_top_k=8,
                suggested_search_emphasis="semantic",
            )

    # Summary indicators
    summary_keywords = [
        "summarize", "summary", "overview", "key findings",
        "main points", "what are the", "describe",
    ]
    for kw in summary_keywords:
        if kw in q:
            return IntentClassification(
                intent="summary", confidence=0.7,
                reasoning=f"Matched summary keyword: {kw}",
                suggested_top_k=10,
                suggested_search_emphasis="balanced",
            )

    # Fact lookup indicators (short, specific questions)
    fact_patterns = [
        "how much", "how many", "what is the", "what was the",
        "who is", "who was", "who signed", "what date",
        "what amount", "what percentage",
    ]
    for kw in fact_patterns:
        if kw in q:
            return IntentClassification(
                intent="fact_lookup", confidence=0.7,
                reasoning=f"Matched fact lookup pattern: {kw}",
                suggested_top_k=4,
                suggested_search_emphasis="keyword",
            )

    # Default: exploratory
    return IntentClassification(
        intent="exploratory", confidence=0.5,
        reasoning="No strong pattern match; defaulting to exploratory",
        suggested_top_k=settings.top_k,
        suggested_search_emphasis="balanced",
    )


def get_intent_params(intent: QueryIntent) -> dict[str, Any]:
    """Get retrieval parameters for a classified intent."""
    return INTENT_PARAMS.get(intent, INTENT_PARAMS["exploratory"]).copy()


def route_query(query: str, use_llm: bool = True) -> tuple[IntentClassification, dict[str, Any]]:
    """Classify intent and return tuned parameters.

    Parameters
    ----------
    query : str
        The user's query text.
    use_llm : bool
        Whether to use LLM for classification (True) or fast heuristics (False).

    Returns
    -------
    tuple of (IntentClassification, params_dict)
    """
    if use_llm and settings.intent_routing_enabled:
        intent = classify_intent(query)
    else:
        intent = classify_intent_fast(query)

    params = get_intent_params(intent.intent)

    # Override top_k with LLM suggestion if confident
    if intent.confidence >= 0.8:
        params["top_k"] = intent.suggested_top_k

    return intent, params
