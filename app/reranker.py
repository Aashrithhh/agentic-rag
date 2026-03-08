"""Cross-encoder reranker — rerank retrieved documents after hybrid search.

Supports multiple reranking strategies:
  1. Cohere Rerank API (production — best quality)
  2. LLM-based reranking via GPT-4o (fallback)
  3. Local cross-encoder model via sentence-transformers (offline)

Plugs in between the retriever and grader nodes in the LangGraph pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


def rerank_with_cohere(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """Rerank documents using Cohere Rerank API (cross-encoder).

    Returns documents sorted by relevance score (highest first).
    The Cohere reranker uses a cross-encoder architecture that jointly
    processes query-document pairs for much better ranking than
    bi-encoder similarity.
    """
    if not documents:
        return []

    top_n = top_n or min(len(documents), settings.top_k)

    try:
        import cohere

        client = cohere.Client(api_key=settings.cohere_api_key)
        doc_texts = [d.page_content for d in documents]

        response = client.rerank(
            model=settings.rerank_model,
            query=query,
            documents=doc_texts,
            top_n=top_n,
        )

        reranked: list[Document] = []
        for result in response.results:
            idx = result.index
            doc = documents[idx]
            # Attach rerank score to metadata
            doc.metadata["rerank_score"] = result.relevance_score
            doc.metadata["original_rank"] = idx
            reranked.append(doc)

        logger.info(
            "Cohere rerank: %d → %d docs, top score=%.4f",
            len(documents), len(reranked),
            reranked[0].metadata["rerank_score"] if reranked else 0,
        )
        return reranked

    except Exception as exc:
        logger.warning("Cohere rerank failed: %s — falling back to LLM rerank", exc)
        return rerank_with_llm(query, documents, top_n)


def rerank_with_llm(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """Rerank documents using GPT-4o as a cross-encoder substitute.

    Asks the LLM to score each document's relevance to the query on a
    0–10 scale, then sorts by score descending.
    """
    if not documents:
        return []

    top_n = top_n or min(len(documents), settings.top_k)

    try:
        from app.llm import get_chat_llm
        from pydantic import BaseModel, Field

        class DocScore(BaseModel):
            """Relevance score for a single document."""
            index: int = Field(description="0-based index of the document")
            score: float = Field(ge=0, le=10, description="Relevance score 0-10")
            reasoning: str = Field(description="Brief reason for the score")

        class RerankResult(BaseModel):
            """Reranking result with scored documents."""
            scores: list[DocScore] = Field(default_factory=list)

        doc_texts = "\n\n".join(
            f"[DOC {i}] (source: {d.metadata.get('source', '?')})\n{d.page_content[:500]}"
            for i, d in enumerate(documents)
        )

        llm = get_chat_llm(temperature=0, max_tokens=2048)

        structured_llm = llm.with_structured_output(RerankResult)
        result = structured_llm.invoke(
            f"Score each document's relevance to the query on a 0-10 scale.\n\n"
            f"Query: {query}\n\n"
            f"Documents:\n{doc_texts}\n\n"
            f"Return scores for all {len(documents)} documents."
        )

        # Sort by score descending
        scored = sorted(result.scores, key=lambda s: s.score, reverse=True)
        reranked: list[Document] = []
        for s in scored[:top_n]:
            if s.index < len(documents):
                doc = documents[s.index]
                doc.metadata["rerank_score"] = s.score / 10.0  # Normalize to 0-1
                doc.metadata["rerank_reasoning"] = s.reasoning
                doc.metadata["original_rank"] = s.index
                reranked.append(doc)

        logger.info("LLM rerank: %d → %d docs", len(documents), len(reranked))
        return reranked

    except Exception as exc:
        logger.error("LLM rerank failed: %s — returning original order", exc)
        return documents[:top_n]


def rerank_with_cross_encoder(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> list[Document]:
    """Rerank using a local sentence-transformers cross-encoder model.

    Requires: pip install sentence-transformers
    """
    if not documents:
        return []

    top_n = top_n or min(len(documents), settings.top_k)

    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        pairs = [(query, d.page_content) for d in documents]
        scores = model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked: list[Document] = []
        for doc, score in scored_docs[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        logger.info(
            "Cross-encoder rerank: %d → %d docs, top score=%.4f",
            len(documents), len(reranked),
            reranked[0].metadata["rerank_score"] if reranked else 0,
        )
        return reranked

    except ImportError:
        logger.warning("sentence-transformers not installed — falling back to Cohere rerank")
        return rerank_with_cohere(query, documents, top_n)
    except Exception as exc:
        logger.error("Cross-encoder rerank failed: %s", exc)
        return documents[:top_n]


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
) -> list[Document]:
    """Primary reranking entry point — dispatches to configured strategy.

    Strategy is selected via ``settings.rerank_strategy``:
      - ``cohere``        — Cohere Rerank API (default)
      - ``llm``           — GPT-4o reranking
      - ``cross-encoder`` — Local cross-encoder model
      - ``none``          — No reranking (passthrough)
    """
    from app.metrics import metrics
    metrics.inc("rerank_invocations")

    if not settings.rerank_enabled:
        return documents

    strategy = settings.rerank_strategy

    if strategy == "cohere":
        return rerank_with_cohere(query, documents, top_n)
    elif strategy == "llm":
        return rerank_with_llm(query, documents, top_n)
    elif strategy == "cross-encoder":
        return rerank_with_cross_encoder(query, documents, top_n)
    elif strategy == "none":
        return documents
    else:
        logger.warning("Unknown rerank strategy '%s' — skipping rerank", strategy)
        return documents
