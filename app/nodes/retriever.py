"""Retriever Node — hybrid semantic + keyword search over pgvector."""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.config import settings
from app.db import get_engine, hybrid_search
from app.state import AgentState

logger = logging.getLogger(__name__)

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine


def _embed_query(text: str) -> list[float]:
    """Produce an embedding vector using Cohere embed-english-v3.0."""
    try:
        from langchain_cohere import CohereEmbeddings
        _model = CohereEmbeddings(
            model=settings.cohere_embed_model if hasattr(settings, 'cohere_embed_model') else "embed-english-v3.0",
            cohere_api_key=settings.cohere_api_key if hasattr(settings, 'cohere_api_key') else "",
        )
        return _model.embed_query(text)
    except Exception:
        # Fallback: deterministic hash-based stub
        import hashlib, struct
        h = hashlib.sha512(text.encode()).digest()
        vec = []
        for i in range(settings.embedding_dim):
            byte_pair = h[(i * 2) % len(h) : (i * 2) % len(h) + 2] or b"\x00\x00"
            val = struct.unpack(">H", byte_pair.ljust(2, b"\x00"))[0] / 65535.0
            vec.append(val * 2 - 1)
        return vec


def retriever_node(state: AgentState) -> dict:
    """LangGraph node: retrieve documents via hybrid search."""
    query_text = state.get("rewritten_query") or state["query"]

    # ── Case-level isolation: case_id takes priority ────────────
    case_id = state.get("case_id")
    if case_id:
        from app.cases import metadata_filter_for_case
        meta_filter = metadata_filter_for_case(case_id)
    else:
        meta_filter = state.get("metadata_filter")

    logger.info("Retriever — query: %s | case: %s | filter: %s", query_text, case_id, meta_filter)

    query_embedding = _embed_query(query_text)

    raw_results = hybrid_search(
        engine=_get_engine(),
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=settings.top_k,
        metadata_filter=meta_filter,
    )

    docs: list[Document] = []
    for row in raw_results:
        docs.append(
            Document(
                page_content=row["content"],
                metadata={
                    "id": row["id"],
                    "source": row["source"],
                    "page": row.get("page"),
                    "doc_type": row.get("doc_type"),
                    "entity_name": row.get("entity_name"),
                    "rrf_score": row.get("rrf_score"),
                    **(row.get("metadata_extra") or {}),
                },
            )
        )

    logger.info("Retriever — returned %d documents.", len(docs))
    return {"retrieved_docs": docs}
