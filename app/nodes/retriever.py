"""Retriever Node — hybrid semantic + keyword search over pgvector.

Each case has its own isolated database.  The retriever resolves the
correct engine via ``get_engine(case_id)`` and fails closed when
``case_id`` is missing.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document

from app.cache import cache, key_for
from app.config import settings
from app.db import get_engine, hybrid_search
from app.state import AgentState

logger = logging.getLogger(__name__)


def _embed_query(text: str) -> list[float]:
    """Produce an embedding vector using Cohere embed-english-v3.0.

    Results are cached in the ``embedding`` namespace keyed on
    ``{model, text}`` so identical queries never call Cohere twice.
    """
    # ── Cache check ────────────────────────────────────────
    if settings.cache_embeddings_enabled:
        ck = key_for("embedding", {"model": settings.cohere_embed_model, "text": text})
        hit, vec = cache.get("embedding", ck)
        if hit:
            logger.debug("Embedding cache HIT for query: %s", text[:60])
            return vec

    # ── Compute embedding ──────────────────────────────────
    try:
        from langchain_cohere import CohereEmbeddings
        _model = CohereEmbeddings(
            model=settings.cohere_embed_model if hasattr(settings, 'cohere_embed_model') else "embed-english-v3.0",
            cohere_api_key=settings.cohere_api_key if hasattr(settings, 'cohere_api_key') else "",
        )
        vec = _model.embed_query(text)
    except Exception:
        # Fallback: deterministic hash-based stub
        import hashlib, struct
        h = hashlib.sha512(text.encode()).digest()
        vec = []
        for i in range(settings.embedding_dim):
            byte_pair = h[(i * 2) % len(h) : (i * 2) % len(h) + 2] or b"\x00\x00"
            val = struct.unpack(">H", byte_pair.ljust(2, b"\x00"))[0] / 65535.0
            vec.append(val * 2 - 1)

    # ── Cache store ────────────────────────────────────────
    if settings.cache_embeddings_enabled:
        cache.set("embedding", ck, vec)
        logger.debug("Embedding cache MISS — stored for query: %s", text[:60])

    return vec


def retriever_node(state: AgentState) -> dict:
    """LangGraph node: retrieve documents via hybrid search.

    Requires ``case_id`` in state — fails with a clear error when it is
    missing to prevent accidental cross-case data access.
    """
    import time as _time
    from app.metrics import metrics
    metrics.inc("node_invocations.retriever")
    _t0 = _time.perf_counter()
    try:
        return _retriever_node_inner(state)
    finally:
        metrics.observe("node_latency.retriever", _time.perf_counter() - _t0)


def _retriever_node_inner(state: AgentState) -> dict:
    query_text = state.get("rewritten_query") or state["query"]

    # ── Strict case isolation: case_id is mandatory ─────────────
    case_id = state.get("case_id")
    if not case_id:
        raise ValueError(
            "Retriever: case_id is missing from state.  Every query must "
            "be scoped to a case for database-level isolation."
        )

    # Metadata filter is kept as a secondary guard inside the
    # per-case database (belt-and-suspenders).
    from app.cases import metadata_filter_for_case
    meta_filter = metadata_filter_for_case(case_id)

    logger.info("Retriever — query: %s | case: %s | db: %s | filter: %s",
                query_text, case_id,
                f"audit_rag_{case_id.replace('-', '_')}",
                meta_filter)

    # ── Retrieval cache check (key includes case_id for isolation) ──
    if settings.cache_retrieval_enabled:
        rk = key_for(
            "retrieval",
            {"case_id": case_id, "query": query_text,
             "top_k": settings.top_k, "filter": meta_filter},
            prefix=case_id,
        )
        hit, cached_docs = cache.get("retrieval", rk)
        if hit:
            logger.info("Retrieval cache HIT — returning %d cached docs for '%s'.",
                        len(cached_docs), case_id)
            return {"retrieved_docs": cached_docs}

    engine = get_engine(case_id)
    query_embedding = _embed_query(query_text)

    from app.resilience import db_invoke_with_retry
    raw_results = db_invoke_with_retry(
        hybrid_search,
        engine=engine,
        query_embedding=query_embedding,
        query_text=query_text,
        top_k=settings.top_k,
        metadata_filter=meta_filter,
        node_name="retriever",
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

    # ── Store in retrieval cache ───────────────────────────
    if settings.cache_retrieval_enabled:
        cache.set("retrieval", rk, docs)
        logger.info("Retrieval cache MISS — stored %d docs for '%s'.", len(docs), case_id)

    # ── Reranking ─────────────────────────────────────────
    reranked = False
    if settings.rerank_enabled and docs:
        try:
            from app.reranker import rerank_documents
            docs = rerank_documents(query_text, docs)
            reranked = True
            logger.info("Reranker — kept top %d docs after reranking.", len(docs))
        except Exception as exc:
            logger.warning("Reranking failed (using original order): %s", exc)

    logger.info("Retriever — returned %d documents from '%s'.", len(docs), case_id)
    return {"retrieved_docs": docs, "reranked": reranked}
