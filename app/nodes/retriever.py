"""Retriever Node — hybrid semantic + keyword search over pgvector.

Each case has its own isolated database.  The retriever resolves the
correct engine via ``get_engine(case_id)`` and fails closed when
``case_id`` is missing.

Enhanced features (gated by feature flags):
  - **Neighbor stitching**: auto-fetch chunk_index ± 1 from same message_id
  - **Attachment linking**: include attachment chunks for retrieved emails (and vice-versa)
  - **Metadata boosting**: boost chunks with exact entity/date/token matches
  - **Context merging**: deduplicate and merge stitched chunks before passing to generator
"""

from __future__ import annotations

import logging
import re

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

    # Merge user-provided email filters from state (UI/API)
    user_filters = state.get("metadata_filter")
    if user_filters and isinstance(user_filters, dict):
        # Case-set constraints (meta_filter) take precedence over user filters
        meta_filter = {**(user_filters or {}), **(meta_filter or {})}

    logger.info("Retriever — query: %s | case: %s | db: %s | filter: %s",
                query_text, case_id,
                f"audit_rag_{case_id.replace('-', '_')}",
                meta_filter)

    # ── Retrieval cache check (key includes case_id for isolation) ──
    if settings.cache_retrieval_enabled:
        rk = key_for(
            "retrieval",
            {"case_id": case_id, "query": query_text,
             "top_k": settings.top_k, "filter": meta_filter,
             "neighbor_stitching": settings.neighbor_stitching,
             "attachment_context": settings.attachment_context_link,
             "metadata_boost_enabled": settings.metadata_boost_enabled,
             "email_completion": True},
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
                    "chunk_index": row.get("chunk_index"),
                    "doc_type": row.get("doc_type"),
                    "entity_name": row.get("entity_name"),
                    "message_id": row.get("message_id", ""),
                    "thread_id": row.get("thread_id", ""),
                    "email_subject": row.get("email_subject", ""),
                    "email_from": row.get("email_from", ""),
                    "email_to": row.get("email_to", ""),
                    "email_date": row.get("email_date", ""),
                    "chunk_type": row.get("chunk_type", ""),
                    "attachment_names": row.get("attachment_names", ""),
                    "parent_message_id": row.get("parent_message_id", ""),
                    "total_chunks": row.get("total_chunks"),
                    "body_only": row.get("body_only", ""),
                    "rrf_score": row.get("rrf_score"),
                    **(row.get("metadata_extra") or {}),
                },
            )
        )

    # ── Context stitching: fetch neighbor chunks ─────────────────
    if settings.neighbor_stitching and docs:
        docs = _stitch_neighbor_chunks(engine, docs, query_embedding)

    # ── Attachment context linking ─────────────────────────────────
    if settings.attachment_context_link and docs:
        docs = _link_attachment_context(engine, docs, query_embedding)

    # ── Metadata boosting ─────────────────────────────────────────
    if settings.metadata_boost_enabled and docs:
        docs = _apply_metadata_boost(query_text, docs)

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

    # ── Email completion: fetch ALL sibling chunks AFTER reranking ──
    # This ensures the reranker doesn't discard sibling chunks needed
    # to assemble complete emails in the UI.
    docs = _complete_email_chunks(engine, docs)

    logger.info("Retriever — returned %d documents from '%s'.", len(docs), case_id)
    return {"retrieved_docs": docs, "reranked": reranked}


# ── Context Stitching ────────────────────────────────────────────────


def _stitch_neighbor_chunks(
    engine,
    docs: list[Document],
    query_embedding: list[float],
) -> list[Document]:
    """Fetch neighbor chunks (chunk_index ± window) for each retrieved doc.

    Merges neighbors into the result set, deduplicating by chunk ID.
    """
    from app.db import fetch_neighbor_chunks

    seen_ids: set[int] = {d.metadata.get("id") for d in docs if d.metadata.get("id")}
    new_docs: list[Document] = []

    for doc in docs:
        msg_id = doc.metadata.get("message_id", "")
        chunk_idx = doc.metadata.get("chunk_index")

        if not msg_id or chunk_idx is None:
            continue

        neighbors = fetch_neighbor_chunks(
            engine, msg_id, chunk_idx,
            window=settings.neighbor_stitch_window,
        )

        for nb in neighbors:
            if nb["id"] in seen_ids:
                continue
            seen_ids.add(nb["id"])
            new_docs.append(Document(
                page_content=nb["content"],
                metadata={
                    "id": nb["id"],
                    "source": nb["source"],
                    "page": nb.get("page"),
                    "chunk_index": nb.get("chunk_index"),
                    "doc_type": nb.get("doc_type"),
                    "entity_name": nb.get("entity_name"),
                    "message_id": nb.get("message_id", ""),
                    "thread_id": nb.get("thread_id", ""),
                    "chunk_type": nb.get("chunk_type", ""),
                    "is_stitched": True,
                    **(nb.get("metadata_extra") or {}),
                },
            ))

    if new_docs:
        logger.info("Context stitching: added %d neighbor chunks.", len(new_docs))

    # Merge and sort: original docs first, then stitched by chunk_index
    all_docs = docs + new_docs

    if settings.context_merge:
        all_docs = _merge_and_dedup(all_docs)

    return all_docs


def _merge_and_dedup(docs: list[Document]) -> list[Document]:
    """Deduplicate documents by ID and merge overlapping content."""
    seen: dict[int, Document] = {}
    for doc in docs:
        doc_id = doc.metadata.get("id")
        if doc_id and doc_id in seen:
            continue
        if doc_id:
            seen[doc_id] = doc
        else:
            seen[id(doc)] = doc
    return list(seen.values())


# ── Attachment Context Linking ────────────────────────────────────────


def _link_attachment_context(
    engine,
    docs: list[Document],
    query_embedding: list[float],
) -> list[Document]:
    """Link attachment chunks to parent emails and vice-versa.

    - If a parent email chunk is retrieved, include top attachment chunks.
    - If an attachment chunk is retrieved, include parent email chunks.
    """
    from app.db import fetch_attachment_chunks, fetch_parent_email_chunks

    seen_ids: set[int] = {d.metadata.get("id") for d in docs if d.metadata.get("id")}
    new_docs: list[Document] = []

    for doc in docs:
        msg_id = doc.metadata.get("message_id", "")
        parent_msg_id = doc.metadata.get("parent_message_id", "")

        # If this is a parent email, fetch its attachment chunks
        if msg_id and not parent_msg_id:
            att_chunks = fetch_attachment_chunks(
                engine, msg_id,
                top_k=settings.attachment_context_top_k,
                query_embedding=query_embedding,
            )
            for ac in att_chunks:
                if ac["id"] not in seen_ids:
                    seen_ids.add(ac["id"])
                    new_docs.append(Document(
                        page_content=ac["content"],
                        metadata={
                            "id": ac["id"],
                            "source": ac["source"],
                            "page": ac.get("page"),
                            "chunk_index": ac.get("chunk_index"),
                            "doc_type": ac.get("doc_type"),
                            "chunk_type": ac.get("chunk_type", ""),
                            "parent_message_id": ac.get("parent_message_id", ""),
                            "is_attachment_context": True,
                            **(ac.get("metadata_extra") or {}),
                        },
                    ))

        # If this is an attachment chunk, fetch parent email
        elif parent_msg_id:
            parent_chunks = fetch_parent_email_chunks(
                engine, parent_msg_id,
                top_k=settings.attachment_context_top_k,
            )
            for pc in parent_chunks:
                if pc["id"] not in seen_ids:
                    seen_ids.add(pc["id"])
                    new_docs.append(Document(
                        page_content=pc["content"],
                        metadata={
                            "id": pc["id"],
                            "source": pc["source"],
                            "page": pc.get("page"),
                            "chunk_index": pc.get("chunk_index"),
                            "doc_type": pc.get("doc_type"),
                            "chunk_type": pc.get("chunk_type", ""),
                            "message_id": pc.get("message_id", ""),
                            "is_parent_context": True,
                            **(pc.get("metadata_extra") or {}),
                        },
                    ))

    if new_docs:
        logger.info("Attachment context: linked %d additional chunks.", len(new_docs))

    return docs + new_docs


# ── Metadata Boosting ────────────────────────────────────────────────


def _apply_metadata_boost(query_text: str, docs: list[Document]) -> list[Document]:
    """Boost documents whose metadata matches query tokens.

    Boosts chunks with:
      - Exact entity/name matches in email_from, email_to, entity_name
      - Date/token matches from the query
      - Subject line keyword matches

    Penalizes stale/irrelevant chunks if enabled.
    """
    # Extract potential entities/terms from query
    query_lower = query_text.lower()
    query_tokens = set(re.findall(r'\b\w{3,}\b', query_lower))

    # Extract dates from query
    date_patterns = re.findall(
        r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s*\d{4}\b',
        query_lower,
    )

    for doc in docs:
        boost = 0.0
        meta = doc.metadata

        # Boost: sender/recipient name match
        for field in ("email_from", "email_to", "entity_name"):
            val = str(meta.get(field, "")).lower()
            if val:
                matching_tokens = query_tokens.intersection(set(re.findall(r'\b\w{3,}\b', val)))
                if matching_tokens:
                    boost += settings.metadata_boost_weight * len(matching_tokens)

        # Boost: subject line keyword match
        subject = str(meta.get("email_subject", "")).lower()
        if subject:
            subject_tokens = set(re.findall(r'\b\w{3,}\b', subject))
            matching = query_tokens.intersection(subject_tokens)
            if matching:
                boost += settings.metadata_boost_weight * 0.5 * len(matching)

        # Boost: date match
        email_date = str(meta.get("email_date", "")).lower()
        if email_date and date_patterns:
            for dp in date_patterns:
                if dp in email_date:
                    boost += settings.metadata_boost_weight

        # Penalty: staleness (optional)
        if settings.staleness_penalty_enabled:
            # Apply penalty for chunks without dates or with very old dates
            if not email_date and not meta.get("effective_date"):
                boost -= settings.staleness_penalty_weight

        # Store boost score for downstream reranking
        rrf = meta.get("rrf_score", 0) or 0
        doc.metadata["metadata_boost"] = boost
        doc.metadata["boosted_score"] = rrf + boost

    # Re-sort by boosted score
    docs.sort(key=lambda d: d.metadata.get("boosted_score", 0), reverse=True)

    boosted_count = sum(1 for d in docs if d.metadata.get("metadata_boost", 0) > 0)
    if boosted_count:
        logger.info("Metadata boost: boosted %d/%d documents.", boosted_count, len(docs))

    return docs


# ── Email Completion ─────────────────────────────────────────────────


def _complete_email_chunks(
    engine,
    docs: list[Document],
) -> list[Document]:
    """For each retrieved email chunk, fetch ALL sibling chunks from the same
    message_id so the full email can be assembled in the UI.

    Deduplicates by chunk ID to avoid repeating chunks already present.
    """
    from app.db import fetch_all_email_chunks

    seen_ids: set[int] = {d.metadata.get("id") for d in docs if d.metadata.get("id")}
    # Track which message_ids we've already completed
    completed_msg_ids: set[str] = set()
    new_docs: list[Document] = []

    for doc in docs:
        msg_id = doc.metadata.get("message_id", "")
        if not msg_id or msg_id in completed_msg_ids:
            continue
        completed_msg_ids.add(msg_id)

        siblings = fetch_all_email_chunks(engine, msg_id)

        for sib in siblings:
            if sib["id"] in seen_ids:
                continue
            seen_ids.add(sib["id"])
            new_docs.append(Document(
                page_content=sib["content"],
                metadata={
                    "id": sib["id"],
                    "source": sib["source"],
                    "page": sib.get("page"),
                    "chunk_index": sib.get("chunk_index"),
                    "doc_type": sib.get("doc_type"),
                    "entity_name": sib.get("entity_name"),
                    "message_id": sib.get("message_id", ""),
                    "thread_id": sib.get("thread_id", ""),
                    "chunk_type": sib.get("chunk_type", ""),
                    "email_subject": sib.get("email_subject", ""),
                    "email_from": sib.get("email_from", ""),
                    "email_to": sib.get("email_to", ""),
                    "email_date": sib.get("email_date", ""),
                    "body_only": sib.get("body_only", ""),
                    "total_chunks": sib.get("total_chunks"),
                    "is_sibling": True,
                    **(sib.get("metadata_extra") or {}),
                },
            ))

    if new_docs:
        logger.info("Email completion: added %d sibling chunks to assemble full emails.", len(new_docs))

    return docs + new_docs
