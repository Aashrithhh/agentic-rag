"""Postgres + pgvector database layer.

Responsibilities
-----------------
1. Create the `document_chunks` table with an HNSW index on the embedding column.
2. Provide helpers for upserting chunks and running hybrid (semantic + keyword) search.

HNSW is preferred over IVFFlat for this use-case because:
  * No training step — works correctly from the first row.
  * Better recall / latency trade-off at production scale.
"""

from __future__ import annotations

import logging
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    literal_column,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.engine import Engine

from app.config import settings

logger = logging.getLogger(__name__)

metadata_obj = MetaData()

document_chunks = Table(
    "document_chunks",
    metadata_obj,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(settings.embedding_dim), nullable=False),
    Column("source", String(512), nullable=False),
    Column("page", Integer),
    Column("chunk_index", Integer),
    Column("doc_type", String(64)),           # e.g. "10-K", "contract", "regulation"
    Column("entity_name", String(256)),       # company / counterparty name
    Column("effective_date", DateTime),
    Column("metadata_extra", JSONB, server_default=text("'{}'")),
    Column(
        "ts_content",                         # tsvector for BM25-style keyword search
        TSVECTOR,
        nullable=True,
    ),
)

# ── HNSW index for cosine similarity (high-recall, no training step) ─────
hnsw_index = Index(
    "ix_chunks_embedding_hnsw",
    document_chunks.c.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 200},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

# ── GIN index on the tsvector column for full-text keyword search ────────
gin_index = Index(
    "ix_chunks_ts_content_gin",
    document_chunks.c.ts_content,
    postgresql_using="gin",
)


def get_engine() -> Engine:
    """Return a SQLAlchemy engine with connection pooling."""
    return create_engine(
        settings.database_url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
    )


def init_db(engine: Engine | None = None) -> Engine:
    """Create the pgvector extension, table, indexes, and tsvector trigger.

    Safe to call repeatedly — all operations are idempotent.
    """
    engine = engine or get_engine()

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))

    metadata_obj.create_all(engine, checkfirst=True)

    # Populate tsvector column via trigger so keyword search stays in sync.
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'document_chunks'
                          AND column_name = 'ts_content'
                          AND data_type = 'tsvector'
                    ) THEN
                        ALTER TABLE document_chunks
                            DROP COLUMN IF EXISTS ts_content,
                            ADD COLUMN ts_content tsvector
                                GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
                    END IF;
                END
                $$;
                """
            )
        )
    logger.info("Database initialised (pgvector HNSW + GIN indexes ready).")
    return engine


def upsert_chunks(
    engine: Engine,
    chunks: list[dict[str, Any]],
    *,
    batch_size: int = 256,
) -> int:
    """Insert chunks into *document_chunks*. Returns the number of rows inserted."""
    inserted = 0
    with engine.begin() as conn:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            conn.execute(document_chunks.insert(), batch)
            inserted += len(batch)
    logger.info("Inserted %d chunks.", inserted)
    return inserted


def hybrid_search(
    engine: Engine,
    query_embedding: list[float],
    query_text: str,
    *,
    top_k: int = settings.top_k,
    metadata_filter: dict[str, Any] | None = None,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Run Reciprocal Rank Fusion over semantic (HNSW) and keyword (GIN) results."""
    fetch_n = top_k * 3

    where_clauses: list[Any] = []
    if metadata_filter:
        for col_name, value in metadata_filter.items():
            col = document_chunks.c.get(col_name)
            if col is not None:
                where_clauses.append(col == value)

    # 1) Semantic search via HNSW cosine distance
    sem_q = (
        select(
            document_chunks.c.id,
            document_chunks.c.content,
            document_chunks.c.source,
            document_chunks.c.page,
            document_chunks.c.doc_type,
            document_chunks.c.entity_name,
            document_chunks.c.metadata_extra,
            (
                1 - document_chunks.c.embedding.cosine_distance(query_embedding)
            ).label("sem_score"),
        )
        .where(*where_clauses)
        .order_by(document_chunks.c.embedding.cosine_distance(query_embedding))
        .limit(fetch_n)
    )

    # 2) Keyword search via ts_content GIN index
    ts_query = func.websearch_to_tsquery("english", query_text)
    kw_q = (
        select(
            document_chunks.c.id,
            document_chunks.c.content,
            document_chunks.c.source,
            document_chunks.c.page,
            document_chunks.c.doc_type,
            document_chunks.c.entity_name,
            document_chunks.c.metadata_extra,
            func.ts_rank_cd(document_chunks.c.ts_content, ts_query).label(
                "kw_score"
            ),
        )
        .where(document_chunks.c.ts_content.op("@@")(ts_query), *where_clauses)
        .order_by(
            func.ts_rank_cd(document_chunks.c.ts_content, ts_query).desc()
        )
        .limit(fetch_n)
    )

    with engine.connect() as conn:
        sem_rows = conn.execute(sem_q).mappings().all()
        kw_rows = conn.execute(kw_q).mappings().all()

    # 3) Reciprocal Rank Fusion
    rrf_scores: dict[int, float] = {}
    doc_map: dict[int, dict] = {}

    for rank, row in enumerate(sem_rows, start=1):
        rid = row["id"]
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + semantic_weight / (rrf_k + rank)
        doc_map[rid] = dict(row)

    for rank, row in enumerate(kw_rows, start=1):
        rid = row["id"]
        rrf_scores[rid] = rrf_scores.get(rid, 0.0) + keyword_weight / (rrf_k + rank)
        if rid not in doc_map:
            doc_map[rid] = dict(row)

    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    return [{**doc_map[rid], "rrf_score": rrf_scores[rid]} for rid in sorted_ids]
