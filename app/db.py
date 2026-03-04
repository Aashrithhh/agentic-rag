"""Postgres + pgvector database layer — **per-case database isolation**.

Responsibilities
-----------------
1. Maintain a **separate PostgreSQL database** for each investigation case
   (e.g. ``audit_rag_big_thorium``, ``audit_rag_purview_exchange``).
2. Create the ``document_chunks`` table + HNSW / GIN indexes inside each DB.
3. Provide helpers for upserting chunks and running hybrid search.

The ``get_engine(case_id)`` function is the only way to obtain an engine.
It raises ``ValueError`` when *case_id* is missing, enforcing strict
isolation so one case can **never** accidentally access another case's data.
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
    # ── Email-specific metadata columns ─────────────────────────
    Column("message_id", String(512)),        # unique email Message-ID
    Column("thread_id", String(512)),         # email thread/conversation ID
    Column("email_subject", String(1024)),    # email subject line
    Column("email_from", String(512)),        # sender
    Column("email_to", String(1024)),         # recipients
    Column("email_date", String(128)),        # email date string
    Column("chunk_type", String(64)),         # heading, paragraph, email_header, etc.
    Column("attachment_names", String(2048)), # comma-separated attachment filenames
    Column("parent_message_id", String(512)), # for attachment chunks → links to parent
    Column("total_chunks", Integer),          # total chunks from parent document
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

# ── Index on message_id for neighbor stitching lookups ───────────────────
msg_id_index = Index(
    "ix_chunks_message_id",
    document_chunks.c.message_id,
)

# ── Index on thread_id for thread-based retrieval ────────────────────────
thread_id_index = Index(
    "ix_chunks_thread_id",
    document_chunks.c.thread_id,
)

# ── Index on parent_message_id for attachment linkage ────────────────────
parent_msg_index = Index(
    "ix_chunks_parent_message_id",
    document_chunks.c.parent_message_id,
)


# ── Per-case engine management ───────────────────────────────────────────
_engine_cache: dict[str, Engine] = {}


def _case_db_name(case_id: str) -> str:
    """Derive the PostgreSQL database name for a case.

    ``big-thorium``  →  ``audit_rag_big_thorium``
    """
    slug = case_id.replace("-", "_")
    return f"audit_rag_{slug}"


def _case_database_url(case_id: str) -> str:
    """Build the full SQLAlchemy URL for a case's dedicated database."""
    base = settings.database_url          # e.g. …/audit_rag
    last_slash = base.rfind("/")
    return base[: last_slash + 1] + _case_db_name(case_id)


def get_base_engine() -> Engine:
    """Return an engine for the *base* (admin) database.

    Used only for ``CREATE DATABASE`` and migration tasks — never for
    retrieval or ingestion.
    """
    return create_engine(
        settings.database_url,
        isolation_level="AUTOCOMMIT",
        pool_size=2,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def get_engine(case_id: str) -> Engine:
    """Return a **cached** SQLAlchemy engine for the given case.

    Raises ``ValueError`` if *case_id* is empty / None — this is the
    primary guardrail that prevents cross-case data leakage.
    """
    if not case_id:
        raise ValueError(
            "case_id is required — the system enforces per-case database "
            "isolation; a missing case_id would break that guarantee."
        )

    if case_id not in _engine_cache:
        url = _case_database_url(case_id)
        _engine_cache[case_id] = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        logger.debug("Created engine for case '%s' → %s", case_id, _case_db_name(case_id))

    return _engine_cache[case_id]


def ensure_database_exists(case_id: str) -> None:
    """Create the per-case PostgreSQL database if it doesn't already exist."""
    db_name = _case_db_name(case_id)
    admin_engine = get_base_engine()
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": db_name},
            ).scalar()
            if not exists:
                # Identifiers can't be parameterised; slug is safe (alphanumeric + _)
                conn.execute(text(f'CREATE DATABASE "{db_name}"'))
                logger.info("Created database '%s'.", db_name)
    finally:
        admin_engine.dispose()


def init_db(case_id: str) -> Engine:
    """Create the per-case database, pgvector extension, table, indexes,
    and tsvector trigger.

    Safe to call repeatedly — all operations are idempotent.
    """
    ensure_database_exists(case_id)
    engine = get_engine(case_id)

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
    logger.info("Database '%s' initialised (pgvector HNSW + GIN indexes ready).", _case_db_name(case_id))

    # ── Ensure email metadata columns exist (idempotent migration) ──
    _ensure_email_columns(engine)

    return engine


def _ensure_email_columns(engine: Engine) -> None:
    """Add email metadata columns to existing tables (safe to re-run)."""
    new_columns = [
        ("message_id", "VARCHAR(512)"),
        ("thread_id", "VARCHAR(512)"),
        ("email_subject", "VARCHAR(1024)"),
        ("email_from", "VARCHAR(512)"),
        ("email_to", "VARCHAR(1024)"),
        ("email_date", "VARCHAR(128)"),
        ("chunk_type", "VARCHAR(64)"),
        ("attachment_names", "VARCHAR(2048)"),
        ("parent_message_id", "VARCHAR(512)"),
        ("total_chunks", "INTEGER"),
    ]
    with engine.begin() as conn:
        for col_name, col_type in new_columns:
            try:
                conn.execute(text(
                    f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS "
                    f"{col_name} {col_type}"
                ))
            except Exception:
                pass  # Column already exists

        # Create indexes if they don't exist
        for idx_name, col_name in [
            ("ix_chunks_message_id", "message_id"),
            ("ix_chunks_thread_id", "thread_id"),
            ("ix_chunks_parent_message_id", "parent_message_id"),
        ]:
            try:
                conn.execute(text(
                    f"CREATE INDEX IF NOT EXISTS {idx_name} ON document_chunks ({col_name})"
                ))
            except Exception:
                pass


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
            # Email-specific filters use ILIKE for partial matching
            if col_name == "email_from":
                where_clauses.append(document_chunks.c.email_from.ilike(f"%{value}%"))
            elif col_name == "email_to":
                where_clauses.append(document_chunks.c.email_to.ilike(f"%{value}%"))
            elif col_name == "source_file":
                where_clauses.append(document_chunks.c.source.ilike(f"%{value}%"))
            elif col_name == "email_date_from":
                where_clauses.append(document_chunks.c.email_date >= value)
            elif col_name == "email_date_to":
                where_clauses.append(document_chunks.c.email_date <= value)
            else:
                col = document_chunks.c.get(col_name)
                if col is not None:
                    where_clauses.append(col == value)

    # Common columns for both queries
    _common_cols = [
        document_chunks.c.id,
        document_chunks.c.content,
        document_chunks.c.source,
        document_chunks.c.page,
        document_chunks.c.chunk_index,
        document_chunks.c.doc_type,
        document_chunks.c.entity_name,
        document_chunks.c.message_id,
        document_chunks.c.thread_id,
        document_chunks.c.email_subject,
        document_chunks.c.email_from,
        document_chunks.c.email_to,
        document_chunks.c.email_date,
        document_chunks.c.chunk_type,
        document_chunks.c.attachment_names,
        document_chunks.c.parent_message_id,
        document_chunks.c.total_chunks,
        document_chunks.c.metadata_extra,
    ]

    # 1) Semantic search via HNSW cosine distance
    sem_q = (
        select(
            *_common_cols,
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
            *_common_cols,
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


# ── Neighbor Chunk Fetching (context stitching) ─────────────────────────


def fetch_neighbor_chunks(
    engine: Engine,
    message_id: str,
    chunk_index: int,
    window: int = 1,
) -> list[dict[str, Any]]:
    """Fetch neighboring chunks (chunk_index ± window) for the same message_id.

    Used for context stitching — when a chunk is retrieved, its
    neighbors provide additional context from the same email/document.
    """
    if not message_id:
        return []

    neighbor_indices = list(range(
        max(0, chunk_index - window),
        chunk_index + window + 1,
    ))
    # Exclude the original chunk_index
    neighbor_indices = [i for i in neighbor_indices if i != chunk_index]

    if not neighbor_indices:
        return []

    q = (
        select(
            document_chunks.c.id,
            document_chunks.c.content,
            document_chunks.c.source,
            document_chunks.c.page,
            document_chunks.c.chunk_index,
            document_chunks.c.doc_type,
            document_chunks.c.entity_name,
            document_chunks.c.message_id,
            document_chunks.c.thread_id,
            document_chunks.c.chunk_type,
            document_chunks.c.metadata_extra,
        )
        .where(
            document_chunks.c.message_id == message_id,
            document_chunks.c.chunk_index.in_(neighbor_indices),
        )
        .order_by(document_chunks.c.chunk_index)
    )

    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()

    return [dict(row) for row in rows]


def fetch_attachment_chunks(
    engine: Engine,
    message_id: str,
    *,
    top_k: int = 2,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Fetch attachment chunks linked to a parent email via parent_message_id.

    If ``query_embedding`` is provided, returns the top-k most relevant
    attachment chunks by cosine similarity; otherwise returns the first
    top-k by chunk_index.
    """
    if not message_id:
        return []

    if query_embedding is not None:
        q = (
            select(
                document_chunks.c.id,
                document_chunks.c.content,
                document_chunks.c.source,
                document_chunks.c.page,
                document_chunks.c.chunk_index,
                document_chunks.c.doc_type,
                document_chunks.c.entity_name,
                document_chunks.c.message_id,
                document_chunks.c.parent_message_id,
                document_chunks.c.chunk_type,
                document_chunks.c.metadata_extra,
                (1 - document_chunks.c.embedding.cosine_distance(query_embedding)).label("sem_score"),
            )
            .where(document_chunks.c.parent_message_id == message_id)
            .order_by(document_chunks.c.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )
    else:
        q = (
            select(
                document_chunks.c.id,
                document_chunks.c.content,
                document_chunks.c.source,
                document_chunks.c.page,
                document_chunks.c.chunk_index,
                document_chunks.c.doc_type,
                document_chunks.c.entity_name,
                document_chunks.c.message_id,
                document_chunks.c.parent_message_id,
                document_chunks.c.chunk_type,
                document_chunks.c.metadata_extra,
            )
            .where(document_chunks.c.parent_message_id == message_id)
            .order_by(document_chunks.c.chunk_index)
            .limit(top_k)
        )

    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()

    return [dict(row) for row in rows]


def fetch_parent_email_chunks(
    engine: Engine,
    parent_message_id: str,
    *,
    top_k: int = 2,
) -> list[dict[str, Any]]:
    """Fetch parent email chunks for an attachment chunk (reverse linkage).

    When an attachment chunk is retrieved, this fetches the parent email
    chunks for additional context.
    """
    if not parent_message_id:
        return []

    q = (
        select(
            document_chunks.c.id,
            document_chunks.c.content,
            document_chunks.c.source,
            document_chunks.c.page,
            document_chunks.c.chunk_index,
            document_chunks.c.doc_type,
            document_chunks.c.entity_name,
            document_chunks.c.message_id,
            document_chunks.c.chunk_type,
            document_chunks.c.metadata_extra,
        )
        .where(document_chunks.c.message_id == parent_message_id)
        .order_by(document_chunks.c.chunk_index)
        .limit(top_k)
    )

    with engine.connect() as conn:
        rows = conn.execute(q).mappings().all()

    return [dict(row) for row in rows]
