"""MCP tools for Docker Postgres + pgvector diagnostics."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any

from sqlalchemy import text

from app.cases import metadata_filter_for_case
from app.config import settings
from app.db import _case_db_name, get_base_engine, get_engine, hybrid_search
from app.mcp_ops.safety import ReadOnlySqlError, apply_select_limit, validate_readonly_postgres_sql


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def _find_postgres_container_name() -> str:
    if settings.postgres_container_name:
        return settings.postgres_container_name

    raw = _run(["docker", "ps", "--format", "{{json .}}"])
    if not raw:
        raise RuntimeError("No running Docker containers found.")

    for line in raw.splitlines():
        info = json.loads(line)
        image = str(info.get("Image", "")).lower()
        name = str(info.get("Names", ""))
        if "postgres" in image:
            if settings.docker_compose_service_name:
                if settings.docker_compose_service_name.lower() in name.lower():
                    return name
            else:
                return name

    raise RuntimeError("No running Postgres container found. Set POSTGRES_CONTAINER_NAME.")


def _db_exists(case_id: str) -> bool:
    db_name = _case_db_name(case_id)
    engine = get_base_engine()
    try:
        with engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": db_name},
            ).scalar()
            return bool(exists)
    finally:
        engine.dispose()


def docker_postgres_status() -> dict[str, Any]:
    container = _find_postgres_container_name()
    inspect_raw = _run(["docker", "inspect", container])
    info = json.loads(inspect_raw)[0]
    state = info.get("State", {})
    ports = info.get("NetworkSettings", {}).get("Ports", {})
    port_pairs: list[str] = []
    for key, vals in ports.items():
        if vals:
            for v in vals:
                port_pairs.append(f"{key}->{v.get('HostPort')}")
        else:
            port_pairs.append(key)

    return {
        "container_name": container,
        "status": state.get("Status", "unknown"),
        "health": (state.get("Health") or {}).get("Status", "n/a"),
        "ports": ", ".join(port_pairs),
        "started_at": state.get("StartedAt", ""),
    }


def case_db_status(case_id: str) -> dict[str, Any]:
    db_name = _case_db_name(case_id)
    exists = _db_exists(case_id)
    connectable = False
    table_exists = False

    if exists:
        try:
            engine = get_engine(case_id)
            with engine.connect() as conn:
                connectable = True
                table_exists = bool(
                    conn.execute(
                        text(
                            """
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema='public'
                              AND table_name='document_chunks'
                            """
                        )
                    ).scalar()
                )
        except Exception:
            connectable = False

    return {
        "case_id": case_id,
        "database_name": db_name,
        "db_exists": exists,
        "connectable": connectable,
        "schema": "public",
        "document_chunks_exists": table_exists,
    }


def pgvector_status(case_id: str) -> dict[str, Any]:
    engine = get_engine(case_id)
    with engine.connect() as conn:
        ext = conn.execute(text("SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
        dims = conn.execute(text("SELECT vector_dims(embedding) FROM document_chunks LIMIT 1")).scalar()

    return {
        "case_id": case_id,
        "extension_enabled": bool(ext),
        "extension_version": str(ext or ""),
        "embedding_dims": int(dims) if dims is not None else None,
    }


def vector_index_status(case_id: str) -> dict[str, Any]:
    engine = get_engine(case_id)
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT i.indexname,
                       i.indexdef,
                       pg_relation_size(to_regclass(i.indexname)) AS size_bytes
                FROM pg_indexes i
                WHERE i.schemaname='public'
                  AND i.tablename='document_chunks'
                ORDER BY i.indexname
                """
            )
        ).mappings().all()

    indexes = []
    for row in rows:
        definition = row["indexdef"]
        idx_type = "unknown"
        if re.search(r"\bUSING hnsw\b", definition, flags=re.I):
            idx_type = "hnsw"
        elif re.search(r"\bUSING gin\b", definition, flags=re.I):
            idx_type = "gin"
        indexes.append(
            {
                "name": row["indexname"],
                "type": idx_type,
                "valid": True,
                "size_bytes": int(row["size_bytes"] or 0),
                "definition": definition,
            }
        )

    return {"case_id": case_id, "indexes": indexes}


def case_table_stats(case_id: str) -> dict[str, Any]:
    engine = get_engine(case_id)
    with engine.connect() as conn:
        stats = conn.execute(
            text(
                """
                SELECT COUNT(*) AS row_count,
                       COUNT(DISTINCT source) AS distinct_sources,
                       MAX(id) AS max_id,
                       SUM(CASE WHEN source IS NULL THEN 1 ELSE 0 END) AS null_source_rows,
                       SUM(CASE WHEN btrim(content) = '' THEN 1 ELSE 0 END) AS empty_content_rows
                FROM document_chunks
                """
            )
        ).mappings().one()

    return {
        "case_id": case_id,
        "row_count": int(stats["row_count"] or 0),
        "distinct_sources": int(stats["distinct_sources"] or 0),
        "max_id": int(stats["max_id"]) if stats["max_id"] is not None else None,
        "null_source_rows": int(stats["null_source_rows"] or 0),
        "empty_content_rows": int(stats["empty_content_rows"] or 0),
    }


def retrieve_preview(case_id: str, query: str, top_k: int = 6) -> dict[str, Any]:
    engine = get_engine(case_id)
    from app.nodes.retriever import _embed_query

    query_embedding = _embed_query(query)
    rows = hybrid_search(
        engine=engine,
        query_embedding=query_embedding,
        query_text=query,
        top_k=max(1, min(int(top_k), 20)),
        metadata_filter=metadata_filter_for_case(case_id),
    )
    docs = [
        {
            "id": r.get("id"),
            "source": r.get("source"),
            "page": r.get("page"),
            "rrf_score": float(r.get("rrf_score") or 0.0),
            "snippet": str(r.get("content", ""))[:300],
        }
        for r in rows
    ]
    return {"case_id": case_id, "query": query, "top_k": top_k, "results": docs}


def trace_query(case_id: str, query: str, max_steps: int = 20) -> dict[str, Any]:
    max_steps = max(1, min(int(max_steps), 50))
    from app.graph import compile_graph

    agent = compile_graph()
    initial_state = {"query": query, "loop_count": 0, "case_id": case_id}

    node_order: list[str] = []
    final_state: dict[str, Any] = {}

    for idx, step in enumerate(agent.stream(initial_state), start=1):
        node = list(step.keys())[0]
        node_order.append(node)
        final_state.update(step[node])
        if idx >= max_steps:
            break

    return {
        "case_id": case_id,
        "node_order": node_order,
        "loop_count": int(final_state.get("loop_count", 0)),
        "grader_score": final_state.get("grader_score"),
        "validation_status": final_state.get("validation_status"),
        "doc_count": len(final_state.get("retrieved_docs", []) or []),
    }


def execute_readonly_postgres_sql(case_id: str, sql: str, limit: int = 200) -> dict[str, Any]:
    try:
        cleaned = validate_readonly_postgres_sql(sql)
    except ReadOnlySqlError as exc:
        return {
            "case_id": case_id,
            "allowed": False,
            "error": str(exc),
        }

    final_sql = apply_select_limit(cleaned, limit)
    engine = get_engine(case_id)
    with engine.connect() as conn:
        rows = conn.execute(text(final_sql)).mappings().all()

    result_rows = [dict(r) for r in rows]
    return {
        "case_id": case_id,
        "allowed": True,
        "row_count": len(result_rows),
        "rows": result_rows,
    }


def register_tools(mcp) -> None:
    mcp.tool()(docker_postgres_status)
    mcp.tool()(case_db_status)
    mcp.tool()(pgvector_status)
    mcp.tool()(vector_index_status)
    mcp.tool()(case_table_stats)
    mcp.tool()(retrieve_preview)
    mcp.tool()(trace_query)
    mcp.tool()(execute_readonly_postgres_sql)
