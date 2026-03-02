"""MCP resources for architecture and operational runbooks."""

from __future__ import annotations

import json

from app.cases import metadata_filter_for_case
from app.mcp_ops.tools import case_db_status, case_table_stats, pgvector_status, vector_index_status


def register_resources(mcp) -> None:
    @mcp.resource("rag://pipeline/architecture")
    def pipeline_architecture() -> str:
        return (
            "START -> Retriever -> Grader -> (Validator or Rewriter loop) -> Generator -> END\n"
            "Retriever uses hybrid pgvector(HNSW) + full text(GIN) inside case-isolated Postgres DBs."
        )

    @mcp.resource("rag://runbook/postgres-pgvector-health")
    def postgres_pgvector_health_runbook() -> str:
        return (
            "1) Verify Postgres Docker container is running.\n"
            "2) Verify case DB exists and is connectable.\n"
            "3) Verify extension 'vector' is installed in case DB.\n"
            "4) Verify HNSW and GIN indexes exist on document_chunks.\n"
            "5) Verify embedding dims match settings.embedding_dim.\n"
            "6) Run retrieve_preview and compare to expected documents."
        )

    @mcp.resource("rag://case/{case_id}/config")
    def case_config(case_id: str) -> str:
        payload = {
            "case_id": case_id,
            "metadata_filter": metadata_filter_for_case(case_id),
        }
        return json.dumps(payload, indent=2)

    @mcp.resource("rag://case/{case_id}/health-summary")
    def case_health_summary(case_id: str) -> str:
        payload = {
            "db": case_db_status(case_id),
            "pgvector": pgvector_status(case_id),
            "indexes": vector_index_status(case_id),
            "stats": case_table_stats(case_id),
        }
        return json.dumps(payload, indent=2, default=str)
