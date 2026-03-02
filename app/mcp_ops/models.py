"""Typed response models for MCP ops tools."""

from __future__ import annotations

from typing import Any, TypedDict


class DockerStatus(TypedDict):
    container_name: str
    status: str
    health: str
    ports: str
    started_at: str


class CaseDbStatus(TypedDict):
    case_id: str
    database_name: str
    db_exists: bool
    connectable: bool
    schema: str
    document_chunks_exists: bool


class PgvectorStatus(TypedDict):
    case_id: str
    extension_enabled: bool
    extension_version: str
    embedding_dims: int | None


class IndexStatus(TypedDict):
    case_id: str
    indexes: list[dict[str, Any]]


class CaseStats(TypedDict):
    case_id: str
    row_count: int
    distinct_sources: int
    max_id: int | None
    null_source_rows: int
    empty_content_rows: int
