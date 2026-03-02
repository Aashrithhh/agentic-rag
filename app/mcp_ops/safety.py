"""Read-only SQL safety checks for PostgreSQL MCP tools."""

from __future__ import annotations

import re

_BLOCKED_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "truncate",
    "create",
    "alter",
    "drop",
    "copy",
    "do",
    "call",
    "grant",
    "revoke",
    "vacuum",
    "reindex",
    "refresh",
}


class ReadOnlySqlError(ValueError):
    """Raised when SQL violates read-only policy."""


def _strip_comments(sql: str) -> str:
    no_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.S)
    no_line = re.sub(r"--.*?$", "", no_block, flags=re.M)
    return no_line.strip()


def validate_readonly_postgres_sql(sql: str) -> str:
    """Validate that SQL is read-only and safe to execute."""
    cleaned = _strip_comments(sql)
    if not cleaned:
        raise ReadOnlySqlError("SQL is empty after removing comments.")

    if ";" in cleaned:
        raise ReadOnlySqlError("Multiple statements are not allowed.")

    lowered = cleaned.lower().strip()
    if not (
        lowered.startswith("select ")
        or lowered.startswith("with ")
        or lowered.startswith("explain select ")
        or lowered.startswith("explain with ")
    ):
        raise ReadOnlySqlError("Only SELECT/WITH SELECT/EXPLAIN SELECT queries are allowed.")

    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            raise ReadOnlySqlError(f"Blocked keyword detected: {kw}")

    return cleaned


def apply_select_limit(sql: str, limit: int) -> str:
    """Wrap read queries with a defensive limit unless already EXPLAIN."""
    cleaned = sql.strip()
    if cleaned.lower().startswith("explain"):
        return cleaned

    return f"SELECT * FROM ({cleaned}) AS mcp_q LIMIT {max(1, min(limit, 1000))}"
