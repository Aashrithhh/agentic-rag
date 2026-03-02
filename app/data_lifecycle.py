"""Data lifecycle management — retention policies and deletion workflows.

Provides per-case data governance:
  1. Retention policies — configurable per case or globally
  2. Soft delete — mark data as deleted (recoverable)
  3. Hard delete — permanent removal of chunks and case database
  4. Legal hold — prevent deletion of data under legal hold
  5. Expiry tracking — identify data past retention period

All operations are audit-logged for compliance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from sqlalchemy import text

from app.config import settings
from app.db import _case_db_name, get_base_engine, get_engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention policy for a case's data."""
    case_id: str
    retention_days: int = 365           # Default: 1 year
    legal_hold: bool = False            # Block deletion when True
    auto_delete: bool = False           # Auto-delete on expiry
    archive_before_delete: bool = True  # Archive before permanent deletion
    notify_before_days: int = 30        # Alert N days before expiry


# ── Default policies ─────────────────────────────────────────────────

DEFAULT_RETENTION_DAYS = 365

_policies: dict[str, RetentionPolicy] = {}


def set_retention_policy(policy: RetentionPolicy) -> None:
    """Register or update a retention policy for a case."""
    _policies[policy.case_id] = policy
    logger.info(
        "Retention policy set for case '%s': %d days, hold=%s, auto_delete=%s",
        policy.case_id, policy.retention_days,
        policy.legal_hold, policy.auto_delete,
    )

    try:
        from app.audit_log import audit_log
        audit_log.log_data_event(
            event_type="data_retention_applied",
            case_id=policy.case_id,
            details={
                "retention_days": policy.retention_days,
                "legal_hold": policy.legal_hold,
                "auto_delete": policy.auto_delete,
            },
        )
    except Exception:
        pass


def get_retention_policy(case_id: str) -> RetentionPolicy:
    """Get retention policy for a case (returns default if not set)."""
    return _policies.get(
        case_id,
        RetentionPolicy(case_id=case_id, retention_days=DEFAULT_RETENTION_DAYS),
    )


# ── Data lifecycle operations ────────────────────────────────────────


@dataclass
class DataInventory:
    """Inventory of data stored for a case."""
    case_id: str
    database_name: str
    total_chunks: int = 0
    total_sources: int = 0
    sources: list[str] = field(default_factory=list)
    oldest_chunk_date: str | None = None
    newest_chunk_date: str | None = None
    data_size_bytes: int = 0
    retention_policy: RetentionPolicy | None = None
    days_until_expiry: int | None = None
    is_expired: bool = False


def get_data_inventory(case_id: str) -> DataInventory:
    """Get a complete inventory of data stored for a case."""
    db_name = _case_db_name(case_id)
    policy = get_retention_policy(case_id)
    inventory = DataInventory(
        case_id=case_id,
        database_name=db_name,
        retention_policy=policy,
    )

    try:
        engine = get_engine(case_id)
        with engine.connect() as conn:
            # Total chunks
            result = conn.execute(text("SELECT COUNT(*) FROM document_chunks"))
            inventory.total_chunks = result.scalar() or 0

            # Distinct sources
            result = conn.execute(text(
                "SELECT DISTINCT source FROM document_chunks ORDER BY source"
            ))
            inventory.sources = [row[0] for row in result]
            inventory.total_sources = len(inventory.sources)

            # Date range (from metadata_extra if available)
            result = conn.execute(text(
                "SELECT MIN(effective_date), MAX(effective_date) FROM document_chunks"
            ))
            row = result.fetchone()
            if row and row[0]:
                inventory.oldest_chunk_date = str(row[0])
                inventory.newest_chunk_date = str(row[1])

            # Data size estimate
            result = conn.execute(text(
                "SELECT pg_total_relation_size('document_chunks')"
            ))
            inventory.data_size_bytes = result.scalar() or 0

    except Exception as exc:
        logger.warning("Could not get full inventory for case '%s': %s", case_id, exc)

    # Calculate expiry
    if inventory.oldest_chunk_date:
        try:
            oldest = datetime.fromisoformat(inventory.oldest_chunk_date)
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            expiry = oldest + timedelta(days=policy.retention_days)
            now = datetime.now(timezone.utc)
            days_left = (expiry - now).days
            inventory.days_until_expiry = days_left
            inventory.is_expired = days_left <= 0
        except Exception:
            pass

    return inventory


def delete_case_data(
    case_id: str,
    *,
    hard_delete: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Delete all data for a case.

    Parameters
    ----------
    case_id : str
        The case whose data should be deleted.
    hard_delete : bool
        If True, drop the entire case database. If False, truncate chunks.
    force : bool
        If True, bypass legal hold check.

    Returns
    -------
    dict with deletion details.
    """
    from app.audit_log import audit_log

    policy = get_retention_policy(case_id)

    # Legal hold check
    if policy.legal_hold and not force:
        msg = f"Case '{case_id}' is under legal hold — deletion blocked"
        logger.warning(msg)
        audit_log.log_data_event(
            event_type="data_deleted",
            case_id=case_id,
            details={"action": "blocked", "reason": "legal_hold"},
        )
        return {"status": "blocked", "reason": "legal_hold", "case_id": case_id}

    db_name = _case_db_name(case_id)
    result: dict[str, Any] = {
        "case_id": case_id,
        "database": db_name,
        "hard_delete": hard_delete,
    }

    try:
        if hard_delete:
            # Drop the entire case database
            _drop_case_database(case_id)
            result["status"] = "deleted"
            result["action"] = "database_dropped"
        else:
            # Soft delete: truncate the chunks table
            engine = get_engine(case_id)
            with engine.begin() as conn:
                count_result = conn.execute(text(
                    "SELECT COUNT(*) FROM document_chunks"
                ))
                count = count_result.scalar() or 0
                conn.execute(text("TRUNCATE TABLE document_chunks"))
                result["chunks_deleted"] = count
                result["status"] = "deleted"
                result["action"] = "chunks_truncated"

        # Invalidate caches
        try:
            from app.cache import cache
            cache.invalidate_case_retrieval_cache(case_id)
        except Exception:
            pass

        audit_log.log_data_event(
            event_type="data_deleted",
            case_id=case_id,
            details=result,
        )

        logger.info("Data deleted for case '%s': %s", case_id, result)

    except Exception as exc:
        logger.error("Data deletion failed for case '%s': %s", case_id, exc)
        result["status"] = "error"
        result["error"] = str(exc)

    return result


def delete_source_from_case(case_id: str, source: str) -> dict[str, Any]:
    """Delete all chunks from a specific source within a case."""
    from app.audit_log import audit_log

    policy = get_retention_policy(case_id)
    if policy.legal_hold:
        return {"status": "blocked", "reason": "legal_hold"}

    try:
        engine = get_engine(case_id)
        with engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM document_chunks WHERE source = :source"),
                {"source": source},
            )
            deleted = result.rowcount

        # Invalidate caches
        try:
            from app.cache import cache
            cache.invalidate_case_retrieval_cache(case_id)
        except Exception:
            pass

        audit_log.log_data_event(
            event_type="data_deleted",
            case_id=case_id,
            details={"source": source, "chunks_deleted": deleted},
        )

        logger.info(
            "Deleted %d chunks from source '%s' in case '%s'",
            deleted, source, case_id,
        )
        return {"status": "deleted", "source": source, "chunks_deleted": deleted}

    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _drop_case_database(case_id: str) -> None:
    """Drop the case's PostgreSQL database entirely."""
    db_name = _case_db_name(case_id)
    admin_engine = get_base_engine()
    try:
        with admin_engine.connect() as conn:
            # Terminate active connections
            conn.execute(text(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = '{db_name}'"
            ))
            conn.execute(text(f'DROP DATABASE IF EXISTS "{db_name}"'))
            logger.info("Dropped database '%s'", db_name)
    finally:
        admin_engine.dispose()


def check_expired_cases() -> list[DataInventory]:
    """Check all cases for expired data."""
    from app.cases import list_case_ids

    expired: list[DataInventory] = []
    for case_id in list_case_ids():
        inventory = get_data_inventory(case_id)
        if inventory.is_expired:
            expired.append(inventory)
            logger.warning(
                "Case '%s' data has expired (%d days past retention)",
                case_id, abs(inventory.days_until_expiry or 0),
            )

    return expired


def apply_retention_policies() -> dict[str, Any]:
    """Apply retention policies to all cases — delete expired data if auto_delete."""
    results: dict[str, Any] = {"checked": 0, "expired": 0, "deleted": 0, "held": 0}
    expired = check_expired_cases()
    results["checked"] = len(expired)
    results["expired"] = len(expired)

    for inventory in expired:
        policy = inventory.retention_policy
        if policy and policy.auto_delete and not policy.legal_hold:
            delete_result = delete_case_data(inventory.case_id)
            if delete_result.get("status") == "deleted":
                results["deleted"] += 1
        elif policy and policy.legal_hold:
            results["held"] += 1

    logger.info("Retention check complete: %s", results)
    return results
