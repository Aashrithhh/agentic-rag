"""Structured audit logging for security events.

Captures and persists:
  - Authentication events (login success/failure, session expiry)
  - Query access logs (who queried what case, when)
  - Admin diagnostics usage (MCP tools, SQL execution)
  - Data lifecycle events (ingestion, deletion, retention)
  - Security events (API key usage, rate limiting, CORS violations)

Logs are written to both the application logger (for aggregation)
and a dedicated audit log file for compliance retention.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("app.audit")

AuditEventType = Literal[
    "auth_login_success",
    "auth_login_failure",
    "auth_session_expired",
    "query_submitted",
    "query_completed",
    "query_failed",
    "admin_mcp_tool_used",
    "admin_sql_executed",
    "data_ingested",
    "data_deleted",
    "data_retention_applied",
    "pii_detected",
    "pii_redacted",
    "api_key_used",
    "api_rate_limited",
    "security_cors_violation",
    "circuit_breaker_state_change",
    "hitl_review_submitted",
    "hitl_review_completed",
    "hitl_flagged",
    "secret_rotated",
    "config_changed",
    "feedback_submitted",
    "chat_saved",
]


@dataclass
class AuditEvent:
    """A single audit log entry."""
    event_type: AuditEventType
    timestamp: str = ""
    user: str = "system"
    ip_address: str = ""
    case_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: Literal["info", "warning", "critical"] = "info"
    request_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Thread-safe audit logger with file and structured output."""

    def __init__(self, log_dir: str = "logs/audit"):
        self._log_dir = Path(log_dir)
        self._lock = threading.Lock()
        self._file_handler: logging.FileHandler | None = None
        self._audit_logger = logging.getLogger("app.audit.events")
        self._events: list[AuditEvent] = []  # in-memory buffer for recent events
        self._max_buffer = 10000
        # Bounded LRU set for deduplication (event_key -> True)
        self._recent_event_keys: OrderedDict[str, bool] = OrderedDict()
        self._dedup_max = 5000
        self._setup_file_handler()

    def _setup_file_handler(self) -> None:
        """Set up a dedicated file handler for audit logs."""
        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            log_file = self._log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            handler = logging.FileHandler(str(log_file), encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._audit_logger.addHandler(handler)
            self._audit_logger.setLevel(logging.INFO)
            # Prevent propagation to root logger to avoid duplicate output
            self._audit_logger.propagate = False
        except Exception as exc:
            logger.warning("Could not set up audit file handler: %s", exc)

    def _compute_event_key(self, event: AuditEvent) -> str:
        """Compute a stable dedup key from event_type + session/case + timestamp."""
        raw = f"{event.event_type}|{event.case_id}|{json.dumps(event.details, sort_keys=True, default=str)}|{event.timestamp}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def _is_duplicate(self, event: AuditEvent) -> bool:
        """Return True if this exact event was already logged recently."""
        key = self._compute_event_key(event)
        if key in self._recent_event_keys:
            return True
        self._recent_event_keys[key] = True
        if len(self._recent_event_keys) > self._dedup_max:
            self._recent_event_keys.popitem(last=False)
        return False

    def _get_request_context(self) -> tuple[str, str]:
        """Get ip_address and request_id from the current request context."""
        ip_address = ""
        request_id = ""
        try:
            from app.api import request_id_var
            request_id = request_id_var.get("")
        except Exception:
            pass
        # ip_address is populated by callers who have access to the Request object
        return ip_address, request_id

    def log(self, event: AuditEvent) -> None:
        """Record an audit event with deduplication."""
        # Populate request_id from context if not already set
        if not event.request_id:
            _, ctx_rid = self._get_request_context()
            event.request_id = ctx_rid

        with self._lock:
            if self._is_duplicate(event):
                return
            self._events.append(event)
            if len(self._events) > self._max_buffer:
                self._events = self._events[-self._max_buffer:]

        # Write to audit log file
        self._audit_logger.info(event.to_json())

        # Also log to main logger for aggregation
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "critical": logging.CRITICAL,
        }.get(event.severity, logging.INFO)

        logger.log(
            log_level,
            "AUDIT [%s] user=%s case=%s: %s",
            event.event_type, event.user, event.case_id,
            json.dumps(event.details, default=str)[:200],
        )

    # ── Convenience methods ──────────────────────────────────────

    def log_auth_success(self, user: str = "ui_user", ip: str = "") -> None:
        self.log(AuditEvent(
            event_type="auth_login_success",
            user=user, ip_address=ip,
            details={"method": "password"},
        ))

    def log_auth_failure(self, user: str = "unknown", ip: str = "") -> None:
        self.log(AuditEvent(
            event_type="auth_login_failure",
            user=user, ip_address=ip,
            severity="warning",
            details={"method": "password"},
        ))

    def log_query(
        self,
        query: str,
        case_id: str,
        user: str = "api_user",
        request_id: str = "",
        ip_address: str = "",
        latency_seconds: float = 0.0,
        success: bool = True,
    ) -> None:
        event_type: AuditEventType = "query_completed" if success else "query_failed"
        # Hash the query for privacy — do not log plaintext
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        self.log(AuditEvent(
            event_type=event_type,
            user=user, case_id=case_id,
            request_id=request_id,
            ip_address=ip_address,
            details={
                "query_hash": query_hash,
                "query_length": len(query),
                "redaction_method": "sha256_truncated",
                "latency_seconds": latency_seconds,
            },
        ))

    def log_admin_action(
        self,
        action: str,
        user: str = "admin",
        case_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.log(AuditEvent(
            event_type="admin_mcp_tool_used",
            user=user, case_id=case_id,
            severity="warning",
            details={"action": action, **(details or {})},
        ))

    def log_data_event(
        self,
        event_type: AuditEventType,
        case_id: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.log(AuditEvent(
            event_type=event_type,
            case_id=case_id,
            details=details or {},
        ))

    def log_api_key_usage(self, key_prefix: str, endpoint: str, ip: str = "") -> None:
        self.log(AuditEvent(
            event_type="api_key_used",
            user=f"key:{key_prefix}",
            ip_address=ip,
            details={"endpoint": endpoint},
        ))

    def log_pii_event(
        self,
        case_id: str,
        pii_types: list[str],
        count: int,
        action: Literal["detected", "redacted"] = "detected",
    ) -> None:
        event_type: AuditEventType = "pii_detected" if action == "detected" else "pii_redacted"
        if not case_id:
            case_id = "unknown_case"
            logger.warning("PII event logged without case_id — using 'unknown_case'")
        self.log(AuditEvent(
            event_type=event_type,
            case_id=case_id,
            severity="warning",
            details={"pii_types": pii_types, "count": count},
        ))

    def log_feedback_submitted(
        self,
        case_id: str,
        session_id: str,
        num_flags: int,
        ip_address: str = "",
    ) -> None:
        self.log(AuditEvent(
            event_type="feedback_submitted",
            case_id=case_id,
            ip_address=ip_address,
            details={"session_id": session_id, "num_flags": num_flags},
        ))

    def log_chat_saved(
        self,
        case_id: str,
        chat_id: str,
        ip_address: str = "",
    ) -> None:
        self.log(AuditEvent(
            event_type="chat_saved",
            case_id=case_id,
            ip_address=ip_address,
            details={"chat_id": chat_id},
        ))

    def log_secret_rotation(self, secret_name: str, success: bool) -> None:
        self.log(AuditEvent(
            event_type="secret_rotated",
            severity="info" if success else "critical",
            details={"secret_name": secret_name, "success": success},
        ))

    # ── Query interface ──────────────────────────────────────────

    def recent_events(
        self,
        limit: int = 100,
        event_type: AuditEventType | None = None,
        case_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query recent audit events from the in-memory buffer."""
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if case_id:
            events = [e for e in events if e.case_id == case_id]

        return [e.to_dict() for e in events[-limit:]]

    def event_counts(self) -> dict[str, int]:
        """Return counts by event type."""
        with self._lock:
            counts: dict[str, int] = {}
            for e in self._events:
                counts[e.event_type] = counts.get(e.event_type, 0) + 1
            return counts


# Module-level singleton
audit_log = AuditLogger()
