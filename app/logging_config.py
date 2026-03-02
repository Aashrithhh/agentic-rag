"""Structured logging configuration.

Supports two output formats controlled by ``LOG_FORMAT`` env var:
  - ``text`` (default) — human-readable coloured output for development
  - ``json`` — machine-readable JSON lines for production log aggregators

Usage::

    from app.logging_config import setup_logging
    setup_logging()  # call once at startup
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line.

    Fields: ``timestamp``, ``level``, ``logger``, ``message``,
    plus any ``extra`` keys attached to the record.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Inject request_id from contextvars if available
        try:
            from app.api import request_id_var
            rid = request_id_var.get("")
            if rid:
                log_entry["request_id"] = rid
        except ImportError:
            pass
        # Preserve extra fields added via logger.info("msg", extra={...})
        for key in ("case_id", "node", "duration_ms", "chunks", "query", "request_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val
        return json.dumps(log_entry, default=str)


_TEXT_FORMAT = "%(levelname)-8s | %(name)-30s | %(message)s"


def setup_logging(
    *,
    json_format: bool | None = None,
    level: str | None = None,
) -> None:
    """Configure the root logger.

    Parameters
    ----------
    json_format:
        Force JSON output. When *None*, reads from ``settings.log_format``.
    level:
        Log level string (DEBUG, INFO, WARNING, etc.).
        When *None*, reads from ``settings.log_level``.
    """
    from app.config import settings

    use_json = json_format if json_format is not None else (
        getattr(settings, "log_format", "text") == "json"
    )
    log_level = level or getattr(settings, "log_level", "INFO")

    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicate output on re-init
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))

    root.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
