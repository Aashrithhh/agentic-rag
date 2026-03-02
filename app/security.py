"""API security: key authentication, error sanitization, security headers.

When ``settings.api_keys`` is empty (default), authentication is disabled
for backward-compatible local development.  Set ``API_KEYS=key1,key2`` in
``.env`` to enforce API key checks on protected endpoints.
"""

from __future__ import annotations

import logging
import secrets
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _valid_api_keys() -> set[str]:
    """Parse comma-separated API keys from settings."""
    raw = settings.api_keys.strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def require_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)] = None,
) -> str:
    """FastAPI dependency — rejects requests without a valid X-API-Key.

    When ``settings.api_keys`` is empty, all requests are allowed (dev mode).
    """
    valid_keys = _valid_api_keys()
    if not valid_keys:
        return "dev-mode"
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")
    # Timing-safe comparison against each valid key
    if not any(secrets.compare_digest(api_key, k) for k in valid_keys):
        logger.warning("Rejected invalid API key: %s...", api_key[:8])
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


def sanitize_error(exc: Exception) -> str:
    """Return a safe error message that never leaks internals.

    Maps known exception types to user-friendly messages.  Unknown
    exceptions get a generic message with no traceback.
    """
    safe_messages = {
        "ValueError": "Invalid input provided.",
        "KeyError": "Required configuration or data not found.",
        "ConnectionError": "External service temporarily unavailable.",
        "TimeoutError": "Request timed out. Please retry.",
        "OperationalError": "Database temporarily unavailable.",
    }
    return safe_messages.get(type(exc).__name__, "An internal error occurred. Please try again.")
