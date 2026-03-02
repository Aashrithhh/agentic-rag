"""Thread-safe in-memory cache with namespaces and LRU eviction.

Namespaces
----------
- ``embedding``  — query text → embedding vector
- ``retrieval``  — (case_id, query, top_k, filter) → search results
- ``llm``        — grader / rewriter / generator LLM responses

Each namespace is independently bounded by
``settings.cache_max_entries_per_namespace``.  When the limit is reached
the *least-recently-used* entry is evicted.

TTL
---
``settings.cache_ttl_seconds = 0`` means entries live until the process
restarts (process-lifetime cache).  A positive value expires entries
after that many seconds.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────


def stable_hash(payload: Any) -> str:
    """Return a deterministic SHA-256 hex digest for *payload*.

    Works with dicts, lists, strings, numbers, and None.  Dict keys are
    sorted so insertion order doesn't matter.
    """
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def key_for(namespace: str, payload: Any, prefix: str = "") -> str:
    """Build a cache key string: ``<namespace>:<prefix>:<hash>``."""
    h = stable_hash(payload)
    parts = [namespace]
    if prefix:
        parts.append(prefix)
    parts.append(h)
    return ":".join(parts)


# ── Cache entry ────────────────────────────────────────────────────────


class _Entry:
    __slots__ = ("value", "created_at")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.created_at = time.monotonic()


# ── Namespace store (LRU + optional TTL) ──────────────────────────────


class _NamespaceStore:
    """LRU cache for a single namespace, protected by its own lock."""

    def __init__(self, max_entries: int, ttl: float) -> None:
        self._max = max_entries
        self._ttl = ttl  # 0 = no expiry
        self._data: OrderedDict[str, _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    # ── public API ────────────────────────────────────────

    def get(self, key: str) -> tuple[bool, Any]:
        """Return ``(True, value)`` on hit, ``(False, None)`` on miss."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return False, None

            # TTL check
            if self._ttl > 0 and (time.monotonic() - entry.created_at) > self._ttl:
                del self._data[key]
                self.misses += 1
                return False, None

            # Move to end (most-recently-used)
            self._data.move_to_end(key)
            self.hits += 1
            return True, entry.value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = _Entry(value)
            else:
                self._data[key] = _Entry(value)
                # Evict LRU if over capacity
                while len(self._data) > self._max:
                    self._data.popitem(last=False)

    def invalidate(self, key: str) -> bool:
        """Remove a single key. Returns True if it existed."""
        with self._lock:
            return self._data.pop(key, None) is not None

    def invalidate_prefix(self, prefix: str) -> int:
        """Remove all keys whose string starts with *prefix*.

        Returns the count of removed entries.
        """
        with self._lock:
            to_delete = [k for k in self._data if k.startswith(prefix)]
            for k in to_delete:
                del self._data[k]
            return len(to_delete)

    def clear(self) -> int:
        """Remove all entries. Returns count removed."""
        with self._lock:
            n = len(self._data)
            self._data.clear()
            return n

    def size(self) -> int:
        with self._lock:
            return len(self._data)

    def stats_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "size": len(self._data),
                "max_entries": self._max,
                "ttl_seconds": self._ttl,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{self.hits / (self.hits + self.misses) * 100:.1f}%"
                if (self.hits + self.misses) > 0
                else "N/A",
            }


# ── Global cache singleton ────────────────────────────────────────────


class _Cache:
    """Process-lifetime cache managing multiple namespaces."""

    def __init__(self) -> None:
        self._namespaces: dict[str, _NamespaceStore] = {}
        self._lock = threading.Lock()

    def _ns(self, namespace: str) -> _NamespaceStore:
        """Return (or lazily create) the namespace store."""
        if namespace not in self._namespaces:
            with self._lock:
                if namespace not in self._namespaces:
                    self._namespaces[namespace] = _NamespaceStore(
                        max_entries=settings.cache_max_entries_per_namespace,
                        ttl=float(settings.cache_ttl_seconds),
                    )
        return self._namespaces[namespace]

    # ── Core get / set ────────────────────────────────────

    def get(self, namespace: str, key: str) -> tuple[bool, Any]:
        if not settings.cache_enabled:
            return False, None
        return self._ns(namespace).get(key)

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not settings.cache_enabled:
            return
        self._ns(namespace).set(key, value)

    # ── Targeted invalidation ─────────────────────────────

    def invalidate_case_retrieval_cache(self, case_id: str) -> int:
        """Remove all retrieval cache entries for a specific case."""
        prefix = f"retrieval:{case_id}:"
        removed = self._ns("retrieval").invalidate_prefix(prefix)
        if removed:
            logger.info("Cache: invalidated %d retrieval entries for case '%s'", removed, case_id)
        return removed

    def invalidate_all_llm_cache(self) -> int:
        """Remove all LLM (grader/rewriter/generator) cache entries."""
        removed = self._ns("llm").clear()
        if removed:
            logger.info("Cache: invalidated %d LLM entries", removed)
        return removed

    def invalidate_namespace(self, namespace: str) -> int:
        """Clear an entire namespace."""
        return self._ns(namespace).clear()

    def invalidate_all(self) -> int:
        """Clear every namespace."""
        total = 0
        with self._lock:
            for ns in self._namespaces.values():
                total += ns.clear()
        if total:
            logger.info("Cache: invalidated all %d entries", total)
        return total

    # ── Diagnostics ───────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Return per-namespace stats for diagnostics / UI display."""
        result: dict[str, Any] = {
            "enabled": settings.cache_enabled,
            "ttl_seconds": settings.cache_ttl_seconds,
            "max_entries_per_namespace": settings.cache_max_entries_per_namespace,
            "namespaces": {},
        }
        with self._lock:
            for name, ns in self._namespaces.items():
                result["namespaces"][name] = ns.stats_dict()

        # Totals
        total_hits = sum(ns.get("hits", 0) for ns in result["namespaces"].values())
        total_misses = sum(ns.get("misses", 0) for ns in result["namespaces"].values())
        total_size = sum(ns.get("size", 0) for ns in result["namespaces"].values())
        result["total_entries"] = total_size
        result["total_hits"] = total_hits
        result["total_misses"] = total_misses
        result["total_hit_rate"] = (
            f"{total_hits / (total_hits + total_misses) * 100:.1f}%"
            if (total_hits + total_misses) > 0
            else "N/A"
        )
        return result


# Module-level singleton
cache = _Cache()
