"""Tests for app.cache — deterministic hashing, LRU eviction, TTL."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from app.cache import _NamespaceStore, key_for, stable_hash


# ── stable_hash ────────────────────────────────────────────────────


def test_stable_hash_deterministic():
    """Same payload always produces the same hash."""
    payload = {"model": "gpt-4o", "query": "hello"}
    assert stable_hash(payload) == stable_hash(payload)


def test_stable_hash_order_independent():
    """Dict key order should not affect the hash."""
    a = {"z": 1, "a": 2}
    b = {"a": 2, "z": 1}
    assert stable_hash(a) == stable_hash(b)


def test_stable_hash_different_payloads():
    """Different payloads produce different hashes."""
    assert stable_hash("hello") != stable_hash("world")


# ── key_for ─────────────────────────────────────────────────────────


def test_key_for_structure():
    """key_for returns namespace:prefix:hash format."""
    k = key_for("embedding", {"text": "hello"}, prefix="case-1")
    parts = k.split(":")
    assert parts[0] == "embedding"
    assert parts[1] == "case-1"
    assert len(parts) == 3


def test_key_for_no_prefix():
    """key_for without prefix returns namespace:hash."""
    k = key_for("llm", {"query": "test"})
    parts = k.split(":")
    assert parts[0] == "llm"
    assert len(parts) == 2


# ── LRU eviction ───────────────────────────────────────────────────


def test_lru_eviction():
    """When max_entries is exceeded, oldest entry is evicted."""
    store = _NamespaceStore(max_entries=3, ttl=0)
    store.set("a", 1)
    store.set("b", 2)
    store.set("c", 3)
    store.set("d", 4)  # "a" should be evicted

    hit_a, _ = store.get("a")
    hit_d, val_d = store.get("d")
    assert not hit_a, "Evicted entry 'a' should not be found"
    assert hit_d and val_d == 4


def test_lru_access_refreshes():
    """Accessing an entry moves it to most-recently-used, keeping it alive."""
    store = _NamespaceStore(max_entries=3, ttl=0)
    store.set("a", 1)
    store.set("b", 2)
    store.set("c", 3)

    # Access "a" to refresh it
    store.get("a")

    # Now insert "d" — "b" (the true LRU) should be evicted, not "a"
    store.set("d", 4)

    hit_a, _ = store.get("a")
    hit_b, _ = store.get("b")
    assert hit_a, "'a' was recently accessed, should still exist"
    assert not hit_b, "'b' was the LRU and should have been evicted"


# ── TTL expiry ──────────────────────────────────────────────────────


def test_ttl_expiry():
    """Entries expire after TTL seconds."""
    store = _NamespaceStore(max_entries=100, ttl=0.1)  # 100ms TTL
    store.set("key", "value")

    hit, val = store.get("key")
    assert hit and val == "value", "Entry should exist immediately"

    time.sleep(0.15)

    hit, val = store.get("key")
    assert not hit, "Entry should have expired after TTL"


def test_zero_ttl_never_expires():
    """TTL=0 means entries never expire."""
    store = _NamespaceStore(max_entries=100, ttl=0)
    store.set("key", "value")

    # Even after a small sleep, it should still be there
    time.sleep(0.05)
    hit, val = store.get("key")
    assert hit and val == "value"


# ── Stats ──────────────────────────────────────────────────────────


def test_stats_tracking():
    """Hits and misses are tracked correctly."""
    store = _NamespaceStore(max_entries=100, ttl=0)
    store.set("x", 1)
    store.get("x")   # hit
    store.get("x")   # hit
    store.get("y")   # miss

    stats = store.stats_dict()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1


# ── Invalidation ──────────────────────────────────────────────────


def test_invalidate_prefix():
    """invalidate_prefix removes matching entries and returns count."""
    store = _NamespaceStore(max_entries=100, ttl=0)
    store.set("retrieval:case-1:abc", "doc1")
    store.set("retrieval:case-1:def", "doc2")
    store.set("retrieval:case-2:ghi", "doc3")

    removed = store.invalidate_prefix("retrieval:case-1:")
    assert removed == 2
    assert store.size() == 1
