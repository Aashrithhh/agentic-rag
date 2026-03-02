"""Tests for app.metrics — counters, histograms, timer, reset."""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.unit

from app.metrics import _Metrics


@pytest.fixture
def m():
    """Fresh metrics instance for each test."""
    return _Metrics()


# ── Counters ──────────────────────────────────────────────────────


def test_counter_increment(m):
    """inc() increases the counter and counter() reads it."""
    m.inc("requests")
    m.inc("requests")
    m.inc("requests", 3)
    assert m.counter("requests") == 5


def test_counter_default_zero(m):
    """Unset counter returns 0."""
    assert m.counter("nonexistent") == 0


# ── Histograms ────────────────────────────────────────────────────


def test_histogram_observation(m):
    """observe() records values and summary computes stats."""
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.observe("latency", v)
    summary = m.histogram_summary("latency")
    assert summary["count"] == 5
    assert summary["mean"] == 3.0
    assert summary["max"] == 5.0


def test_histogram_empty(m):
    """Empty histogram returns zero-valued summary."""
    summary = m.histogram_summary("empty")
    assert summary["count"] == 0
    assert summary["mean"] == 0.0


# ── Timer ─────────────────────────────────────────────────────────


def test_timer_context_manager(m):
    """timer() records elapsed time as a histogram observation."""
    with m.timer("op_time"):
        time.sleep(0.01)
    summary = m.histogram_summary("op_time")
    assert summary["count"] == 1
    assert summary["max"] >= 0.005  # at least 5ms


# ── Snapshot ──────────────────────────────────────────────────────


def test_snapshot(m):
    """snapshot() returns counters and histograms as a dict."""
    m.inc("a")
    m.observe("b", 1.0)
    snap = m.snapshot()
    assert "counters" in snap
    assert "histograms" in snap
    assert snap["counters"]["a"] == 1
    assert snap["histograms"]["b"]["count"] == 1


# ── Reset ─────────────────────────────────────────────────────────


def test_reset(m):
    """reset() clears all counters and histograms."""
    m.inc("x")
    m.observe("y", 1.0)
    m.reset()
    assert m.counter("x") == 0
    assert m.histogram_summary("y")["count"] == 0
    snap = m.snapshot()
    assert snap["counters"] == {}
    assert snap["histograms"] == {}
