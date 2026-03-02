"""Lightweight in-process metrics registry — counters, histograms, timers.

Thread-safe singleton. No external dependencies — pure stdlib.

Usage::

    from app.metrics import metrics

    metrics.inc("node_invocations.retriever")

    with metrics.timer("node_latency.retriever"):
        ...  # timed work

    snapshot = metrics.snapshot()  # {"counters": {...}, "histograms": {...}}
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Any

_HISTOGRAM_MAX_OBSERVATIONS = 1000


class _Metrics:
    """Thread-safe metrics collector with counters and histograms."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_HISTOGRAM_MAX_OBSERVATIONS)
        )

    # ── Counters ──────────────────────────────────────────────────

    def inc(self, name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] += amount

    def counter(self, name: str) -> int:
        """Return the current value of a counter."""
        with self._lock:
            return self._counters.get(name, 0)

    # ── Histograms ────────────────────────────────────────────────

    def observe(self, name: str, value: float) -> None:
        """Record an observation in a named histogram."""
        with self._lock:
            self._histograms[name].append(value)

    def histogram_summary(self, name: str) -> dict[str, float]:
        """Return count, mean, p50, p95, p99, max for a histogram."""
        with self._lock:
            data = list(self._histograms.get(name, []))
        if not data:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
        data_sorted = sorted(data)
        n = len(data_sorted)
        return {
            "count": n,
            "mean": round(statistics.mean(data_sorted), 6),
            "p50": round(data_sorted[int(n * 0.50)], 6),
            "p95": round(data_sorted[min(int(n * 0.95), n - 1)], 6),
            "p99": round(data_sorted[min(int(n * 0.99), n - 1)], 6),
            "max": round(data_sorted[-1], 6),
        }

    # ── Timer ─────────────────────────────────────────────────────

    @contextmanager
    def timer(self, name: str):
        """Context manager that observes elapsed seconds in a histogram."""
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.observe(name, elapsed)

    # ── Snapshot ──────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of all metrics."""
        with self._lock:
            counter_names = list(self._counters.keys())
            histogram_names = list(self._histograms.keys())
        return {
            "counters": {name: self.counter(name) for name in counter_names},
            "histograms": {name: self.histogram_summary(name) for name in histogram_names},
        }

    # ── Reset (for testing) ───────────────────────────────────────

    def reset(self) -> None:
        """Clear all counters and histograms."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Module-level singleton
metrics = _Metrics()
