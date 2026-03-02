"""Tests for app.evaluation — retrieval metrics (pure math, no mocking)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from app.evaluation import mean_reciprocal_rank, precision_at_k, recall_at_k


# ── precision_at_k ──────────────────────────────────────────────


def test_precision_perfect():
    """All retrieved docs are relevant → precision = 1.0."""
    assert precision_at_k(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0


def test_precision_half():
    """Half the top-k docs are relevant → precision = 0.5."""
    assert precision_at_k(["a", "x", "b", "y"], ["a", "b"], k=4) == 0.5


def test_precision_none_relevant():
    """No relevant docs in top-k → precision = 0.0."""
    assert precision_at_k(["x", "y", "z"], ["a", "b"], k=3) == 0.0


def test_precision_empty_relevant():
    """Empty relevant set → precision = 0.0."""
    assert precision_at_k(["a", "b"], [], k=2) == 0.0


def test_precision_k_zero():
    """k=0 → precision = 0.0 (avoid division by zero)."""
    assert precision_at_k(["a"], ["a"], k=0) == 0.0


# ── recall_at_k ─────────────────────────────────────────────────


def test_recall_perfect():
    """All relevant docs appear in top-k → recall = 1.0."""
    assert recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0


def test_recall_partial():
    """Only some relevant docs found → recall between 0 and 1."""
    assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == 0.5


def test_recall_none():
    """No relevant docs found → recall = 0.0."""
    assert recall_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_recall_empty_relevant():
    """Empty relevant set → recall = 0.0."""
    assert recall_at_k(["a", "b"], [], k=2) == 0.0


# ── mean_reciprocal_rank ────────────────────────────────────────


def test_mrr_first_position():
    """First relevant doc at position 1 → MRR = 1.0."""
    assert mean_reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_second_position():
    """First relevant doc at position 2 → MRR = 0.5."""
    assert mean_reciprocal_rank(["x", "a", "b"], ["a", "b"]) == 0.5


def test_mrr_third_position():
    """First relevant doc at position 3 → MRR ≈ 0.333."""
    result = mean_reciprocal_rank(["x", "y", "a"], ["a"])
    assert abs(result - 1 / 3) < 1e-9


def test_mrr_no_relevant():
    """No relevant docs at all → MRR = 0.0."""
    assert mean_reciprocal_rank(["x", "y", "z"], ["a"]) == 0.0
