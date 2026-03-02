"""Tests for app.resilience — retry wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.resilience import db_invoke_with_retry, invoke_with_retry


class _FakeChain:
    """Controllable mock chain with an invoke method."""

    def __init__(self, side_effects: list):
        self._effects = list(side_effects)
        self._call_count = 0

    def invoke(self, inputs):
        self._call_count += 1
        effect = self._effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    @property
    def call_count(self):
        return self._call_count


# ── invoke_with_retry ───────────────────────────────────────────


def test_invoke_success_first_try():
    """When invoke succeeds on first call, return the result."""
    chain = _FakeChain([{"answer": "ok"}])
    result = invoke_with_retry(chain, {"q": "test"}, node_name="test")
    assert result == {"answer": "ok"}
    assert chain.call_count == 1


def test_invoke_retry_then_succeed():
    """When first call fails but second succeeds, return the result."""
    chain = _FakeChain([RuntimeError("transient"), {"answer": "recovered"}])
    result = invoke_with_retry(chain, {"q": "test"}, node_name="test", max_attempts=3)
    assert result == {"answer": "recovered"}
    assert chain.call_count == 2


def test_invoke_all_fail_with_fallback():
    """When all retries fail and fallback is provided, return fallback."""
    chain = _FakeChain([RuntimeError("fail")] * 3)
    fallback = {"answer": "fallback"}
    result = invoke_with_retry(
        chain, {"q": "test"}, node_name="test", max_attempts=3, fallback=fallback,
    )
    assert result == fallback
    assert chain.call_count == 3


def test_invoke_all_fail_no_fallback_raises():
    """When all retries fail and no fallback, raise the exception."""
    chain = _FakeChain([RuntimeError("fail")] * 3)
    with pytest.raises(RuntimeError, match="fail"):
        invoke_with_retry(chain, {"q": "test"}, node_name="test", max_attempts=3)


# ── db_invoke_with_retry ────────────────────────────────────────


def test_db_invoke_success():
    """DB call succeeds on first try."""
    func = MagicMock(return_value=[{"id": 1}])
    result = db_invoke_with_retry(func, "arg1", node_name="test", key="val")
    assert result == [{"id": 1}]
    func.assert_called_once_with("arg1", key="val")


def test_db_invoke_retry():
    """DB call fails once then succeeds."""
    func = MagicMock(side_effect=[ConnectionError("lost"), [{"id": 1}]])
    result = db_invoke_with_retry(func, node_name="test", max_attempts=3)
    assert result == [{"id": 1}]
    assert func.call_count == 2


def test_db_invoke_exhausted_raises():
    """When all DB retries fail, the exception propagates."""
    func = MagicMock(side_effect=ConnectionError("down"))
    with pytest.raises(ConnectionError):
        db_invoke_with_retry(func, node_name="test", max_attempts=2)
