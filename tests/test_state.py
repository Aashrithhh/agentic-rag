"""Tests for app.state — verify AgentState shape and typing."""

from __future__ import annotations

import typing

import pytest

pytestmark = pytest.mark.unit

from app.state import AgentState, ValidationResult


def test_no_duplicate_keys():
    """AgentState should have no duplicate keys (was a real bug: metadata_filter)."""
    hints = typing.get_type_hints(AgentState)
    keys = list(hints.keys())
    assert len(keys) == len(set(keys)), f"Duplicate keys found: {keys}"


def test_metadata_filter_typed_correctly():
    """metadata_filter should be dict[str, str], not bare dict."""
    hints = typing.get_type_hints(AgentState)
    assert "metadata_filter" in hints
    # Check it's dict[str, str] — the origin is dict and args are (str, str)
    mf = hints["metadata_filter"]
    assert getattr(mf, "__origin__", None) is dict
    assert mf.__args__ == (str, str)


def test_validation_result_keys():
    """ValidationResult should have the expected fields."""
    hints = typing.get_type_hints(ValidationResult)
    expected = {"claim", "endpoint", "status_code", "api_response", "is_valid"}
    assert expected == set(hints.keys())
