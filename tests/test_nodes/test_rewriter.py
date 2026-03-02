"""Tests for app.nodes.rewriter — query rewriting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.nodes.rewriter import rewriter_node


@patch("app.nodes.rewriter.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.rewriter.ChatOpenAI")
def test_rewrites_query(mock_openai, mock_retry, mock_settings, sample_state):
    """Rewriter produces a new query string and increments loop_count."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_response = MagicMock()
    mock_response.content = "Big Thorium welding safety incidents audit deficiencies"
    mock_retry.return_value = mock_response

    state = {**sample_state, "loop_count": 1}
    result = rewriter_node(state)

    assert "rewritten_query" in result
    assert result["loop_count"] == 2
    assert "safety" in result["rewritten_query"].lower()


@patch("app.nodes.rewriter.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.rewriter.ChatOpenAI")
def test_strips_quotes_from_response(mock_openai, mock_retry, mock_settings, sample_state):
    """Rewriter strips surrounding quotes from the LLM response."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_response = MagicMock()
    mock_response.content = '"safety incidents at Big Thorium plant"'
    mock_retry.return_value = mock_response

    result = rewriter_node(sample_state)
    assert not result["rewritten_query"].startswith('"')
    assert not result["rewritten_query"].endswith('"')
