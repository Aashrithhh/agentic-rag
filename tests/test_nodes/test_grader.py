"""Tests for app.nodes.grader — relevance grading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.nodes.grader import GradeResult, grader_node


def test_no_docs_returns_no(sample_state):
    """When retrieved_docs is empty, grader returns score='no'."""
    state = {**sample_state, "retrieved_docs": []}
    result = grader_node(state)
    assert result["grader_score"] == "no"
    assert "No documents" in result["grader_reasoning"]


@patch("app.nodes.grader.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.grader.ChatOpenAI")
def test_relevant_docs_return_yes(mock_openai, mock_retry, mock_settings, sample_state):
    """When LLM judges docs as relevant, grader returns score='yes'."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_retry.return_value = GradeResult(score="yes", reasoning="Docs contain safety info.")

    result = grader_node(sample_state)
    assert result["grader_score"] == "yes"
    assert "safety" in result["grader_reasoning"].lower()


@patch("app.nodes.grader.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.grader.ChatOpenAI")
def test_irrelevant_docs_return_no(mock_openai, mock_retry, mock_settings, sample_state):
    """When LLM judges docs as irrelevant, grader returns score='no'."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_retry.return_value = GradeResult(score="no", reasoning="Documents are about finance, not safety.")

    result = grader_node(sample_state)
    assert result["grader_score"] == "no"
