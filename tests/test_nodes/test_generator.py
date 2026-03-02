"""Tests for app.nodes.generator — answer synthesis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.nodes.generator import generator_node


@patch("app.nodes.generator.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.generator.ChatOpenAI")
def test_generates_answer(mock_openai, mock_retry, mock_settings, sample_state):
    """Generator produces an answer string from retrieved docs."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_response = MagicMock()
    mock_response.content = "## Safety Findings\n\nThe welding audit found 3 critical issues. [Source: safety_audit.pdf, p.5]"
    mock_retry.return_value = mock_response

    result = generator_node(sample_state)
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "Safety" in result["answer"]


@patch("app.nodes.generator.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.generator.ChatOpenAI")
def test_handles_no_docs(mock_openai, mock_retry, mock_settings):
    """Generator works even with empty document list."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    mock_response = MagicMock()
    mock_response.content = "No documents were available to answer this query."
    mock_retry.return_value = mock_response

    state = {
        "query": "What is the revenue?",
        "case_id": "test-case",
        "retrieved_docs": [],
        "validation_results": [],
        "validation_status": "pass",
    }
    result = generator_node(state)
    assert "answer" in result


@patch("app.nodes.generator.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.generator.ChatOpenAI")
def test_handles_validation_failure(mock_openai, mock_retry, mock_settings, sample_state):
    """Generator includes discrepancy info when validation_status is 'fail'."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"

    state = {
        **sample_state,
        "validation_status": "fail",
        "validation_results": [
            {"claim": "Revenue was $2.3M", "endpoint": "http://api/filings",
             "is_valid": False, "status_code": 200, "api_response": {}},
        ],
    }

    mock_response = MagicMock()
    mock_response.content = "## DISCREPANCY\n\nThe revenue claim could not be verified."
    mock_retry.return_value = mock_response

    result = generator_node(state)
    assert "DISCREPANCY" in result["answer"]
