"""Tests for app.nodes.validator — claim extraction and HTTP verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.nodes.validator import (
    ClaimList,
    ExtractedClaim,
    _aggregate_status,
    validator_node,
)


# ── _aggregate_status (pure logic) ──────────────────────────────


def test_aggregate_all_valid():
    """All valid results → 'pass'."""
    results = [
        {"claim": "a", "is_valid": True, "endpoint": "", "status_code": 200, "api_response": {}},
        {"claim": "b", "is_valid": True, "endpoint": "", "status_code": 200, "api_response": {}},
    ]
    assert _aggregate_status(results) == "pass"


def test_aggregate_none_valid():
    """No valid results → 'fail'."""
    results = [
        {"claim": "a", "is_valid": False, "endpoint": "", "status_code": 404, "api_response": {}},
    ]
    assert _aggregate_status(results) == "fail"


def test_aggregate_mixed():
    """Mix of valid and invalid → 'partial'."""
    results = [
        {"claim": "a", "is_valid": True, "endpoint": "", "status_code": 200, "api_response": {}},
        {"claim": "b", "is_valid": False, "endpoint": "", "status_code": 404, "api_response": {}},
    ]
    assert _aggregate_status(results) == "partial"


def test_aggregate_empty():
    """Empty results → 'pass'."""
    assert _aggregate_status([]) == "pass"


# ── validator_node ───────────────────────────────────────────────


def test_no_docs_returns_pass(sample_state):
    """When no docs, validator returns pass status."""
    state = {**sample_state, "retrieved_docs": []}
    result = validator_node(state)
    assert result["validation_status"] == "pass"
    assert result["validation_results"] == []


@patch("app.nodes.validator.settings")
@patch("app.resilience.invoke_with_retry")
@patch("app.nodes.validator.ChatOpenAI")
@patch("app.nodes.validator.asyncio")
def test_claims_extracted_and_verified(
    mock_asyncio, mock_openai, mock_retry, mock_settings, sample_state
):
    """When LLM extracts claims, they are verified against the API."""
    mock_settings.cache_llm_enabled = False
    mock_settings.openai_model = "gpt-4o"
    mock_settings.openai_api_key = "fake"
    mock_settings.validation_api_base = "http://localhost:8000"
    # Critical fact protection settings
    mock_settings.critical_fact_protection = False

    claim = ExtractedClaim(
        claim="Revenue was $2.3M",
        table="filings",
        filter_field="entity",
        filter_value="Big Thorium",
    )
    mock_retry.return_value = ClaimList(claims=[claim])

    # Mock the async verification
    mock_asyncio.run.return_value = [
        {
            "claim": "Revenue was $2.3M",
            "endpoint": "http://localhost:8000/api/validate-all/filings",
            "status_code": 200,
            "api_response": {"value": [{"amount": 2300000}]},
            "is_valid": True,
        }
    ]

    result = validator_node(sample_state)
    assert result["validation_status"] == "pass"
    assert len(result["validation_results"]) == 1
