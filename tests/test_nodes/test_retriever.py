"""Tests for app.nodes.retriever — hybrid search retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from app.nodes.retriever import retriever_node


def test_missing_case_id_raises():
    """Retriever must raise ValueError when case_id is missing."""
    state = {"query": "test question"}
    with pytest.raises(ValueError, match="case_id is missing"):
        retriever_node(state)


def test_empty_case_id_raises():
    """Retriever must raise ValueError when case_id is empty string."""
    state = {"query": "test question", "case_id": ""}
    with pytest.raises(ValueError, match="case_id is missing"):
        retriever_node(state)


@patch("app.nodes.retriever.settings")
@patch("app.resilience.db_invoke_with_retry")
@patch("app.nodes.retriever._embed_query")
@patch("app.nodes.retriever.get_engine")
@patch("app.cases.metadata_filter_for_case")
def test_returns_documents(
    mock_filter, mock_engine, mock_embed, mock_db_retry, mock_settings
):
    """Retriever returns Document objects from hybrid_search results."""
    mock_settings.cache_retrieval_enabled = False
    mock_settings.top_k = 3
    # Email-pipeline feature flags (must be set to avoid MagicMock comparisons)
    mock_settings.neighbor_stitching = False
    mock_settings.attachment_context_link = False
    mock_settings.metadata_boost_enabled = False
    mock_settings.rerank_enabled = False

    mock_filter.return_value = {"doc_type": "compliance-report"}
    mock_engine.return_value = MagicMock()
    mock_embed.return_value = [0.1] * 1024

    mock_db_retry.return_value = [
        {
            "id": 1,
            "content": "Safety audit results for Q4.",
            "source": "audit.pdf",
            "page": 5,
            "doc_type": "compliance-report",
            "entity_name": "Big Thorium",
            "rrf_score": 0.85,
            "metadata_extra": None,
        },
    ]

    state = {"query": "safety audit", "case_id": "big-thorium"}
    result = retriever_node(state)

    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0].page_content == "Safety audit results for Q4."
    assert result["retrieved_docs"][0].metadata["source"] == "audit.pdf"
