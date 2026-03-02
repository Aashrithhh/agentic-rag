"""Shared fixtures for the test suite."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from langchain_core.documents import Document


# ── Ensure dummy API keys so Settings() doesn't fail in CI ──────────


@pytest.fixture(autouse=True)
def _dummy_env(monkeypatch):
    """Provide minimal env vars so pydantic Settings() validates."""
    defaults = {
        "OPENAI_API_KEY": "test-key-not-real",
        "COHERE_API_KEY": "test-key-not-real",
        "DATABASE_URL": "postgresql+psycopg://user:pw@localhost:5432/test_db",
    }
    for key, val in defaults.items():
        monkeypatch.setenv(key, val)


# ── Sample data fixtures ────────────────────────────────────────────


@pytest.fixture
def sample_docs() -> list[Document]:
    """A handful of fake documents for grader / generator tests."""
    return [
        Document(
            page_content="Big Thorium reported $2.3M revenue in Q4 2024.",
            metadata={"source": "10K_2024.pdf", "page": 12, "doc_type": "compliance-report"},
        ),
        Document(
            page_content="The welding safety audit found 3 critical deficiencies.",
            metadata={"source": "safety_audit.pdf", "page": 5, "doc_type": "compliance-report"},
        ),
        Document(
            page_content="Email from William Davis regarding worker housing conditions.",
            metadata={"source": "emails_export.pst", "page": None, "doc_type": "exchange-email"},
        ),
    ]


@pytest.fixture
def sample_state(sample_docs) -> dict:
    """A populated AgentState dict for node tests."""
    return {
        "query": "What safety incidents were reported at Big Thorium?",
        "case_id": "big-thorium",
        "retrieved_docs": sample_docs,
        "loop_count": 0,
        "grader_score": "yes",
        "grader_reasoning": "Documents contain safety audit information.",
        "validation_results": [],
        "validation_status": "pass",
    }
