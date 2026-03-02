"""Tests for Settings.validate_secrets() — secret hygiene at startup."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_database_url_warns():
    """Default user:password credentials should trigger a warning."""
    from app.config import Settings
    s = Settings(database_url="postgresql+psycopg://user:password@localhost:5432/audit_rag")
    warnings = s.validate_secrets()
    assert any("default credentials" in w for w in warnings)


def test_placeholder_cohere_key_warns():
    """Placeholder COHERE_API_KEY should trigger a warning."""
    from app.config import Settings
    s = Settings(cohere_api_key="")
    warnings = s.validate_secrets()
    assert any("COHERE_API_KEY" in w for w in warnings)


def test_valid_config_no_llm_warning():
    """With a real OpenAI key, no LLM warning should appear."""
    from app.config import Settings
    s = Settings(
        openai_api_key="sk-realkey123",
        cohere_api_key="realkey456",
        database_url="postgresql+psycopg://prod_user:strongpass@db.internal:5432/audit_rag",
    )
    warnings = s.validate_secrets()
    assert not any("LLM API key" in w for w in warnings)
    assert not any("default credentials" in w for w in warnings)
