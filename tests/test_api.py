"""Integration tests for app.api — endpoint behavior + security."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """TestClient with API auth disabled (default — no API_KEYS set)."""
    return TestClient(app)


# ── Health endpoint ──────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_no_auth_required(self, client):
        """Health should work even without an API key."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Cache-Control"] == "no-store"


# ── Cases endpoint ───────────────────────────────────────────────────


class TestCasesEndpoint:
    def test_list_cases_returns_list(self, client):
        resp = client.get("/api/v1/cases")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 2  # big-thorium + purview-exchange

    def test_case_has_expected_fields(self, client):
        resp = client.get("/api/v1/cases")
        case = resp.json()[0]
        assert "case_id" in case
        assert "display_name" in case
        assert "description" in case
        assert "doc_type" in case

    @patch("app.security.settings")
    def test_rejects_invalid_api_key(self, mock_settings, client):
        mock_settings.api_keys = "valid-key-123"
        resp = client.get("/api/v1/cases", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    @patch("app.security.settings")
    def test_rejects_missing_api_key(self, mock_settings, client):
        mock_settings.api_keys = "valid-key-123"
        resp = client.get("/api/v1/cases")
        assert resp.status_code == 401


# ── Query endpoint ───────────────────────────────────────────────────


class TestQueryEndpoint:
    @patch("app.api.compile_graph")
    @patch("app.api.init_db")
    def test_unknown_case_returns_404(self, mock_init, mock_graph, client):
        resp = client.post("/api/v1/query", json={
            "query": "test question",
            "case_id": "nonexistent-case",
        })
        assert resp.status_code == 404

    @patch("app.api.compile_graph")
    @patch("app.api.init_db")
    def test_success_returns_answer(self, mock_init, mock_graph, client):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "answer": "Test answer",
            "grader_score": "yes",
            "loop_count": 0,
            "validation_status": "pass",
            "hallucination_score": 0.12,
            "hallucination_decision": "pass",
            "hallucination_flags": [],
        }
        mock_graph.return_value = mock_agent

        resp = client.post("/api/v1/query", json={
            "query": "What safety incidents were reported?",
            "case_id": "big-thorium",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Test answer"
        assert data["case_id"] == "big-thorium"
        assert data["latency_seconds"] >= 0
        assert data["hallucination_score"] == 0.12
        assert data["hallucination_decision"] == "pass"
        assert data["hallucination_flags"] == []

    @patch("app.api.compile_graph")
    @patch("app.api.init_db")
    def test_hallucination_fields_default_when_missing(self, mock_init, mock_graph, client):
        """When pipeline result lacks hallucination fields, defaults are used."""
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "answer": "Legacy answer",
            "grader_score": "yes",
            "loop_count": 0,
            "validation_status": "pass",
        }
        mock_graph.return_value = mock_agent

        resp = client.post("/api/v1/query", json={
            "query": "What happened?",
            "case_id": "big-thorium",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["hallucination_score"] == 0.0
        assert data["hallucination_decision"] == "pass"
        assert data["hallucination_flags"] == []

    def test_empty_query_rejected(self, client):
        resp = client.post("/api/v1/query", json={
            "query": "",
            "case_id": "big-thorium",
        })
        assert resp.status_code == 422

    @patch("app.api.compile_graph")
    @patch("app.api.init_db")
    def test_pipeline_error_is_sanitized(self, mock_init, mock_graph, client):
        """Exception details must NOT appear in the response."""
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = RuntimeError("SECRET_DB_PASSWORD=hunter2")
        mock_graph.return_value = mock_agent

        resp = client.post("/api/v1/query", json={
            "query": "test",
            "case_id": "big-thorium",
        })
        assert resp.status_code == 500
        assert "SECRET_DB_PASSWORD" not in resp.text
        assert "hunter2" not in resp.text
        assert "internal error" in resp.json()["detail"].lower()
