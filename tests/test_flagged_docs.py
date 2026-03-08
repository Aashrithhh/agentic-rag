"""Tests for the flagged-documents feature (DB helpers + API endpoints)."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app


# ───────────────────────────────────────────────────────────────────────
# DB helper unit tests (mock the engine / connection)
# ───────────────────────────────────────────────────────────────────────


class TestSavedFlaggedDocsTable:
    """Verify the table definition is correct."""

    def test_table_exists_in_metadata(self):
        from app.db import metadata_obj, saved_flagged_docs
        assert "saved_flagged_docs" in metadata_obj.tables
        assert saved_flagged_docs is not None

    def test_table_columns(self):
        from app.db import saved_flagged_docs
        col_names = {c.name for c in saved_flagged_docs.columns}
        expected = {"id", "case_id", "filename", "query_text", "saved_at", "saved_by", "docs_json"}
        assert expected.issubset(col_names)

    def test_unique_constraint(self):
        from app.db import saved_flagged_docs
        constraints = [c for c in saved_flagged_docs.constraints if hasattr(c, "columns")]
        uq_names = [c.name for c in constraints if c.name and "uq_" in c.name]
        assert "uq_saved_flagged_docs_case_filename" in uq_names


class TestSaveFlaggedDocSet:
    """Test the save_flagged_doc_set helper."""

    def test_save_calls_insert(self):
        from app.db import save_flagged_doc_set

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.scalar_one.return_value = 42

        result = save_flagged_doc_set(
            mock_engine,
            case_id="test-case",
            filename="test_file",
            query_text="What happened?",
            docs_json=[{"source": "a.pdf", "content": "hello"}],
            saved_by="tester",
        )

        assert result == 42
        assert mock_conn.execute.called


class TestListFlaggedDocSets:
    """Test the list_flagged_doc_sets helper."""

    def test_list_returns_rows(self):
        from app.db import list_flagged_doc_sets

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_row = {
            "id": 1,
            "case_id": "test-case",
            "filename": "batch1",
            "query_text": "test query",
            "saved_at": datetime(2026, 3, 5, 12, 0, 0),
            "saved_by": None,
            "doc_count": 3,
        }
        mock_conn.execute.return_value.mappings.return_value.all.return_value = [mock_row]

        rows = list_flagged_doc_sets(mock_engine, "test-case")
        assert len(rows) == 1
        assert rows[0]["filename"] == "batch1"
        assert rows[0]["doc_count"] == 3


class TestGetFlaggedDocSet:
    """Test the get_flagged_doc_set helper."""

    def test_get_returns_row(self):
        from app.db import get_flagged_doc_set

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        mock_row = {
            "id": 1,
            "case_id": "test-case",
            "filename": "batch1",
            "query_text": "test query",
            "saved_at": datetime(2026, 3, 5, 12, 0, 0),
            "saved_by": None,
            "docs_json": [{"source": "a.pdf"}],
        }
        mock_conn.execute.return_value.mappings.return_value.first.return_value = mock_row

        result = get_flagged_doc_set(mock_engine, 1)
        assert result is not None
        assert result["filename"] == "batch1"

    def test_get_returns_none_when_missing(self):
        from app.db import get_flagged_doc_set

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.mappings.return_value.first.return_value = None

        result = get_flagged_doc_set(mock_engine, 999)
        assert result is None


class TestDeleteFlaggedDocSet:
    """Test the delete_flagged_doc_set helper."""

    def test_delete_returns_true(self):
        from app.db import delete_flagged_doc_set

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.rowcount = 1

        assert delete_flagged_doc_set(mock_engine, 1) is True

    def test_delete_returns_false_when_missing(self):
        from app.db import delete_flagged_doc_set

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.rowcount = 0

        assert delete_flagged_doc_set(mock_engine, 999) is False


# ───────────────────────────────────────────────────────────────────────
# API endpoint tests (via FastAPI TestClient with mocked DB)
# ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def api_client():
    """TestClient for the FastAPI app (auth passes when no API_KEYS set)."""
    return TestClient(app)


class TestSaveFlaggedDocsEndpoint:

    @patch("app.db.save_flagged_doc_set", return_value=7)
    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_save_success(self, mock_cases, mock_init, mock_save, api_client):
        from app.cases import CaseConfig
        mock_cases.return_value = [
            CaseConfig(
                case_id="test-case",
                display_name="Test",
                description="Test case",
                doc_type="email",
                data_dir="data/test",
                sample_questions=["q?"],
            )
        ]
        mock_init.return_value = MagicMock()

        resp = api_client.post(
            "/api/v1/flagged-docs/save",
            json={
                "case_id": "test-case",
                "filename": "batch1",
                "query_text": "What happened?",
                "docs": [
                    {"source": "a.pdf", "content": "hello world", "metadata": {}},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["id"] == 7
        assert body["doc_count"] == 1

    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_save_unknown_case(self, mock_cases, mock_init, api_client):
        mock_cases.return_value = []

        resp = api_client.post(
            "/api/v1/flagged-docs/save",
            json={
                "case_id": "nonexistent",
                "filename": "batch1",
                "query_text": "q",
                "docs": [{"source": "a.pdf", "content": "x"}],
            },
        )
        assert resp.status_code == 404

    def test_save_invalid_filename(self, api_client):
        resp = api_client.post(
            "/api/v1/flagged-docs/save",
            json={
                "case_id": "test-case",
                "filename": "../../etc/passwd",
                "query_text": "q",
                "docs": [{"source": "a.pdf", "content": "x"}],
            },
        )
        assert resp.status_code == 422  # validation error

    def test_save_empty_docs(self, api_client):
        resp = api_client.post(
            "/api/v1/flagged-docs/save",
            json={
                "case_id": "test-case",
                "filename": "batch1",
                "query_text": "q",
                "docs": [],
            },
        )
        assert resp.status_code == 422  # min_length=1 on docs


class TestListFlaggedDocsEndpoint:

    @patch("app.api.init_db")
    @patch("app.db.list_flagged_doc_sets")
    def test_list_success(self, mock_list, mock_init, api_client):
        mock_init.return_value = MagicMock()
        mock_list.return_value = [
            {
                "id": 1,
                "case_id": "test-case",
                "filename": "batch1",
                "query_text": "q",
                "saved_at": datetime(2026, 3, 5),
                "saved_by": None,
                "doc_count": 2,
            }
        ]

        resp = api_client.get(
            "/api/v1/flagged-docs/list",
            params={"case_id": "test-case"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["filename"] == "batch1"


class TestGetFlaggedDocSetEndpoint:

    @patch("app.db.get_flagged_doc_set")
    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_get_success(self, mock_cases, mock_init, mock_get, api_client):
        from app.cases import CaseConfig
        mock_cases.return_value = [
            CaseConfig(
                case_id="test-case",
                display_name="Test",
                description="d",
                doc_type="email",
                data_dir="data/test",
                sample_questions=["q?"],
            )
        ]
        mock_init.return_value = MagicMock()
        mock_get.return_value = {
            "id": 1,
            "case_id": "test-case",
            "filename": "batch1",
            "query_text": "q",
            "saved_at": datetime(2026, 3, 5),
            "saved_by": None,
            "docs_json": [{"source": "a.pdf"}],
        }

        resp = api_client.get("/api/v1/flagged-docs/1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "batch1"
        assert len(body["docs"]) == 1

    @patch("app.db.get_flagged_doc_set", return_value=None)
    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_get_not_found(self, mock_cases, mock_init, mock_get, api_client):
        from app.cases import CaseConfig
        mock_cases.return_value = [
            CaseConfig(
                case_id="test-case",
                display_name="Test",
                description="d",
                doc_type="email",
                data_dir="data/test",
                sample_questions=["q?"],
            )
        ]
        mock_init.return_value = MagicMock()

        resp = api_client.get("/api/v1/flagged-docs/999")
        assert resp.status_code == 404


class TestDeleteFlaggedDocSetEndpoint:

    @patch("app.db.delete_flagged_doc_set", return_value=True)
    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_delete_success(self, mock_cases, mock_init, mock_del, api_client):
        from app.cases import CaseConfig
        mock_cases.return_value = [
            CaseConfig(
                case_id="test-case",
                display_name="Test",
                description="d",
                doc_type="email",
                data_dir="data/test",
                sample_questions=["q?"],
            )
        ]
        mock_init.return_value = MagicMock()

        resp = api_client.delete("/api/v1/flagged-docs/1")
        assert resp.status_code == 200
        assert resp.json()["deleted_id"] == 1

    @patch("app.db.delete_flagged_doc_set", return_value=False)
    @patch("app.api.init_db")
    @patch("app.api.list_cases")
    def test_delete_not_found(self, mock_cases, mock_init, mock_del, api_client):
        from app.cases import CaseConfig
        mock_cases.return_value = [
            CaseConfig(
                case_id="test-case",
                display_name="Test",
                description="d",
                doc_type="email",
                data_dir="data/test",
                sample_questions=["q?"],
            )
        ]
        mock_init.return_value = MagicMock()

        resp = api_client.delete("/api/v1/flagged-docs/999")
        assert resp.status_code == 404


# ───────────────────────────────────────────────────────────────────────
# Pydantic model validation tests
# ───────────────────────────────────────────────────────────────────────


class TestSaveFlaggedDocsRequestValidation:
    """Validate the Pydantic request model."""

    def test_valid_filename(self):
        from app.api import SaveFlaggedDocsRequest, FlaggedDocPayload
        req = SaveFlaggedDocsRequest(
            case_id="test-case",
            filename="my_batch-1.v2",
            query_text="some query",
            docs=[FlaggedDocPayload(source="a.pdf", content="x")],
        )
        assert req.filename == "my_batch-1.v2"

    def test_invalid_filename_path_traversal(self):
        from app.api import SaveFlaggedDocsRequest, FlaggedDocPayload
        with pytest.raises(Exception):  # ValidationError
            SaveFlaggedDocsRequest(
                case_id="test-case",
                filename="../etc/passwd",
                query_text="q",
                docs=[FlaggedDocPayload(source="a.pdf", content="x")],
            )

    def test_empty_docs_rejected(self):
        from app.api import SaveFlaggedDocsRequest
        with pytest.raises(Exception):
            SaveFlaggedDocsRequest(
                case_id="test-case",
                filename="batch1",
                query_text="q",
                docs=[],
            )
