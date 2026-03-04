"""Unit + integration tests for document feedback and chat archive feature."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.feedback_models import ArchivedDocument, ChatArchive, RetrievalFeedback

pytestmark = pytest.mark.integration


# ═══════════════════════════════════════════════════════════════════════
# Unit tests — Pydantic models
# ═══════════════════════════════════════════════════════════════════════


class TestRetrievalFeedbackModel:
    def test_valid_yes_flag(self):
        fb = RetrievalFeedback(
            case_id="case-1",
            session_id="sess-1",
            query="What happened?",
            document_source="doc.pdf",
            flag="yes",
        )
        assert fb.flag == "yes"
        assert fb.id  # auto-generated
        assert fb.timestamp  # auto-generated

    def test_valid_no_flag(self):
        fb = RetrievalFeedback(
            case_id="case-1",
            session_id="sess-1",
            query="What happened?",
            document_source="doc.pdf",
            flag="no",
        )
        assert fb.flag == "no"

    def test_invalid_flag_rejected(self):
        with pytest.raises(Exception):
            RetrievalFeedback(
                case_id="case-1",
                session_id="sess-1",
                query="Q",
                document_source="doc.pdf",
                flag="maybe",
            )

    def test_page_optional(self):
        fb = RetrievalFeedback(
            case_id="c", session_id="s", query="Q",
            document_source="src", flag="yes",
        )
        assert fb.document_page is None


class TestChatArchiveModel:
    def test_creates_with_defaults(self):
        archive = ChatArchive(
            case_id="case-1",
            session_id="sess-1",
            query="What happened?",
            answer="Something happened.",
        )
        assert archive.id
        assert archive.saved_at
        assert archive.storage_type == "local"
        assert archive.retrieved_docs == []
        assert archive.flags == []

    def test_with_docs_and_flags(self):
        doc = ArchivedDocument(source="a.pdf", page=1, content_preview="text", flag="yes")
        fb = RetrievalFeedback(
            case_id="c1", session_id="s1", query="Q",
            document_source="a.pdf", flag="yes",
        )
        archive = ChatArchive(
            case_id="c1", session_id="s1",
            query="Q", answer="A",
            retrieved_docs=[doc],
            flags=[fb],
        )
        assert len(archive.retrieved_docs) == 1
        assert len(archive.flags) == 1

    def test_roundtrip_json(self):
        archive = ChatArchive(
            case_id="c1", session_id="s1",
            query="Q", answer="A",
        )
        data = json.loads(archive.model_dump_json())
        restored = ChatArchive.model_validate(data)
        assert restored.id == archive.id
        assert restored.query == archive.query


class TestArchivedDocumentModel:
    def test_defaults(self):
        doc = ArchivedDocument(source="x.pdf")
        assert doc.flag == "unflagged"
        assert doc.content_preview == ""

    def test_valid_flags(self):
        for f in ("yes", "no", "unflagged"):
            doc = ArchivedDocument(source="x.pdf", flag=f)
            assert doc.flag == f


# ═══════════════════════════════════════════════════════════════════════
# Unit tests — chat_storage
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_archive_dir(tmp_path):
    """Provide a temporary storage dir and patch settings to use it."""
    with patch("app.chat_storage.settings") as mock_settings:
        mock_settings.chat_local_storage_dir = str(tmp_path / "archive")
        mock_settings.max_chat_archive_files = 5
        mock_settings.max_chat_save_size_mb = 10.0
        yield tmp_path / "archive"


def _make_archive(case_id: str = "case-1", query: str = "Q") -> ChatArchive:
    return ChatArchive(
        case_id=case_id, session_id="s1", query=query, answer="A",
    )


class TestSaveChatLocal:
    def test_save_creates_file(self, tmp_archive_dir):
        from app.chat_storage import save_chat_local

        archive = _make_archive()
        chat_id = save_chat_local(archive)

        saved_path = tmp_archive_dir / "case-1" / f"{chat_id}.json"
        assert saved_path.exists()

        data = json.loads(saved_path.read_text(encoding="utf-8"))
        assert data["query"] == "Q"
        assert data["answer"] == "A"

    def test_file_count_guardrail(self, tmp_archive_dir):
        from app.chat_storage import save_chat_local

        for i in range(5):
            save_chat_local(_make_archive(query=f"Q{i}"))

        with pytest.raises(ValueError, match="Archive limit reached"):
            save_chat_local(_make_archive(query="Q-overflow"))


class TestListSavedChats:
    def test_empty_case(self, tmp_archive_dir):
        from app.chat_storage import list_saved_chats

        result = list_saved_chats("nonexistent-case")
        assert result == []

    def test_returns_saved_entries(self, tmp_archive_dir):
        from app.chat_storage import list_saved_chats, save_chat_local

        save_chat_local(_make_archive(query="First question"))
        save_chat_local(_make_archive(query="Second question"))

        result = list_saved_chats("case-1")
        assert len(result) == 2
        assert all("id" in e and "query_preview" in e and "saved_at" in e for e in result)


class TestGetSavedChat:
    def test_returns_none_for_missing(self, tmp_archive_dir):
        from app.chat_storage import get_saved_chat

        assert get_saved_chat("case-1", "nonexistent") is None

    def test_roundtrip(self, tmp_archive_dir):
        from app.chat_storage import get_saved_chat, save_chat_local

        archive = _make_archive(query="Roundtrip test")
        chat_id = save_chat_local(archive)

        loaded = get_saved_chat("case-1", chat_id)
        assert loaded is not None
        assert loaded.query == "Roundtrip test"
        assert loaded.case_id == "case-1"


# ═══════════════════════════════════════════════════════════════════════
# Integration tests — API endpoints
# ═══════════════════════════════════════════════════════════════════════


@pytest.fixture
def client():
    from app.api import app
    return TestClient(app)


class TestFeedbackEndpoint:
    def test_submit_valid_feedback(self, client):
        resp = client.post("/api/v1/feedback/retrieval", json={
            "case_id": "case-1",
            "session_id": "sess-1",
            "query": "What happened?",
            "flags": [
                {"document_source": "doc.pdf", "document_page": 1, "flag": "yes"},
                {"document_source": "report.pdf", "flag": "no"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["feedback_count"] == 2

    def test_reject_invalid_flag_value(self, client):
        resp = client.post("/api/v1/feedback/retrieval", json={
            "case_id": "case-1",
            "session_id": "sess-1",
            "query": "Q",
            "flags": [
                {"document_source": "doc.pdf", "flag": "maybe"},
            ],
        })
        assert resp.status_code == 422

    def test_reject_empty_flags(self, client):
        resp = client.post("/api/v1/feedback/retrieval", json={
            "case_id": "case-1",
            "session_id": "sess-1",
            "query": "Q",
            "flags": [],
        })
        assert resp.status_code == 422

    def test_reject_missing_case_id(self, client):
        resp = client.post("/api/v1/feedback/retrieval", json={
            "session_id": "s",
            "query": "Q",
            "flags": [{"document_source": "x", "flag": "yes"}],
        })
        assert resp.status_code == 422


class TestSaveChatEndpoint:
    def test_save_valid_chat(self, client, tmp_archive_dir):
        resp = client.post("/api/v1/chats/save-local", json={
            "case_id": "case-1",
            "session_id": "sess-1",
            "query": "What happened?",
            "answer": "Something happened.",
            "retrieved_docs": [
                {"source": "doc.pdf", "page": 1, "content_preview": "text", "flag": "yes"},
            ],
            "flags": [
                {"document_source": "doc.pdf", "document_page": 1, "flag": "yes"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "chat_id" in data


class TestListChatsEndpoint:
    def test_list_empty(self, client):
        resp = client.get("/api/v1/chats/local", params={"case_id": "empty-case"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chats"] == []
