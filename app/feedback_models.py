"""Data models for document relevance feedback and chat archival."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class RetrievalFeedback(BaseModel):
    """Per-document relevance flag submitted by the user."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    case_id: str
    session_id: str
    query: str = Field(..., max_length=4000)
    document_source: str
    document_page: int | str | None = None
    flag: Literal["yes", "no"]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ArchivedDocument(BaseModel):
    """Lightweight snapshot of a retrieved document stored inside an archive."""

    source: str
    page: int | str | None = None
    content_preview: str = Field("", max_length=1000)
    flag: Literal["yes", "no", "unflagged"] = "unflagged"


class ChatArchive(BaseModel):
    """Complete record of a query session saved locally."""

    id: str = Field(default_factory=lambda: uuid4().hex[:16])
    case_id: str
    session_id: str
    query: str = Field(..., max_length=4000)
    answer: str
    retrieved_docs: list[ArchivedDocument] = Field(default_factory=list)
    flags: list[RetrievalFeedback] = Field(default_factory=list)
    validation_status: str = "pass"
    grader_score: str = ""
    loop_count: int = 0
    saved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    storage_type: Literal["local"] = "local"
