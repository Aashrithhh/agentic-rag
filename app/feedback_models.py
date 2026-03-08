"""Data models for document relevance feedback and chat archival."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def deduplicate_docs_and_flags(self) -> "ChatArchive":
        """Deduplicate retrieved_docs by (source, page) and flags by
        (session_id, document_source, document_page), keeping last occurrence.
        Normalizes flag values to 'yes'/'no'/'unflagged'.
        """
        # Deduplicate retrieved_docs: prefer 'yes' > 'no' > 'unflagged'
        seen_docs: dict[tuple, "ArchivedDocument"] = {}
        flag_priority = {"yes": 2, "no": 1, "unflagged": 0}
        for doc in self.retrieved_docs:
            key = (doc.source, doc.page)
            existing = seen_docs.get(key)
            if existing is None or flag_priority.get(doc.flag, 0) > flag_priority.get(existing.flag, 0):
                seen_docs[key] = doc
        self.retrieved_docs = list(seen_docs.values())

        # Deduplicate flags by (session_id, document_source, document_page) — keep last
        seen_flags: dict[tuple, "RetrievalFeedback"] = {}
        for flag in self.flags:
            key = (flag.session_id, flag.document_source, flag.document_page)
            seen_flags[key] = flag  # last wins
        self.flags = list(seen_flags.values())

        return self
