"""Human-in-the-loop (HITL) workflow — review queue and approval gates.

Provides:
  1. Low-confidence answer flagging — answers below a threshold go to review queue
  2. Review queue — list, approve, reject, edit pending answers
  3. Feedback loop — reviewer edits feed back into eval / fine-tuning data
  4. Escalation — flag for domain expert review
  5. Metrics — approval rate, avg review latency, rejection reasons
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    ESCALATED = "escalated"


@dataclass
class ReviewItem:
    """An answer queued for human review."""
    id: str
    case_id: str
    query: str
    answer: str
    sources: list[str]
    confidence: float                       # 0-1 scale
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reviewed_at: str | None = None
    reviewer: str | None = None
    edited_answer: str | None = None
    rejection_reason: str | None = None
    feedback_notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "case_id": self.case_id,
            "query": self.query,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
            "reviewer": self.reviewer,
            "edited_answer": self.edited_answer,
            "rejection_reason": self.rejection_reason,
            "feedback_notes": self.feedback_notes,
        }


class HITLQueue:
    """In-memory human-in-the-loop review queue."""

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        persist_dir: str = "eval/hitl_reviews",
    ) -> None:
        self._threshold = confidence_threshold
        self._items: dict[str, ReviewItem] = {}
        self._persist_dir = Path(persist_dir)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

    # ── Flagging ─────────────────────────────────────────────────

    def should_flag(self, confidence: float) -> bool:
        """Check whether an answer should be flagged for review."""
        return confidence < self._threshold

    def flag_for_review(
        self,
        case_id: str,
        query: str,
        answer: str,
        sources: list[str],
        confidence: float,
        metadata: dict[str, Any] | None = None,
    ) -> ReviewItem:
        """Add an answer to the review queue."""
        item = ReviewItem(
            id=str(uuid.uuid4()),
            case_id=case_id,
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._items[item.id] = item
        logger.info(
            "Answer flagged for review: %s (confidence=%.2f, case=%s)",
            item.id, confidence, case_id,
        )

        try:
            from app.audit_log import audit_log
            audit_log.log_data_event(
                event_type="hitl_flagged",
                case_id=case_id,
                details={"review_id": item.id, "confidence": confidence},
            )
        except Exception:
            pass

        return item

    # ── Queue operations ─────────────────────────────────────────

    def list_pending(self, case_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List items awaiting review."""
        items = [
            i for i in self._items.values()
            if i.status == ReviewStatus.PENDING
        ]
        if case_id:
            items = [i for i in items if i.case_id == case_id]
        items.sort(key=lambda i: i.confidence)  # Lowest confidence first
        return [i.to_dict() for i in items[:limit]]

    def list_all(self, status: ReviewStatus | None = None, limit: int = 100) -> list[dict[str, Any]]:
        items = list(self._items.values())
        if status:
            items = [i for i in items if i.status == status]
        items.sort(key=lambda i: i.created_at, reverse=True)
        return [i.to_dict() for i in items[:limit]]

    def get_item(self, review_id: str) -> ReviewItem | None:
        return self._items.get(review_id)

    # ── Review actions ───────────────────────────────────────────

    def approve(self, review_id: str, reviewer: str, notes: str | None = None) -> ReviewItem | None:
        """Approve an answer as-is."""
        item = self._items.get(review_id)
        if not item or item.status != ReviewStatus.PENDING:
            return None
        item.status = ReviewStatus.APPROVED
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(timezone.utc).isoformat()
        item.feedback_notes = notes
        self._persist_review(item)
        logger.info("Review %s approved by %s", review_id, reviewer)
        return item

    def reject(
        self,
        review_id: str,
        reviewer: str,
        reason: str,
        notes: str | None = None,
    ) -> ReviewItem | None:
        """Reject an answer."""
        item = self._items.get(review_id)
        if not item or item.status != ReviewStatus.PENDING:
            return None
        item.status = ReviewStatus.REJECTED
        item.reviewer = reviewer
        item.rejection_reason = reason
        item.reviewed_at = datetime.now(timezone.utc).isoformat()
        item.feedback_notes = notes
        self._persist_review(item)
        logger.info("Review %s rejected by %s: %s", review_id, reviewer, reason)
        return item

    def edit_and_approve(
        self,
        review_id: str,
        reviewer: str,
        edited_answer: str,
        notes: str | None = None,
    ) -> ReviewItem | None:
        """Edit an answer and approve the edited version."""
        item = self._items.get(review_id)
        if not item or item.status != ReviewStatus.PENDING:
            return None
        item.status = ReviewStatus.EDITED
        item.reviewer = reviewer
        item.edited_answer = edited_answer
        item.reviewed_at = datetime.now(timezone.utc).isoformat()
        item.feedback_notes = notes
        self._persist_review(item)
        logger.info("Review %s edited and approved by %s", review_id, reviewer)
        return item

    def escalate(
        self,
        review_id: str,
        reviewer: str,
        notes: str | None = None,
    ) -> ReviewItem | None:
        """Escalate to domain expert."""
        item = self._items.get(review_id)
        if not item or item.status != ReviewStatus.PENDING:
            return None
        item.status = ReviewStatus.ESCALATED
        item.reviewer = reviewer
        item.reviewed_at = datetime.now(timezone.utc).isoformat()
        item.feedback_notes = notes
        self._persist_review(item)
        logger.info("Review %s escalated by %s", review_id, reviewer)
        return item

    # ── Feedback → training data ─────────────────────────────────

    def export_feedback(self, output_path: str | None = None) -> str:
        """Export reviewed items as JSONL for fine-tuning / eval enrichment."""
        output_path = output_path or str(
            self._persist_dir / f"feedback_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        )
        reviewed = [
            i for i in self._items.values()
            if i.status in (ReviewStatus.APPROVED, ReviewStatus.EDITED, ReviewStatus.REJECTED)
        ]
        with open(output_path, "w") as f:
            for item in reviewed:
                record = {
                    "query": item.query,
                    "answer": item.edited_answer or item.answer,
                    "approved": item.status != ReviewStatus.REJECTED,
                    "confidence": item.confidence,
                    "rejection_reason": item.rejection_reason,
                    "reviewer": item.reviewer,
                    "case_id": item.case_id,
                }
                f.write(json.dumps(record) + "\n")

        logger.info("Exported %d feedback records to %s", len(reviewed), output_path)
        return output_path

    # ── Metrics ───────────────────────────────────────────────────

    def metrics(self) -> dict[str, Any]:
        """Return HITL queue metrics."""
        items = list(self._items.values())
        total = len(items)
        pending = sum(1 for i in items if i.status == ReviewStatus.PENDING)
        approved = sum(1 for i in items if i.status == ReviewStatus.APPROVED)
        rejected = sum(1 for i in items if i.status == ReviewStatus.REJECTED)
        edited = sum(1 for i in items if i.status == ReviewStatus.EDITED)
        escalated = sum(1 for i in items if i.status == ReviewStatus.ESCALATED)

        # Average review latency
        reviewed = [
            i for i in items
            if i.reviewed_at and i.status != ReviewStatus.PENDING
        ]
        avg_latency_s = 0.0
        if reviewed:
            latencies = []
            for i in reviewed:
                try:
                    created = datetime.fromisoformat(i.created_at)
                    done = datetime.fromisoformat(i.reviewed_at)  # type: ignore[arg-type]
                    latencies.append((done - created).total_seconds())
                except Exception:
                    pass
            if latencies:
                avg_latency_s = sum(latencies) / len(latencies)

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "rejected": rejected,
            "edited": edited,
            "escalated": escalated,
            "approval_rate": (approved + edited) / max(total - pending, 1),
            "avg_review_latency_seconds": round(avg_latency_s, 1),
        }

    # ── Persistence ───────────────────────────────────────────────

    def _persist_review(self, item: ReviewItem) -> None:
        """Append completed review to disk."""
        path = self._persist_dir / "reviews.jsonl"
        try:
            with open(path, "a") as f:
                f.write(json.dumps(item.to_dict()) + "\n")
        except Exception as exc:
            logger.warning("Could not persist review %s: %s", item.id, exc)


# ── Module-level singleton ───────────────────────────────────────────

hitl_queue = HITLQueue()
