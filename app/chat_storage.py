"""Local JSON file storage for chat archives and retrieval feedback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app.config import settings
from app.feedback_models import ChatArchive, RetrievalFeedback

logger = logging.getLogger(__name__)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _sanitize_id(value: str, label: str = "id") -> str:
    """Validate that *value* is a safe filename component.

    Allows only alphanumerics, hyphens, and underscores.
    Raises ValueError for empty, '.', '..', absolute, or other unsafe values.
    """
    if not value or value in (".", ".."):
        raise ValueError(f"Invalid {label}: {value!r}")
    if not _SAFE_ID_RE.match(value):
        raise ValueError(
            f"Invalid {label}: {value!r} — only alphanumerics, hyphens, "
            f"and underscores are allowed."
        )
    return value


def _archive_dir(case_id: str) -> Path:
    """Return (and create) the archive directory for a case."""
    safe_case = _sanitize_id(case_id, "case_id")
    base = Path(settings.chat_local_storage_dir).resolve()
    case_dir = (base / safe_case).resolve()
    if not str(case_dir).startswith(str(base)):
        raise ValueError(f"Invalid case_id would escape archive root: {case_id!r}")
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def _count_files(directory: Path) -> int:
    return sum(1 for f in directory.iterdir() if f.suffix == ".json")


def _dir_size_mb(directory: Path) -> float:
    return sum(f.stat().st_size for f in directory.rglob("*.json")) / (1024 * 1024)


def save_chat_local(payload: ChatArchive) -> str:
    """Persist a ChatArchive as a JSON file.

    Returns the chat_id on success.
    Raises ValueError if file-count or size guardrails are exceeded.
    """
    case_dir = _archive_dir(payload.case_id)

    if _count_files(case_dir) >= settings.max_chat_archive_files:
        raise ValueError(
            f"Archive limit reached ({settings.max_chat_archive_files} files) "
            f"for case '{payload.case_id}'. Delete old archives before saving new ones."
        )

    if _dir_size_mb(case_dir) >= settings.max_chat_save_size_mb:
        raise ValueError(
            f"Archive size limit reached ({settings.max_chat_save_size_mb} MB) "
            f"for case '{payload.case_id}'."
        )

    safe_id = _sanitize_id(payload.id, "chat_id")
    out_path = case_dir / f"{safe_id}.json"
    out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Chat archived: %s -> %s", payload.id, out_path)
    return payload.id


def list_saved_chats(case_id: str) -> list[dict]:
    """Return lightweight summaries of all archived chats for a case.

    Each entry: ``{id, query_preview, saved_at}``.
    """
    case_dir = _archive_dir(case_id)
    results: list[dict] = []

    for path in sorted(case_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append({
                "id": data.get("id", path.stem),
                "query_preview": (data.get("query") or "")[:120],
                "saved_at": data.get("saved_at", ""),
            })
        except Exception:
            logger.warning("Skipping corrupt archive file: %s", path)

    return results


def save_feedback_local(records: list[RetrievalFeedback]) -> list[str]:
    """Persist standalone retrieval feedback records as individual JSON files.

    Returns the list of saved feedback IDs.
    """
    if not records:
        return []
    # Cache resolved case dirs to avoid repeated validation/mkdir for the
    # same case_id while still handling mixed-case batches correctly.
    _case_dirs: dict[str, Path] = {}
    saved_ids: list[str] = []
    for fb in records:
        if fb.case_id not in _case_dirs:
            fb_dir = _archive_dir(fb.case_id) / "feedback"
            fb_dir.mkdir(parents=True, exist_ok=True)
            _case_dirs[fb.case_id] = fb_dir
        case_dir = _case_dirs[fb.case_id]
        safe_id = _sanitize_id(fb.id, "feedback_id")
        out_path = case_dir / f"{safe_id}.json"
        out_path.write_text(fb.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Feedback saved: %s -> %s", fb.id, out_path)
        saved_ids.append(fb.id)
    return saved_ids


def get_saved_chat(case_id: str, chat_id: str) -> ChatArchive | None:
    """Load a single archived chat by ID. Returns None if not found."""
    try:
        safe_chat_id = _sanitize_id(chat_id, "chat_id")
    except ValueError:
        logger.warning("Invalid chat_id rejected: %s", chat_id)
        return None
    case_dir = _archive_dir(case_id)
    path = case_dir / f"{safe_chat_id}.json"

    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatArchive.model_validate(data)
    except Exception:
        logger.warning("Failed to parse archive: %s", path)
        return None
