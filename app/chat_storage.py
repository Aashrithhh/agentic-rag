"""Local JSON file storage for chat archives and retrieval feedback."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.config import settings
from app.feedback_models import ChatArchive

logger = logging.getLogger(__name__)


def _archive_dir(case_id: str) -> Path:
    """Return (and create) the archive directory for a case."""
    base = Path(settings.chat_local_storage_dir)
    case_dir = base / case_id.replace("/", "_").replace("\\", "_")
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

    out_path = case_dir / f"{payload.id}.json"
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


def get_saved_chat(case_id: str, chat_id: str) -> ChatArchive | None:
    """Load a single archived chat by ID. Returns None if not found."""
    case_dir = _archive_dir(case_id)
    path = case_dir / f"{chat_id}.json"

    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ChatArchive.model_validate(data)
    except Exception:
        logger.warning("Failed to parse archive: %s", path)
        return None
