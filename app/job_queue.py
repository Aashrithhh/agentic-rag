"""Async job queue for heavy operations (ingestion, evaluation, export).

Uses an in-process asyncio queue with worker tasks.  For production
scale-out, swap the in-process queue for Redis/Celery.

Features:
  1. Job submission (returns job_id immediately)
  2. Job status tracking (pending → running → completed / failed)
  3. Configurable concurrency (max_workers)
  4. Job results retrieval
  5. Priority lanes: urgent / normal / background
"""

from __future__ import annotations

import asyncio
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    URGENT = 0
    NORMAL = 5
    BACKGROUND = 10


@dataclass
class Job:
    """A unit of work in the queue."""
    id: str
    name: str
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    progress: float = 0.0        # 0.0 → 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "priority": self.priority.name.lower(),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
            "error": self.error,
            "metadata": self.metadata,
            # `result` excluded — callers fetch separately to avoid huge payloads
        }


class JobQueue:
    """In-process async job queue with priority support."""

    def __init__(self, max_workers: int = 3) -> None:
        self._max_workers = max_workers
        self._queue: asyncio.PriorityQueue[tuple[int, str]] = asyncio.PriorityQueue()
        self._jobs: dict[str, Job] = {}
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, Any]]] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._running = False

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Start worker tasks."""
        if self._running:
            return
        self._running = True
        for i in range(self._max_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(task)
        logger.info("Job queue started with %d workers", self._max_workers)

    async def stop(self) -> None:
        """Gracefully stop all workers."""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("Job queue stopped")

    # ── Registration ──────────────────────────────────────────────

    def register_handler(
        self,
        name: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register an async handler for a job type."""
        self._handlers[name] = handler

    # ── Submission ────────────────────────────────────────────────

    async def submit(
        self,
        name: str,
        *,
        priority: JobPriority = JobPriority.NORMAL,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Submit a job and return its ID immediately."""
        if name not in self._handlers:
            raise ValueError(f"No handler registered for job type '{name}'")

        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            name=name,
            priority=priority,
            metadata=metadata or {},
        )
        job.metadata["kwargs"] = kwargs
        self._jobs[job_id] = job
        await self._queue.put((priority.value, job_id))
        logger.info("Job submitted: %s (%s) priority=%s", job_id, name, priority.name)
        return job_id

    # ── Status & results ──────────────────────────────────────────

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def get_job_result(self, job_id: str) -> Any:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.result

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return [j.to_dict() for j in jobs[:limit]]

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job and job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            return True
        return False

    # ── Worker loop ───────────────────────────────────────────────

    async def _worker(self, name: str) -> None:
        logger.debug("Worker '%s' started", name)
        while self._running:
            try:
                priority_val, job_id = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if job is None or job.status == JobStatus.CANCELLED:
                self._queue.task_done()
                continue

            handler = self._handlers.get(job.name)
            if handler is None:
                job.status = JobStatus.FAILED
                job.error = f"No handler for '{job.name}'"
                self._queue.task_done()
                continue

            # Run the job
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc).isoformat()
            logger.info("[%s] Running job %s (%s)", name, job.id, job.name)

            try:
                kwargs = job.metadata.get("kwargs", {})
                job.result = await handler(job=job, **kwargs)
                job.status = JobStatus.COMPLETED
                job.progress = 1.0
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "[%s] Job %s failed: %s\n%s",
                    name, job.id, exc, traceback.format_exc(),
                )
            finally:
                job.completed_at = datetime.now(timezone.utc).isoformat()
                self._queue.task_done()

        logger.debug("Worker '%s' stopped", name)


# ── Module-level singleton ───────────────────────────────────────────

job_queue = JobQueue(max_workers=3)


# ── Pre-built job handlers ───────────────────────────────────────────

async def _ingest_job(*, job: Job, case_id: str, source_path: str, **kw: Any) -> dict[str, Any]:
    """Async ingestion job (wraps synchronous ingest)."""
    from app.ingest import ingest_case

    job.progress = 0.1
    result = await asyncio.to_thread(ingest_case, case_id, source_path)
    job.progress = 1.0
    return result


async def _evaluation_job(*, job: Job, case_id: str, eval_file: str, **kw: Any) -> dict[str, Any]:
    """Async evaluation job."""
    from app.evaluation import run_batch_evaluation

    job.progress = 0.1
    result = await asyncio.to_thread(run_batch_evaluation, case_id, eval_file)
    job.progress = 1.0
    return result


async def _retention_job(*, job: Job, **kw: Any) -> dict[str, Any]:
    """Apply retention policies across all cases."""
    from app.data_lifecycle import apply_retention_policies

    job.progress = 0.1
    result = await asyncio.to_thread(apply_retention_policies)
    job.progress = 1.0
    return result


def register_default_handlers(q: JobQueue | None = None) -> None:
    """Register built-in job handlers."""
    q = q or job_queue
    q.register_handler("ingest", _ingest_job)
    q.register_handler("evaluation", _evaluation_job)
    q.register_handler("retention_check", _retention_job)
