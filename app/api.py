"""FastAPI REST API for the Agentic RAG pipeline.

Provides programmatic access to the same pipeline the Streamlit UI uses.

Endpoints:
    GET  /health                  — load-balancer health check (no auth)
    GET  /api/v1/cases            — list available cases
    POST /api/v1/query            — run a query against a case
    GET  /metrics                 — operational metrics snapshot
    GET  /metrics/prometheus      — Prometheus text exposition
    GET  /api/v1/circuit-breakers — circuit breaker status
    GET  /api/v1/hitl/pending     — pending HITL reviews
    POST /api/v1/hitl/{id}/approve — approve a HITL review
    POST /api/v1/hitl/{id}/reject  — reject a HITL review
    GET  /api/v1/jobs             — list async jobs
    POST /api/v1/jobs/submit      — submit async job
    POST /api/v1/feedback/retrieval — submit per-doc relevance flags
    POST /api/v1/chats/save-local  — save chat + flags locally
    GET  /api/v1/chats/local       — list saved chats for a case

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.cases import list_cases
from app.config import settings
from app.db import init_db
from app.graph import compile_graph
from app.logging_config import setup_logging
from app.metrics import metrics
from app.security import require_api_key, sanitize_error
from app.state import AgentState

# ── Request ID context var (propagated to all log calls) ─────────────
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")

setup_logging()
logger = logging.getLogger(__name__)

# ── Rate limiter ──────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=[settings.api_rate_limit])

# ── Lifespan (start/stop job queue) ──────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Start the async job queue on startup, stop on shutdown."""
    from app.job_queue import job_queue, register_default_handlers
    register_default_handlers(job_queue)
    await job_queue.start()
    logger.info("Job queue started")
    yield
    await job_queue.stop()
    logger.info("Job queue stopped")

app = FastAPI(
    title="Agentic RAG API",
    version="1.0.0",
    description="REST interface for the Self-Correction Audit Agent.",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(
        content='{"detail":"Rate limit exceeded. Slow down."}',
        status_code=429,
        media_type="application/json",
    )


# ── CORS ──────────────────────────────────────────────────────────────

_origins = [o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["X-API-Key", "Content-Type"],
    )


# ── Request tracing + timing + security headers ─────────────────────


@app.middleware("http")
async def request_middleware(request: Request, call_next):
    # Assign a unique request ID
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request_id_var.set(rid)

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - t0

    # Metrics
    method = request.method
    path = request.url.path
    metrics.observe(f"http_request_seconds.{method}.{path}", elapsed)
    metrics.inc(f"http_status.{response.status_code}")

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Request-ID"] = rid
    return response


# ── Models ────────────────────────────────────────────────────────────


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=settings.ui_max_query_length)
    case_id: str = Field(..., min_length=1)
    # ── Optional email metadata filters ──────────────────
    email_from: str | None = Field(default=None, description="Filter by sender (partial match)")
    email_to: str | None = Field(default=None, description="Filter by recipient (partial match)")
    email_date_from: str | None = Field(default=None, description="Filter emails from this date (ISO format)")
    email_date_to: str | None = Field(default=None, description="Filter emails up to this date (ISO format)")
    source_file: str | None = Field(default=None, description="Filter by source filename (partial match)")


class QueryResponse(BaseModel):
    answer: str
    case_id: str
    grader_score: str = ""
    loop_count: int = 0
    validation_status: str = "pass"
    latency_seconds: float = 0.0


class CaseInfo(BaseModel):
    case_id: str
    display_name: str
    description: str
    doc_type: str


class HealthResponse(BaseModel):
    status: str = "ok"


# ── Endpoints ─────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Load-balancer health check — no auth required."""
    return HealthResponse()


@app.get("/metrics")
@limiter.limit(settings.api_rate_limit)
def get_metrics(request: Request, _key: str = Depends(require_api_key)):
    """Operational metrics snapshot — pipeline counters, histograms, cache stats."""
    from app.cache import cache
    return {
        "pipeline": metrics.snapshot(),
        "cache": cache.stats(),
    }


@app.get("/api/v1/cases", response_model=list[CaseInfo])
@limiter.limit(settings.api_rate_limit)
def get_cases(request: Request, _key: str = Depends(require_api_key)) -> list[CaseInfo]:
    """List all registered investigation cases."""
    return [
        CaseInfo(
            case_id=c.case_id,
            display_name=c.display_name,
            description=c.description,
            doc_type=c.doc_type,
        )
        for c in list_cases()
    ]


@app.post("/api/v1/query", response_model=QueryResponse)
@limiter.limit(settings.api_rate_limit)
def run_query(request: Request, req: QueryRequest, _key: str = Depends(require_api_key)) -> QueryResponse:
    """Run the RAG pipeline for a query against a specific case."""
    known_ids = [c.case_id for c in list_cases()]
    if req.case_id not in known_ids:
        raise HTTPException(status_code=404, detail=f"Unknown case_id: {req.case_id}")

    init_db(req.case_id)

    # ── Build metadata filter from email filter params ──────────
    email_filters: dict[str, str] = {}
    if req.email_from:
        email_filters["email_from"] = req.email_from
    if req.email_to:
        email_filters["email_to"] = req.email_to
    if req.email_date_from:
        email_filters["email_date_from"] = req.email_date_from
    if req.email_date_to:
        email_filters["email_date_to"] = req.email_date_to
    if req.source_file:
        email_filters["source_file"] = req.source_file

    initial_state: AgentState = {
        "query": req.query,
        "loop_count": 0,
        "case_id": req.case_id,
    }
    if email_filters:
        initial_state["metadata_filter"] = email_filters

    agent = compile_graph()

    t0 = time.time()
    try:
        result = agent.invoke(initial_state)
    except Exception as exc:
        logger.error("Pipeline error for case=%s: %s", req.case_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=sanitize_error(exc))
    elapsed = time.time() - t0

    # ── Audit log ──────────────────────────────────────────
    try:
        from app.audit_log import audit_log
        audit_log.log_query(
            case_id=req.case_id,
            query=req.query,
            user=request.headers.get("X-User", "api"),
        )
    except Exception:
        pass

    return QueryResponse(
        answer=result.get("answer", "(no answer generated)"),
        case_id=req.case_id,
        grader_score=result.get("grader_score", ""),
        loop_count=result.get("loop_count", 0),
        validation_status=result.get("validation_status", "pass"),
        latency_seconds=round(elapsed, 2),
    )


# ── Prometheus metrics endpoint ──────────────────────────────────────


@app.get("/metrics/prometheus")
@limiter.limit(settings.api_rate_limit)
def prometheus_metrics(request: Request, _key: str = Depends(require_api_key)):
    """Prometheus text exposition format."""
    try:
        from app.observability import metrics_to_prometheus
        body = metrics_to_prometheus()
        return Response(content=body, media_type="text/plain; version=0.0.4")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ── Circuit breaker status ───────────────────────────────────────────


@app.get("/api/v1/circuit-breakers")
@limiter.limit(settings.api_rate_limit)
def circuit_breaker_status(request: Request, _key: str = Depends(require_api_key)):
    """Return status of all circuit breakers."""
    from app.resilience import all_circuit_breaker_statuses
    return all_circuit_breaker_statuses()


# ── HITL endpoints ───────────────────────────────────────────────────


class HITLActionRequest(BaseModel):
    reviewer: str
    reason: str = ""
    notes: str = ""
    edited_answer: str = ""


@app.get("/api/v1/hitl/pending")
@limiter.limit(settings.api_rate_limit)
def hitl_pending(request: Request, case_id: str | None = None, _key: str = Depends(require_api_key)):
    """List pending HITL reviews."""
    from app.hitl import hitl_queue
    return hitl_queue.list_pending(case_id=case_id)


@app.post("/api/v1/hitl/{review_id}/approve")
@limiter.limit(settings.api_rate_limit)
def hitl_approve(request: Request, review_id: str, body: HITLActionRequest, _key: str = Depends(require_api_key)):
    from app.hitl import hitl_queue
    item = hitl_queue.approve(review_id, reviewer=body.reviewer, notes=body.notes)
    if not item:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return item.to_dict()


@app.post("/api/v1/hitl/{review_id}/reject")
@limiter.limit(settings.api_rate_limit)
def hitl_reject(request: Request, review_id: str, body: HITLActionRequest, _key: str = Depends(require_api_key)):
    from app.hitl import hitl_queue
    item = hitl_queue.reject(review_id, reviewer=body.reviewer, reason=body.reason, notes=body.notes)
    if not item:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return item.to_dict()


@app.post("/api/v1/hitl/{review_id}/edit")
@limiter.limit(settings.api_rate_limit)
def hitl_edit(request: Request, review_id: str, body: HITLActionRequest, _key: str = Depends(require_api_key)):
    from app.hitl import hitl_queue
    item = hitl_queue.edit_and_approve(review_id, reviewer=body.reviewer, edited_answer=body.edited_answer, notes=body.notes)
    if not item:
        raise HTTPException(status_code=404, detail="Review not found or already processed")
    return item.to_dict()


@app.get("/api/v1/hitl/metrics")
@limiter.limit(settings.api_rate_limit)
def hitl_metrics(request: Request, _key: str = Depends(require_api_key)):
    from app.hitl import hitl_queue
    return hitl_queue.metrics()


# ── Job queue endpoints ──────────────────────────────────────────────


class JobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="Job type: ingest, evaluation, retention_check")
    case_id: str = ""
    source_path: str = ""
    eval_file: str = ""


@app.get("/api/v1/jobs")
@limiter.limit(settings.api_rate_limit)
def list_jobs(request: Request, status: str | None = None, _key: str = Depends(require_api_key)):
    from app.job_queue import job_queue, JobStatus
    s = JobStatus(status) if status else None
    return job_queue.list_jobs(status=s)


@app.get("/api/v1/jobs/{job_id}")
@limiter.limit(settings.api_rate_limit)
def get_job_status(request: Request, job_id: str, _key: str = Depends(require_api_key)):
    from app.job_queue import job_queue
    result = job_queue.get_job_status(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Job not found")
    return result

@app.post("/api/v1/jobs/submit")
@limiter.limit(settings.api_rate_limit)
async def submit_job(request: Request, body: JobSubmitRequest, _key: str = Depends(require_api_key)):
    """Submit a background job (ingest, evaluation, retention_check).

    Returns immediately with a ``job_id`` that can be polled via
    ``GET /api/v1/jobs/{job_id}``.
    """
    from app.job_queue import job_queue, JobPriority

    valid_types = {"ingest", "evaluation", "retention_check"}
    if body.job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job_type '{body.job_type}'. Must be one of: {', '.join(sorted(valid_types))}",
        )

    kwargs: dict[str, str] = {}
    if body.job_type == "ingest":
        if not body.case_id or not body.source_path:
            raise HTTPException(status_code=400, detail="'case_id' and 'source_path' required for ingest jobs")
        kwargs = {"case_id": body.case_id, "source_path": body.source_path}
    elif body.job_type == "evaluation":
        if not body.case_id or not body.eval_file:
            raise HTTPException(status_code=400, detail="'case_id' and 'eval_file' required for evaluation jobs")
        kwargs = {"case_id": body.case_id, "eval_file": body.eval_file}

    job_id = await job_queue.submit(
        name=body.job_type,
        priority=JobPriority.NORMAL,
        metadata={"submitted_via": "api"},
        **kwargs,
    )

    logger.info("Job submitted via API: %s (type=%s, case=%s)", job_id, body.job_type, body.case_id)
    return {"job_id": job_id, "job_type": body.job_type, "status": "pending"}


@app.post("/api/v1/jobs/{job_id}/cancel")
@limiter.limit(settings.api_rate_limit)
def cancel_job(request: Request, job_id: str, _key: str = Depends(require_api_key)):
    """Cancel a pending job."""
    from app.job_queue import job_queue
    success = job_queue.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job not found or already running/completed")
    return {"job_id": job_id, "status": "cancelled"}

# ── Data lifecycle endpoints ─────────────────────────────────────────


@app.get("/api/v1/data/{case_id}/inventory")
@limiter.limit(settings.api_rate_limit)
def data_inventory(request: Request, case_id: str, _key: str = Depends(require_api_key)):
    """Get data inventory and retention status for a case."""
    from app.data_lifecycle import get_data_inventory
    inv = get_data_inventory(case_id)
    return {
        "case_id": inv.case_id,
        "database": inv.database_name,
        "total_chunks": inv.total_chunks,
        "total_sources": inv.total_sources,
        "sources": inv.sources,
        "oldest_chunk_date": inv.oldest_chunk_date,
        "newest_chunk_date": inv.newest_chunk_date,
        "data_size_bytes": inv.data_size_bytes,
        "days_until_expiry": inv.days_until_expiry,
        "is_expired": inv.is_expired,
    }


# ── SLO report endpoint ─────────────────────────────────────────────


@app.get("/api/v1/slo")
@limiter.limit(settings.api_rate_limit)
def slo_report(request: Request, _key: str = Depends(require_api_key)):
    """Run SLO check and return report."""
    from app.slo import SLOThresholds, full_slo_check
    report = full_slo_check(SLOThresholds())
    return {
        "passed": report.passed,
        "violations": [
            {"metric": v.metric, "threshold": v.threshold, "actual": v.actual, "severity": v.severity}
            for v in report.violations
        ],
    }


# ── Feedback & Chat archive endpoints ────────────────────────────────


class FeedbackFlag(BaseModel):
    document_source: str
    document_page: int | str | None = None
    flag: str = Field(..., pattern=r"^(yes|no)$")


class FeedbackRequest(BaseModel):
    case_id: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=200)
    query: str = Field(..., min_length=1, max_length=4000)
    flags: list[FeedbackFlag] = Field(..., min_length=1)


class SaveChatRequest(BaseModel):
    case_id: str = Field(..., min_length=1, max_length=200)
    session_id: str = Field(..., min_length=1, max_length=200)
    query: str = Field(..., min_length=1, max_length=4000)
    answer: str
    retrieved_docs: list[dict] = Field(default_factory=list)
    flags: list[dict] = Field(default_factory=list)
    validation_status: str = "pass"
    grader_score: str = ""
    loop_count: int = 0


@app.post("/api/v1/feedback/retrieval")
@limiter.limit(settings.api_rate_limit)
def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    _key: str = Depends(require_api_key),
):
    """Submit per-document yes/no relevance flags for a query session."""
    from app.audit_log import audit_log
    from app.feedback_models import RetrievalFeedback

    feedback_records = [
        RetrievalFeedback(
            case_id=body.case_id,
            session_id=body.session_id,
            query=body.query,
            document_source=f.document_source,
            document_page=f.document_page,
            flag=f.flag,
        )
        for f in body.flags
    ]

    audit_log.log_feedback_submitted(
        case_id=body.case_id,
        session_id=body.session_id,
        num_flags=len(feedback_records),
    )

    return {
        "status": "ok",
        "feedback_count": len(feedback_records),
        "feedback_ids": [fb.id for fb in feedback_records],
    }


@app.post("/api/v1/chats/save-local")
@limiter.limit(settings.api_rate_limit)
def save_chat_local_endpoint(
    request: Request,
    body: SaveChatRequest,
    _key: str = Depends(require_api_key),
):
    """Save a complete chat session (query, answer, docs, flags) locally."""
    from app.audit_log import audit_log
    from app.chat_storage import save_chat_local
    from app.feedback_models import ArchivedDocument, ChatArchive, RetrievalFeedback

    archived_docs = [
        ArchivedDocument(
            source=d.get("source", "unknown"),
            page=d.get("page"),
            content_preview=(d.get("content_preview") or d.get("content", ""))[:1000],
            flag=d.get("flag", "unflagged"),
        )
        for d in body.retrieved_docs
    ]

    flag_records = [
        RetrievalFeedback(
            case_id=body.case_id,
            session_id=body.session_id,
            query=body.query,
            document_source=f.get("document_source", ""),
            document_page=f.get("document_page"),
            flag=f.get("flag", "no"),
        )
        for f in body.flags
    ]

    archive = ChatArchive(
        case_id=body.case_id,
        session_id=body.session_id,
        query=body.query,
        answer=body.answer,
        retrieved_docs=archived_docs,
        flags=flag_records,
        validation_status=body.validation_status,
        grader_score=body.grader_score,
        loop_count=body.loop_count,
    )

    try:
        chat_id = save_chat_local(archive)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    audit_log.log_chat_saved(case_id=body.case_id, chat_id=chat_id)

    return {"status": "ok", "chat_id": chat_id}


@app.get("/api/v1/chats/local")
@limiter.limit(settings.api_rate_limit)
def list_local_chats(
    request: Request,
    case_id: str,
    _key: str = Depends(require_api_key),
):
    """List saved chat archives for a given case."""
    from app.chat_storage import list_saved_chats

    chats = list_saved_chats(case_id)
    return {"case_id": case_id, "chats": chats}
