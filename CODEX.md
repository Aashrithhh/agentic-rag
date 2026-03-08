# CODEX.md

## Project Snapshot
- Name: Agentic RAG / Self-Correction Audit Agent
- Stack: Python 3.11+, FastAPI, Streamlit, LangGraph, PostgreSQL + pgvector
- Purpose: Case-isolated retrieval and answer generation over legal/forensic corpora (emails, PDFs, PST, attachments)

## Primary Entrypoints
- CLI query runner: `main.py`
- API server: `app/api.py`
- UI: `ui.py`
- Graph wiring: `app/graph.py`
- Ingestion pipeline: `app/ingest.py`
- Configuration: `app/config.py`

## Quickstart Commands
```bash
# Install deps
pip install -r requirements.txt

# Run API
uvicorn app.api:app --host 0.0.0.0 --port 8080

# Run UI
streamlit run ui.py

# Run tests
pytest tests/

# Run a focused test
pytest tests/test_graph.py -v
```

## Architecture Contract
- LangGraph flow:
  `intent_router -> retriever -> grader -> (rewriter loop) -> validator -> generator -> hallucination_guard`
- `app/state.py` (`AgentState`) is the source of truth for graph state fields.
- Add new nodes under `app/nodes/` and wire them in `app/graph.py`.
- Node outputs should mutate state via returned dicts (LangGraph pattern).

## Data and Ingestion Notes
- Ingestion supports many file types including `txt/pdf/docx/xlsx/xls/pptx/md/rtf/doc/xml/csv/tsv/log/json/html/images/eml/msg/pst/zip`.
- `ingest()` enforces blob-only ingestion when `require_blob_source=true`.
- Email/PST metadata and attachment handling are first-class; do not remove metadata fields casually.
- Retrieval cache is invalidated after successful ingestion; preserve this behavior on ingestion changes.

## API and Security Expectations
- `/api/v1/*` endpoints use `X-API-Key` auth when `API_KEYS` is configured.
- Respect existing rate limiting (`slowapi`), CORS restrictions, and error sanitization.
- Keep security headers and request ID tracing middleware intact.

## Configuration Rules
- Add/modify settings only in `app/config.py` (and `.env.example` when applicable).
- Prefer feature flags over hardcoded behavior.
- Avoid embedding secrets in code; rely on env/vault providers.

## Testing and Change Discipline
- Run `pytest tests/` for non-trivial changes.
- For targeted edits, run related tests at minimum (for example, `tests/test_api.py`, `tests/test_ingest_report.py`, `tests/test_graph.py`).
- Preserve backward-compatible API shapes unless explicitly asked to break them.

## Coding Conventions for This Repo
- Keep async patterns where I/O-bound paths already use async.
- Prefer editing existing modules over creating new abstractions.
- Do not introduce broad refactors unless requested.
- Keep logs useful for operations and audits; this project relies heavily on runtime observability.

## Operational Scripts
- Utility scripts live in `scripts/` (backups, migrations, eval, SLO gate).
- Alembic migrations are under `alembic/`.

## Working Preferences
- Do not auto-commit or push without explicit user confirmation.
- Default to minimal, safe, incremental edits.
- If a request touches ingestion/retrieval/validation correctness, include tests or explain why not run.
