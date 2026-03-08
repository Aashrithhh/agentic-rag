# Agentic RAG — Self-Correction Audit Agent

## Project Overview
A production-grade RAG pipeline built on LangGraph for legal/forensic document investigation.
Primary use case: querying and auditing case documents (emails, PDFs, PSTs, attachments).

## Key Commands

```bash
# Start API server
uvicorn app.api:app --host 0.0.0.0 --port 8080

# Start Streamlit UI
streamlit run ui.py

# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_graph.py -v

# Install dependencies (use venv)
pip install -r requirements.txt
```

## Architecture

### Pipeline Flow (LangGraph)
```
START → Intent Router → Retriever → Grader → [Rewriter (loop)] → Validator → Generator → Hallucination Guard → END
```

### Key Files
| File | Purpose |
|---|---|
| [app/graph.py](app/graph.py) | LangGraph graph definition — all nodes and edges |
| [app/state.py](app/state.py) | `AgentState` TypedDict — the state flowing through all nodes |
| [app/llm.py](app/llm.py) | LLM provider factory — `get_chat_llm()` returns OpenAI or Ollama |
| [app/api.py](app/api.py) | FastAPI REST interface |
| [ui.py](ui.py) | Streamlit UI |
| [app/config.py](app/config.py) | All settings via `pydantic-settings` (loaded from `.env`) |
| [app/ingest.py](app/ingest.py) | Document ingestion pipeline |
| [app/db.py](app/db.py) | PostgreSQL + pgvector setup |
| [app/cases.py](app/cases.py) | Case registry — list of investigation cases |

### Nodes (`app/nodes/`)
| Node | File | Role |
|---|---|---|
| Intent Router | `app/graph.py` (inline) | Classifies query intent |
| Retriever | `app/nodes/retriever.py` | Hybrid HNSW + GIN search + reranking + neighbor stitching |
| Grader | `app/nodes/grader.py` | Structured yes/no relevance scoring |
| Rewriter | `app/nodes/rewriter.py` | Query rewriting on failed grades |
| Validator | `app/nodes/validator.py` | Fact-checks claims, critical fact protection |
| Generator | `app/nodes/generator.py` | Answer generation with citation validation |
| Hallucination Guard | `app/nodes/hallucination_guard.py` | Risk scoring: pass / warn / block |

### Supporting Modules
- `app/chunking.py` — email-aware + structure-aware chunking
- `app/reranker.py` — Cohere / cross-encoder / LLM reranking
- `app/intent_router.py` — intent classification (fact_lookup, summary, timeline, comparison, exploratory)
- `app/hallucination_guard.py` — hallucination scoring logic
- `app/hitl.py` — human-in-the-loop review queue
- `app/job_queue.py` — async background job queue
- `app/audit_log.py` — audit logging to `logs/audit/`
- `app/resilience.py` — circuit breakers
- `app/cache.py` — in-process LRU cache (embeddings, retrieval, LLM)
- `app/slo.py` — SLO thresholds and reporting
- `app/pii_redaction.py` — PII detection/redaction (off by default)
- `app/blob_storage.py` — Azure Blob Storage source
- `app/vault.py` — secrets provider (env / Azure Key Vault / HashiCorp Vault)
- `app/data_lifecycle.py` — retention and data inventory

## Configuration (`.env`)

All settings live in `app/config.py` as a `pydantic-settings` `Settings` class.

### LLM Provider
Controlled by `LLM_PROVIDER` (`"openai"` default | `"ollama"`). All nodes call `get_chat_llm()` from `app/llm.py` — never instantiate LLMs directly.

| Provider | Key settings |
|---|---|
| `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL=gpt-4o` |
| `ollama` | `OLLAMA_BASE_URL=http://localhost:11434`, `OLLAMA_MODEL=phi3:mini`, `OLLAMA_NUM_CTX=4096` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` |

Ollama v0.17.6 installed. Models: `phi3:mini` (2.2 GB, active), `llama3.2:3b` (2.0 GB, backup). CPU-only, no GPU. OpenAI key temporarily expired — Ollama is the active LLM provider until renewed.

Embeddings: Cohere (`COHERE_API_KEY`, `embed-english-v3.0`)
Database: PostgreSQL + pgvector (`DATABASE_URL`)

Key feature flags (all in `.env` or `config.py`):
- `RERANK_ENABLED` / `RERANK_STRATEGY` — reranking (cohere | llm | cross_encoder)
- `INTENT_ROUTING_ENABLED` — intent classification
- `HALLUCINATION_GUARD_ENABLED` — hallucination risk scoring
- `HITL_ENABLED` — human-in-the-loop reviews
- `NEIGHBOR_STITCHING` — fetch adjacent chunks at retrieval
- `CRITICAL_FACT_PROTECTION` — require 2+ spans for numeric/date claims
- `PII_REDACTION_ENABLED` — PII redaction (off by default — investigation data)

## API Auth
All `/api/v1/*` endpoints require `X-API-Key` header.
Set `API_KEYS=key1,key2` in `.env`. Empty = dev mode (no auth).

## Conventions
- All modules use `async/await` where I/O-bound; sync elsewhere
- New graph nodes go in `app/nodes/` and must be wired into `app/graph.py`
- State mutations happen only through node return dicts (LangGraph pattern)
- `AgentState` in `app/state.py` is the single source of truth for state fields — add new fields there
- Settings are never hardcoded — always add to `Settings` in `app/config.py`
- **Never instantiate LLMs directly** — always use `get_chat_llm()` from `app/llm.py`
- Tests live in `tests/` — run with `pytest`

## Workflow Preferences
- Never auto-commit or push without confirmation
- Always run `pytest tests/` before suggesting a commit
- Prefer editing existing files over creating new ones
- Do not add docstrings or type annotations to code that wasn't changed
