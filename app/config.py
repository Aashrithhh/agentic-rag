"""Centralized configuration loaded from .env file (takes priority over system env)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Force .env values to override any pre-existing system environment variables
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ───────────────────────────────────────────
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ── Azure OpenAI (alternative) ───────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2025-01-01-preview"

    # ── Google Gemini (legacy) ───────────────────────────
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Cohere (embeddings) ──────────────────────────
    cohere_api_key: str = ""
    cohere_embed_model: str = "embed-english-v3.0"

    # ── Postgres + pgvector ──────────────────────────
    database_url: str = (
        "postgresql+psycopg://user:password@localhost:5432/audit_rag"
    )

    # ── External validation API ──────────────────────
    validation_api_base: str = "http://localhost:8000"

    # ── Attachment processing ─────────────────────────
    process_attachments: bool = True
    azure_whisper_deployment: str = "whisper"  # Azure OpenAI whisper model deployment

    # ── Azure Blob Storage (document source) ─────────
    azure_storage_connection_string: str = ""
    azure_storage_sas_url: str = ""              # Full SAS URL (container-level)
    azure_storage_container_name: str = "rag-documents"
    require_blob_source: bool = True             # Guardrail: reject local-file ingestion when True

    # ── RAG tuning ───────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200
    embedding_dim: int = 1024
    top_k: int = 6
    grader_threshold: float = 0.5
    max_rewrite_loops: int = 3

    # ── Structure-aware chunking ───────────────────
    use_structure_aware_chunking: bool = True

    # ── Metadata enrichment (LLM-based, off by default — expensive) ──
    enable_metadata_enrichment: bool = False
    enrichment_batch_size: int = 10

    # ── Legal Intake Validator ────────────────────────
    # MCP Ops diagnostics
    mcp_ops_enabled: bool = True
    mcp_ops_command: str = "python"
    mcp_ops_args: str = "-m app.mcp_ops.server"
    mcp_ops_timeout_seconds: int = 20
    postgres_container_name: str = ""
    docker_compose_service_name: str = ""
    docker_healthcheck_enabled: bool = True

    # ── Cache ────────────────────────────────────────────
    cache_enabled: bool = True
    cache_ttl_seconds: int = 0                      # 0 = keep until process restarts
    cache_max_entries_per_namespace: int = 2000
    cache_embeddings_enabled: bool = True
    cache_retrieval_enabled: bool = True
    cache_llm_enabled: bool = True

    max_clarification_rounds: int = 5
    legal_domain: str = "telangana"  # enables TS State laws

    # ── Logging ───────────────────────────────────────────
    log_format: str = "text"                             # "text" or "json"
    log_level: str = "INFO"

    # ── UI auth + input validation ────────────────────────
    ui_auth_enabled: bool = False
    ui_auth_password: str = ""
    ui_max_query_length: int = 2000

    # ── API security ──────────────────────────────────────
    api_keys: str = ""                             # comma-separated valid API keys (empty = dev mode)
    api_rate_limit: str = "30/minute"              # slowapi rate limit string
    cors_allowed_origins: str = ""                 # comma-separated origins (empty = deny all)

    # ── Reranker ──────────────────────────────────────────
    rerank_enabled: bool = True
    rerank_strategy: str = "cohere"                # "cohere" | "llm" | "cross_encoder"
    rerank_model: str = "rerank-english-v3.0"      # Cohere model or cross-encoder name
    rerank_top_n: int = 5                          # Number of docs after reranking

    # ── Intent routing ────────────────────────────────────
    intent_routing_enabled: bool = True

    # ── Citation validation & post-check ──────────────────
    enable_citation_validation: bool = True
    enable_answer_post_check: bool = True

    # ── Resilience / fallback ─────────────────────────────
    fallback_model: str = "gpt-4o-mini"            # Cheaper model when primary fails

    # ── PII redaction ─────────────────────────────────────
    pii_redaction_enabled: bool = True
    pii_redaction_mode: str = "redact"             # "redact" | "detect"

    # ── Human-in-the-loop ─────────────────────────────────
    hitl_enabled: bool = False                     # Off by default
    hitl_confidence_threshold: float = 0.65

    # ── Vault secrets provider ────────────────────────────
    vault_provider: str = "env"                    # "env" | "azure" | "hashicorp"
    vault_url: str = ""                            # HashiCorp / Azure Key Vault URL

    # ── Observability ─────────────────────────────────────
    prometheus_enabled: bool = True
    metrics_export_interval: int = 30              # seconds

    # ── Job queue ─────────────────────────────────────────
    job_queue_workers: int = 3

    def validate_secrets(self) -> list[str]:
        """Check that critical secrets are not default/placeholder values.

        Returns a list of warning messages.  Does NOT raise — caller
        decides severity.
        """
        warnings: list[str] = []
        placeholders = {"", "your-key-here", "changeme", "password", "test-key-not-real"}

        if self.database_url and "user:password@" in self.database_url:
            warnings.append("DATABASE_URL contains default credentials (user:password)")

        has_openai = self.openai_api_key and self.openai_api_key not in placeholders
        has_azure = self.azure_openai_api_key and self.azure_openai_api_key not in placeholders
        if not has_openai and not has_azure:
            warnings.append("No valid LLM API key configured (OPENAI_API_KEY or AZURE_OPENAI_API_KEY)")

        if self.cohere_api_key in placeholders:
            warnings.append("COHERE_API_KEY is missing or placeholder — embeddings will fail")

        return warnings


settings = Settings()  # type: ignore[call-arg]

# Emit config warnings once at import time
import logging as _logging
_startup_warnings = settings.validate_secrets()
if _startup_warnings:
    _config_logger = _logging.getLogger("app.config")
    for _w in _startup_warnings:
        _config_logger.warning("CONFIG: %s", _w)
