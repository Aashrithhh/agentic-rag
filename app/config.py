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

    # ── Azure OpenAI ─────────────────────────────────
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o"
    azure_openai_api_version: str = "2025-01-01-preview"

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

    # ── RAG tuning ───────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200
    embedding_dim: int = 1024
    top_k: int = 6
    grader_threshold: float = 0.5
    max_rewrite_loops: int = 3

    # ── Legal Intake Validator ────────────────────────
    max_clarification_rounds: int = 5
    legal_domain: str = "telangana"  # enables TS State laws


settings = Settings()  # type: ignore[call-arg]
