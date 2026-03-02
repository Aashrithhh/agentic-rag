"""Alembic environment — per-case database migration support.

Usage::

    alembic -x case_id=big-thorium upgrade head
    alembic -x case_id=big-thorium current

When ``case_id`` is passed via ``-x``, the migration targets that case's
isolated database (e.g. ``audit_rag_big_thorium``).  Without ``-x``, it
falls back to the default ``DATABASE_URL`` (useful for the template DB).
"""

from __future__ import annotations

import logging
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

# Import the project's shared metadata so autogenerate can diff
from app.db import metadata_obj

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata_obj
logger = logging.getLogger("alembic.env")


def _resolve_url() -> str:
    """Build the database URL, optionally scoped to a specific case."""
    from app.config import settings

    case_id = context.get_x_argument(as_dictionary=True).get("case_id")
    if case_id:
        safe_name = case_id.replace("-", "_")
        # Replace the database name in the URL
        base = settings.database_url.rsplit("/", 1)[0]
        url = f"{base}/audit_rag_{safe_name}"
        logger.info("Alembic targeting case DB: %s", url.split("@")[-1])
        return url
    return settings.database_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a live DB)."""
    url = _resolve_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (against a live DB)."""
    url = _resolve_url()
    connectable = create_engine(url)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
