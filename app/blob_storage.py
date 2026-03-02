"""Azure Blob Storage client for document ingestion.

Provides a file-source abstraction: iterates blobs in a container prefix
and yields (filename, bytes) tuples that the ingest loaders consume.
When no Azure config is present, falls back to local filesystem reading.
"""

from __future__ import annotations

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Iterator

from app.config import settings

logger = logging.getLogger(__name__)

_client: "BlobStorageClient | None" = None

# Download configuration
MAX_DOWNLOAD_RETRIES = 3
RETRY_DELAY_SECONDS = 2
CONNECTION_TIMEOUT = 60  # seconds
READ_TIMEOUT = 300  # 5 minutes for large files


def is_blob_mode() -> bool:
    """Return True when Azure Blob Storage is configured (connection string or SAS URL)."""
    return bool(settings.azure_storage_connection_string or settings.azure_storage_sas_url)


def get_blob_client() -> "BlobStorageClient":
    """Module-level singleton for the blob client."""
    global _client
    if _client is None:
        _client = BlobStorageClient()
    return _client


class BlobStorageClient:
    """Lazy-init Azure Blob client. Only imports azure SDK when actually used."""

    def __init__(self) -> None:
        self._container_client = None

    def _ensure_client(self) -> None:
        if self._container_client is not None:
            return

        # Configure retry and timeout
        from azure.core.pipeline.transport import RequestsTransport
        from azure.storage.blob._shared.policies import StorageRetryPolicy

        transport = RequestsTransport(
            connection_timeout=CONNECTION_TIMEOUT,
            read_timeout=READ_TIMEOUT,
        )

        if settings.azure_storage_sas_url:
            # SAS URL authentication (container-level SAS token)
            from azure.storage.blob import ContainerClient

            self._container_client = ContainerClient.from_container_url(
                settings.azure_storage_sas_url,
                transport=transport,
            )
            logger.info(
                "Connected to Azure Blob Storage via SAS URL: %s",
                settings.azure_storage_sas_url.split("?")[0],  # log URL without token
            )
        elif settings.azure_storage_connection_string:
            # Connection string authentication
            from azure.storage.blob import BlobServiceClient

            service = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string,
                transport=transport,
            )
            self._container_client = service.get_container_client(
                settings.azure_storage_container_name
            )
            logger.info(
                "Connected to Azure Blob Storage container: %s",
                settings.azure_storage_container_name,
            )
        else:
            raise RuntimeError("No Azure Blob Storage credentials configured")

    def list_blobs(
        self, prefix: str, extension: str | None = None
    ) -> list[str]:
        """List blob names under a prefix, optionally filtered by extension."""
        self._ensure_client()
        names: list[str] = []
        for blob in self._container_client.list_blobs(name_starts_with=prefix):
            name = blob.name
            if extension and not name.lower().endswith(extension):
                continue
            names.append(name)
        return sorted(names)

    def download_blob(self, blob_name: str) -> bytes:
        """Download a single blob as bytes with retry logic."""
        self._ensure_client()
        blob_client = self._container_client.get_blob_client(blob_name)
        
        last_error = None
        for attempt in range(1, MAX_DOWNLOAD_RETRIES + 1):
            try:
                logger.debug("Download attempt %d/%d for: %s", attempt, MAX_DOWNLOAD_RETRIES, blob_name)
                
                # Use chunked download for better reliability
                stream = BytesIO()
                download_stream = blob_client.download_blob(
                    max_concurrency=1,  # Serial download for stability
                )
                download_stream.readinto(stream)
                return stream.getvalue()
                
            except Exception as e:
                last_error = e
                logger.warning(
                    "Download attempt %d failed for %s: %s", 
                    attempt, blob_name, str(e)[:200]
                )
                if attempt < MAX_DOWNLOAD_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)  # Exponential backoff
        
        raise RuntimeError(
            f"Failed to download blob '{blob_name}' after {MAX_DOWNLOAD_RETRIES} attempts: {last_error}"
        )

    def iter_files(
        self, prefix: str, extensions: set[str] | None = None
    ) -> Iterator[tuple[str, bytes]]:
        """Yield (filename, file_bytes) for blobs matching extensions.

        ``filename`` is the blob name *without* the prefix (just the file
        name), matching what the ingest loaders expect from ``Path.name``.
        """
        self._ensure_client()
        for blob in self._container_client.list_blobs(name_starts_with=prefix):
            blob_name: str = blob.name
            filename = blob_name.rsplit("/", 1)[-1] if "/" in blob_name else blob_name
            if not filename:
                continue
            ext = Path(filename).suffix.lower()
            if extensions and ext not in extensions:
                continue
            logger.debug("Downloading blob: %s", blob_name)
            data = self.download_blob(blob_name)
            yield filename, data


def iter_files_for_source(
    source: str,
    extensions: set[str] | None = None,
) -> Iterator[tuple[str, bytes]]:
    """Universal file iterator -- works for both blob prefix and local dir.

    *source* is either:
      - A blob prefix like ``"big-thorium"``  (when blob mode is active)
      - A local directory path like ``"data/sample_corpus"``  (fallback)

    When ``settings.require_blob_source`` is True (default), local-dir
    fallback is **disabled** and a RuntimeError is raised if Azure Blob
    Storage is not configured.  This guardrail ensures all ingestion
    flows through the auditable Azure Blob pipeline.

    Yields ``(filename, file_bytes)`` tuples.
    """
    from app.config import settings as _cfg

    if is_blob_mode():
        prefix = source.rstrip("/") + "/"
        logger.info("Reading documents from blob prefix: %s", prefix)
        yield from get_blob_client().iter_files(prefix, extensions)
    elif _cfg.require_blob_source:
        raise RuntimeError(
            f"Azure Blob source is REQUIRED (require_blob_source=True) but "
            f"no AZURE_STORAGE_SAS_URL or AZURE_STORAGE_CONNECTION_STRING is set. "
            f"Local-file fallback is disabled. Set the env var or set "
            f"REQUIRE_BLOB_SOURCE=false to allow local files."
        )
    else:
        doc_dir = Path(source)
        if not doc_dir.is_dir():
            logger.warning("Local directory not found: %s", doc_dir)
            return
        logger.info("Reading documents from local directory: %s", doc_dir)
        for fpath in sorted(doc_dir.iterdir()):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            if extensions and ext not in extensions:
                continue
            yield fpath.name, fpath.read_bytes()
