"""Case registry — single source of truth for case isolation.

Each case gets its **own PostgreSQL database**, enforcing strict data
isolation at the storage layer so one case can never access another
case's document chunks.  The UI and CLI both reference this registry
so behaviour is consistent everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaseConfig:
    """Immutable definition for one investigation case."""

    case_id: str                        # unique key (slug)
    display_name: str                   # human-readable label
    doc_type: str                       # filter value for document_chunks.doc_type
    entity_name: str | None = None      # optional filter on entity_name
    description: str = ""               # one-liner shown in UI
    data_dir: str = ""                  # relative path to source data folder (local fallback)
    blob_prefix: str = ""              # Azure Blob Storage prefix (e.g. "big-thorium")
    database_url: str = ""              # optional explicit DB URL override
    sample_questions: list[str] = field(default_factory=list)


# ── Registered cases ────────────────────────────────────────────────
CASES: dict[str, CaseConfig] = {}
_AUTO_DISCOVERY_DONE = False


def register(cfg: CaseConfig) -> CaseConfig:
    """Add a case to the global registry."""
    CASES[cfg.case_id] = cfg
    return cfg


def _discover_blob_cases() -> None:
    """Auto-discover cases from Azure Blob Storage top-level folders.
    
    Scans the blob container for folders and creates CaseConfig entries
    for any that aren't already manually registered. Only runs once per
    process and only when blob storage is configured.
    """
    global _AUTO_DISCOVERY_DONE
    if _AUTO_DISCOVERY_DONE:
        return
        
    import logging
    from app.blob_storage import is_blob_mode, get_blob_client
    
    logger = logging.getLogger(__name__)
    
    if not is_blob_mode():
        _AUTO_DISCOVERY_DONE = True
        return
    
    try:
        client = get_blob_client()
        # Get all blobs and extract top-level folder names
        blob_names = client.list_blobs(prefix="")
        folders = set()
        
        for blob_name in blob_names:
            if "/" in blob_name:
                folder = blob_name.split("/", 1)[0]
                folders.add(folder)
        
        # Get already-registered blob prefixes to avoid duplicates
        registered_prefixes = {cfg.blob_prefix for cfg in CASES.values() if cfg.blob_prefix}
        
        for folder in sorted(folders):
            # Skip if already registered
            if folder in registered_prefixes:
                continue
                
            # Generate case_id from folder name (slug-friendly)
            case_id = folder.lower().replace("_", "-").replace(" ", "-")
            
            # Skip if case_id already exists (edge case)
            if case_id in CASES:
                continue
            
            # Create display name (title case, replace separators)
            display_name = folder.replace("-", " ").replace("_", " ").title()
            
            # Auto-register with sensible defaults
            auto_case = CaseConfig(
                case_id=case_id,
                display_name=display_name,
                doc_type="document",  # Generic default
                description=f"Auto-discovered from blob folder: {folder}",
                blob_prefix=folder,
                sample_questions=[],
            )
            CASES[case_id] = auto_case
            logger.info("Auto-discovered case '%s' from blob folder '%s'", case_id, folder)
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to auto-discover blob cases: %s", e)
    
    _AUTO_DISCOVERY_DONE = True


# ─── BigThorium ─────────────────────────────────────────────────────
register(CaseConfig(
    case_id="big-thorium",
    display_name="Big Thorium",
    doc_type="compliance-report",
    entity_name="Big Thorium",
    description="Corporate communications & internal documents for Big Thorium",
    data_dir="data/sample_corpus",
    blob_prefix="big-thorium",
    sample_questions=[
        "What were the issues with the Indian workers at Big Thorium?",
        "What were the renewable energy plans discussed for the power plant?",
        "Who is William Davis and what communications did he have about welding?",
        "What were the housing conditions for recruited workers?",
        "What safety incidents were discussed in company emails?",
    ],
))

# ─── Purview Exchange ──────────────────────────────────────────────
register(CaseConfig(
    case_id="purview-exchange",
    display_name="Purview Exchange",
    doc_type="exchange-email",
    description="Purview eDiscovery export — Exchange emails, files, meetings, Teams/IM",
    data_dir="data/purview_exchange",
    blob_prefix="Reports-Case04_Sri-Review04-StartReviewSetExport-New_Export-2026-01-28_06-47-05",
    sample_questions=[
        "What emails did David Williams send and what were they about?",
        "What security-related discussions happened in the emails?",
        "Were there any phishing simulation campaigns discussed?",
        "What did Sarah Johnson communicate about?",
        "What fraud-related topics appear in the email archives?",
    ],
))


def get_case(case_id: str) -> CaseConfig:
    """Look up a case by ID. Raises KeyError if not found."""
    _discover_blob_cases()  # Ensure auto-discovery runs first
    return CASES[case_id]


def metadata_filter_for_case(case_id: str) -> dict[str, str]:
    """Build the metadata filter dict that hybrid_search expects."""
    cfg = get_case(case_id)
    filt: dict[str, str] = {"doc_type": cfg.doc_type}
    if cfg.entity_name:
        filt["entity_name"] = cfg.entity_name
    return filt


def list_case_ids() -> list[str]:
    """Return all registered case IDs in insertion order."""
    _discover_blob_cases()  # Ensure auto-discovery runs first
    return list(CASES.keys())


def list_cases() -> list[CaseConfig]:
    """Return all registered case configs."""
    _discover_blob_cases()  # Ensure auto-discovery runs first
    return list(CASES.values())


def auto_ingest_cases(*, project_root: str | None = None) -> dict[str, int]:
    """Auto-ingest data for all registered cases that have no chunks in their DB.

    Each case has its own isolated database.  When Azure Blob Storage is
    configured (``AZURE_STORAGE_CONNECTION_STRING`` set), documents are read
    from the blob container using the case's ``blob_prefix``.  Otherwise the
    local ``data_dir`` is used as a fallback.

    Returns ``{case_id: chunks_ingested}``.
    """
    import logging
    from pathlib import Path
    from sqlalchemy import select, func

    from app.blob_storage import is_blob_mode
    from app.db import init_db, document_chunks
    from app.ingest import ingest

    _discover_blob_cases()  # Ensure auto-discovery runs first
    
    logger = logging.getLogger(__name__)
    root = Path(project_root) if project_root else Path(__file__).resolve().parent.parent
    results: dict[str, int] = {}
    blob_mode = is_blob_mode()

    for cfg in CASES.values():
        # Determine the document source based on mode
        if blob_mode:
            if not cfg.blob_prefix:
                logger.debug("Auto-ingest: skipping %s — no blob_prefix configured", cfg.case_id)
                continue
            doc_source = cfg.blob_prefix
        else:
            if not cfg.data_dir:
                continue
            data_path = root / cfg.data_dir
            if not data_path.is_dir():
                logger.debug("Auto-ingest: skipping %s — folder %s not found", cfg.case_id, data_path)
                continue
            doc_source = str(data_path)

        # Initialise the per-case database (idempotent)
        engine = init_db(cfg.case_id)

        # Check if chunks already exist in this case's isolated DB
        with engine.connect() as conn:
            count = conn.execute(
                select(func.count()).select_from(document_chunks)
            ).scalar() or 0

        if count > 0:
            logger.info("Auto-ingest: %s already has %d chunks — skipping", cfg.case_id, count)
            continue

        logger.info("Auto-ingest: ingesting %s from %s …", cfg.case_id, doc_source)
        try:
            n, report = ingest(doc_source, case_id=cfg.case_id, doc_type=cfg.doc_type, entity_name=cfg.entity_name)
            results[cfg.case_id] = n
            logger.info("Auto-ingest: %s — %s", cfg.case_id, report.summary())
        except Exception as exc:
            logger.error("Auto-ingest: %s failed — %s", cfg.case_id, exc)
            results[cfg.case_id] = 0

    return results
