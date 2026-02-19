"""Case registry — single source of truth for case isolation.

Each case defines the metadata filters that restrict retrieval to
**only** its own chunks in pgvector.  The UI and CLI both reference
this registry so behaviour is consistent everywhere.
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
    sample_questions: list[str] = field(default_factory=list)


# ── Registered cases ────────────────────────────────────────────────
CASES: dict[str, CaseConfig] = {}


def register(cfg: CaseConfig) -> CaseConfig:
    """Add a case to the global registry."""
    CASES[cfg.case_id] = cfg
    return cfg


# ─── BigThorium ─────────────────────────────────────────────────────
register(CaseConfig(
    case_id="big-thorium",
    display_name="Big Thorium",
    doc_type="corporate-comms",
    entity_name="Big Thorium",
    description="Corporate communications & internal documents for Big Thorium",
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
    description="Purview Exchange email archives (PST exports)",
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
    return list(CASES.keys())


def list_cases() -> list[CaseConfig]:
    """Return all registered case configs."""
    return list(CASES.values())
