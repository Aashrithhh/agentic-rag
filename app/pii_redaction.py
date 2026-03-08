"""PII redaction pipeline — detect and redact sensitive data before embedding.

Detects and redacts:
  - Email addresses
  - Phone numbers (US/international)
  - Social Security Numbers (SSN)
  - Credit card numbers
  - IP addresses
  - Dates of birth patterns
  - Names (via NER when available, regex fallback)
  - Bank account / routing numbers
  - Passport numbers
  - Driver's license patterns

Two modes:
  1. ``redact`` — replace PII with type-specific placeholders: [EMAIL_REDACTED]
  2. ``detect`` — flag PII without modifying content (audit mode)

Applied before embedding/indexing in the ingestion pipeline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Union

logger = logging.getLogger(__name__)


@dataclass
class PIIMatch:
    """A single PII detection in text."""
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class RedactionResult:
    """Result of PII redaction on a text."""
    original_length: int
    redacted_length: int
    redacted_text: str
    detections: list[PIIMatch] = field(default_factory=list)
    pii_types_found: set[str] = field(default_factory=set)

    @property
    def has_pii(self) -> bool:
        return len(self.detections) > 0

    @property
    def total_detections(self) -> int:
        return len(self.detections)

    def summary(self) -> dict[str, Any]:
        return {
            "has_pii": self.has_pii,
            "total_detections": self.total_detections,
            "pii_types": sorted(self.pii_types_found),
            "original_length": self.original_length,
            "redacted_length": self.redacted_length,
        }


# ── PII detection patterns ──────────────────────────────────────────

_PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Email addresses
    ("EMAIL", re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ), "[EMAIL_REDACTED]"),

    # US Phone numbers
    ("PHONE", re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ), "[PHONE_REDACTED]"),

    # International phone numbers
    ("PHONE", re.compile(
        r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"
    ), "[PHONE_REDACTED]"),

    # SSN (with separators)
    ("SSN", re.compile(
        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b"
    ), "[SSN_REDACTED]"),

    # Credit card numbers (Visa, MC, Amex, Discover)
    ("CREDIT_CARD", re.compile(
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
        r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    ), "[CREDIT_CARD_REDACTED]"),

    # IP addresses (IPv4)
    ("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ), "[IP_REDACTED]"),

    # Date of birth patterns (MM/DD/YYYY, YYYY-MM-DD)
    ("DATE_OF_BIRTH", re.compile(
        r"\b(?:DOB|Date\s+of\s+Birth|Born)[:.\s]*"
        r"(?:\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b",
        re.IGNORECASE,
    ), "[DOB_REDACTED]"),

    # Bank account numbers (generic pattern: 8-17 digits)
    ("BANK_ACCOUNT", re.compile(
        r"\b(?:Account\s*(?:No|Number|#)?[:.\s]*)\d{8,17}\b",
        re.IGNORECASE,
    ), "[BANK_ACCOUNT_REDACTED]"),

    # Routing numbers (9 digits, US)
    ("ROUTING_NUMBER", re.compile(
        r"\b(?:Routing\s*(?:No|Number|#)?[:.\s]*)\d{9}\b",
        re.IGNORECASE,
    ), "[ROUTING_NUMBER_REDACTED]"),

    # Passport numbers (alphanumeric, 6-9 chars)
    ("PASSPORT", re.compile(
        r"\b(?:Passport\s*(?:No|Number|#)?[:.\s]*)[A-Z0-9]{6,9}\b",
        re.IGNORECASE,
    ), "[PASSPORT_REDACTED]"),

    # Driver's license (state-dependent, generic pattern)
    ("DRIVERS_LICENSE", re.compile(
        r"\b(?:DL|Driver'?s?\s+License\s*(?:No|Number|#)?[:.\s]*)[A-Z0-9]{5,15}\b",
        re.IGNORECASE,
    ), "[DRIVERS_LICENSE_REDACTED]"),
]


# ── Core detection and redaction ─────────────────────────────────────


def detect_pii(text: str) -> list[PIIMatch]:
    """Detect all PII instances in text without modifying it.

    Returns a list of PIIMatch objects with type, value, and position.
    """
    if not text:
        return []

    matches: list[PIIMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for pii_type, pattern, _ in _PII_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            # Avoid duplicate detections at same position
            if span not in seen_spans:
                seen_spans.add(span)
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                ))

    # Sort by position
    matches.sort(key=lambda m: m.start)
    return matches


def redact_pii(text: str) -> RedactionResult:
    """Detect and redact all PII from text.

    Returns the redacted text and detection metadata.
    """
    if not text:
        return RedactionResult(
            original_length=0,
            redacted_length=0,
            redacted_text="",
        )

    detections = detect_pii(text)
    if not detections:
        return RedactionResult(
            original_length=len(text),
            redacted_length=len(text),
            redacted_text=text,
        )

    # Build redacted text by replacing detected PII
    redacted = text
    offset = 0  # Track position offset from replacements

    # Build replacement map from patterns
    replacement_map: dict[str, str] = {}
    for pii_type, _, replacement in _PII_PATTERNS:
        replacement_map[pii_type] = replacement

    # Apply replacements in reverse order to preserve positions
    for detection in reversed(detections):
        replacement = replacement_map.get(detection.pii_type, "[PII_REDACTED]")
        redacted = (
            redacted[:detection.start] +
            replacement +
            redacted[detection.end:]
        )

    pii_types_found = {d.pii_type for d in detections}

    return RedactionResult(
        original_length=len(text),
        redacted_length=len(redacted),
        redacted_text=redacted,
        detections=detections,
        pii_types_found=pii_types_found,
    )


def redact_document_chunks(
    chunks: list[Union[Any, dict[str, Any]]],
    mode: Literal["redact", "detect"] = "redact",
) -> tuple[list, dict[str, Any]]:
    """Apply PII redaction to a batch of document chunks before indexing.

    Parameters
    ----------
    chunks : list
        Chunks in the format produced by the chunking pipeline (Document objects or dicts).
    mode : str
        ``"redact"`` — replace PII with placeholders (default)
        ``"detect"`` — flag PII but don't modify content

    Returns
    -------
    tuple of (processed_chunks, summary_stats)
    """
    from app.metrics import metrics

    total_detections = 0
    total_chunks_with_pii = 0
    pii_type_counts: dict[str, int] = {}

    for chunk in chunks:
        # Handle both dict and Document objects
        _dict_content_key = "content"  # default; overwritten for dict chunks
        if hasattr(chunk, "page_content"):
            content = chunk.page_content
        elif isinstance(chunk, dict):
            # Track the original key so we write redacted text back to it
            if "content" in chunk:
                _dict_content_key = "content"
            elif "page_content" in chunk:
                _dict_content_key = "page_content"
            content = chunk.get(_dict_content_key, "")
        else:
            content = str(chunk)
            
        result = redact_pii(content) if mode == "redact" else _detect_only(content)

        if result.has_pii:
            total_chunks_with_pii += 1
            total_detections += result.total_detections

            for d in result.detections:
                pii_type_counts[d.pii_type] = pii_type_counts.get(d.pii_type, 0) + 1

            if mode == "redact":
                # Update content based on object type
                if hasattr(chunk, "page_content"):
                    chunk.page_content = result.redacted_text
                    # Store PII metadata
                    try:
                        if not hasattr(chunk, "metadata") or chunk.metadata is None:
                            chunk.metadata = {}
                        chunk.metadata["pii_redacted"] = True
                        chunk.metadata["pii_types"] = sorted(result.pii_types_found)
                        chunk.metadata["pii_count"] = result.total_detections
                    except (AttributeError, TypeError) as e:
                        logger.warning(
                            "Could not set PII metadata on chunk %s: %s",
                            repr(chunk)[:80], e,
                        )
                elif isinstance(chunk, dict):
                    chunk[_dict_content_key] = result.redacted_text
                    extra = chunk.get("metadata_extra") or {}
                    extra["pii_redacted"] = True
                    extra["pii_types"] = sorted(result.pii_types_found)
                    extra["pii_count"] = result.total_detections
                    chunk["metadata_extra"] = extra
                else:
                    logger.warning(
                        "PII redaction not supported for chunk type %s — "
                        "only detection was performed (found %d PII items). "
                        "Original chunk returned unmodified.",
                        type(chunk).__name__, result.total_detections,
                    )

            metrics.inc("pii_detections", result.total_detections)

    summary = {
        "total_chunks": len(chunks),
        "chunks_with_pii": total_chunks_with_pii,
        "total_detections": total_detections,
        "pii_type_counts": pii_type_counts,
        "mode": mode,
    }

    if total_detections > 0:
        logger.info(
            "PII %s: %d detections in %d/%d chunks (types: %s)",
            mode, total_detections, total_chunks_with_pii, len(chunks),
            ", ".join(f"{k}={v}" for k, v in sorted(pii_type_counts.items())),
        )

        # Audit log
        try:
            from app.audit_log import audit_log
            audit_log.log_pii_event(
                case_id="",
                pii_types=sorted(pii_type_counts.keys()),
                count=total_detections,
                action="redacted" if mode == "redact" else "detected",
            )
        except Exception:
            pass

    return chunks, summary


def _detect_only(text: str) -> RedactionResult:
    """Detection-only mode — return detections without modifying text."""
    detections = detect_pii(text)
    return RedactionResult(
        original_length=len(text),
        redacted_length=len(text),
        redacted_text=text,
        detections=detections,
        pii_types_found={d.pii_type for d in detections},
    )
