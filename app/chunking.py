"""Structure-aware document chunking.

Replaces the generic RecursiveCharacterTextSplitter with a chunker that
respects document structure: tables are never split mid-row, headings stay
with their content, and section boundaries are preserved.

Two-phase approach:
  1. Segment detection — parse text into atomic structural units
  2. Chunk assembly — pack segments into size-limited chunks
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

from app.config import settings

logger = logging.getLogger(__name__)


# ── Segment types ────────────────────────────────────────────────────


class SegmentType(Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    EMAIL_HEADER = "email_header"
    TEXT = "text"


@dataclass
class Segment:
    """An atomic structural unit that should not be split."""

    type: SegmentType
    content: str
    level: int = 0  # heading level (1-6), 0 for non-headings
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.content)


# ── Regex patterns ───────────────────────────────────────────────────

# [TABLE]...[/TABLE] markers (produced by the DOCX loader)
_TABLE_RE = re.compile(
    r"\[TABLE\]\s*\n(.*?)\n\s*\[/TABLE\]",
    re.DOTALL,
)

# Markdown-style headings: # Heading, ## Heading, etc.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# Uppercase headings (common in legal / compliance docs):
# standalone line of at least 5 uppercase characters
_UPPER_HEADING_RE = re.compile(r"^([A-Z][A-Z\s,&:;\u2013\u2014-]{4,})$", re.MULTILINE)

# Email header block (From / To / Subject / Date lines)
_EMAIL_HEADER_RE = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received):\s*.+",
    re.MULTILINE | re.IGNORECASE,
)

# Placeholder token used during table extraction
_TABLE_PLACEHOLDER = "__TABLE_{idx}__"


# ── Phase 1: Segment detection ──────────────────────────────────────


def segment_text(text: str) -> list[Segment]:
    """Parse raw document text into structural segments.

    Strategy:
      1. Extract [TABLE]...[/TABLE] blocks, replace with placeholders.
      2. Split remaining text on blank lines (paragraph boundaries).
      3. Classify each paragraph as heading / email_header / paragraph.
      4. Re-insert table segments at their placeholder positions.
    """
    if not text or not text.strip():
        return [Segment(type=SegmentType.TEXT, content=text or "")]

    # ── Step 1: extract tables ─────────────────────────────────────
    tables: list[str] = []

    def _replace_table(m: re.Match) -> str:
        idx = len(tables)
        tables.append(m.group(0))  # keep the full [TABLE]...[/TABLE] block
        return _TABLE_PLACEHOLDER.format(idx=idx)

    text_without_tables = _TABLE_RE.sub(_replace_table, text)

    # ── Step 2: split on blank lines ───────────────────────────────
    raw_paragraphs = re.split(r"\n\s*\n", text_without_tables)

    # ── Step 3 & 4: classify & restore ─────────────────────────────
    segments: list[Segment] = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check for table placeholder
        table_placeholder_match = re.match(r"^__TABLE_(\d+)__$", para)
        if table_placeholder_match:
            idx = int(table_placeholder_match.group(1))
            if idx < len(tables):
                segments.append(Segment(type=SegmentType.TABLE, content=tables[idx]))
            continue

        # A paragraph might contain a table placeholder inline
        if "__TABLE_" in para:
            parts = re.split(r"(__TABLE_\d+__)", para)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                tbl_match = re.match(r"^__TABLE_(\d+)__$", part)
                if tbl_match:
                    idx = int(tbl_match.group(1))
                    if idx < len(tables):
                        segments.append(Segment(type=SegmentType.TABLE, content=tables[idx]))
                else:
                    segments.append(_classify_paragraph(part))
            continue

        segments.append(_classify_paragraph(para))

    return segments if segments else [Segment(type=SegmentType.TEXT, content=text)]


def _classify_paragraph(text: str) -> Segment:
    """Classify a single paragraph into the appropriate segment type."""
    stripped = text.strip()

    # Markdown heading (# Title, ## Subtitle, etc.)
    md_match = _MD_HEADING_RE.match(stripped)
    if md_match and "\n" not in stripped:
        level = len(md_match.group(1))
        return Segment(type=SegmentType.HEADING, content=stripped, level=level)

    # Uppercase heading (EXECUTIVE SUMMARY, SECTION 4: FINDINGS, etc.)
    if (
        len(stripped) >= 5
        and len(stripped) <= 120
        and "\n" not in stripped
        and not stripped.endswith(".")
        and _UPPER_HEADING_RE.match(stripped)
    ):
        return Segment(type=SegmentType.HEADING, content=stripped, level=1)

    # Email header block (multiple From:/To:/Subject: lines)
    header_lines = _EMAIL_HEADER_RE.findall(stripped)
    total_lines = len(stripped.splitlines())
    if header_lines and len(header_lines) >= 2 and len(header_lines) / max(total_lines, 1) > 0.5:
        return Segment(type=SegmentType.EMAIL_HEADER, content=stripped)

    return Segment(type=SegmentType.PARAGRAPH, content=stripped)


# ── Phase 2: Chunk assembly ─────────────────────────────────────────


def assemble_chunks(
    segments: list[Segment],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[str]:
    """Pack segments into chunks respecting size limits.

    Rules:
      1. Never split a segment across chunks.
      2. A heading segment starts a new chunk (unless current chunk is empty).
      3. Oversized segments get their own chunk (no truncation).
      4. Overlap: repeat trailing text from previous chunk.
    """
    if not segments:
        return []

    result_chunks: list[str] = []
    current_parts: list[str] = []
    current_size = 0

    for segment in segments:
        seg_text = segment.content
        seg_len = segment.char_count

        # Rule 2: heading starts a new chunk
        if segment.type == SegmentType.HEADING and current_size > 0:
            result_chunks.append(_join_parts(current_parts))
            overlap_text = _get_overlap(current_parts, chunk_overlap)
            current_parts = [overlap_text] if overlap_text else []
            current_size = len(overlap_text) if overlap_text else 0

        # Would exceed chunk_size — flush current, start new
        if current_size + seg_len + 2 > chunk_size and current_size > 0:
            result_chunks.append(_join_parts(current_parts))
            overlap_text = _get_overlap(current_parts, chunk_overlap)
            current_parts = [overlap_text] if overlap_text else []
            current_size = len(overlap_text) if overlap_text else 0

        # Add segment (even if alone it exceeds chunk_size — oversized chunk)
        current_parts.append(seg_text)
        current_size += seg_len + 2  # +2 for the \n\n separator

    # Flush remaining
    if current_parts:
        chunk_text = _join_parts(current_parts)
        if chunk_text.strip():
            result_chunks.append(chunk_text)

    return result_chunks


def _join_parts(parts: list[str]) -> str:
    """Join segment parts with double newlines."""
    return "\n\n".join(p for p in parts if p.strip())


def _get_overlap(parts: list[str], overlap_size: int) -> str:
    """Extract trailing overlap text from the last segment(s)."""
    if not parts or overlap_size <= 0:
        return ""
    combined = _join_parts(parts)
    if len(combined) <= overlap_size:
        return combined
    return combined[-overlap_size:]


# ── Public API ───────────────────────────────────────────────────────


def _dominant_segment_type(segments: list[Segment], chunk_text: str) -> str:
    """Determine the dominant segment type for a chunk."""
    for seg in segments:
        if seg.content and seg.content in chunk_text:
            return seg.type.value
    return SegmentType.PARAGRAPH.value


def chunk_documents_structured(
    docs: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    """Drop-in replacement for ``chunk_documents()`` in ingest.py.

    Each returned Document inherits the parent's metadata plus
    ``chunk_index`` and ``chunk_type`` fields.
    """
    chunks: list[Document] = []

    for doc in docs:
        segments = segment_text(doc.page_content)
        chunk_texts = assemble_chunks(segments, chunk_size, chunk_overlap)

        for idx, text in enumerate(chunk_texts):
            # Determine chunk_type from the first matching segment
            chunk_type = SegmentType.PARAGRAPH.value
            for seg in segments:
                if seg.content and seg.content[:50] in text:
                    chunk_type = seg.type.value
                    break

            chunks.append(Document(
                page_content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_type": chunk_type,
                },
            ))

    logger.info(
        "Structure-aware chunking: %d docs -> %d chunks (size=%d, overlap=%d)",
        len(docs), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks
