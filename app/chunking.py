"""Structure-aware & email-aware document chunking.

Replaces the generic RecursiveCharacterTextSplitter with a chunker that
respects document structure: tables are never split mid-row, headings stay
with their content, and section boundaries are preserved.

Email-aware enhancements (gated by ``settings.email_chunk_rules``):
  - Email headers are chunked separately and linked to body via message_id
  - Quoted reply blocks (> lines, "On ... wrote:") are kept intact
  - Signature markers ("--", "Regards,") are preserved as units
  - Tables and lists are never split mid-structure
  - Dynamic chunk size by content type (dense legal vs conversational)

Two-phase approach:
  1. Segment detection — parse text into atomic structural units
  2. Chunk assembly — pack segments into size-limited chunks
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
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
    LIST = "list"
    EMAIL_HEADER = "email_header"
    EMAIL_BODY = "email_body"
    QUOTED_REPLY = "quoted_reply"
    SIGNATURE = "signature"
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
    r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received|Message-ID|In-Reply-To|References):\s*.+",
    re.MULTILINE | re.IGNORECASE,
)

# Quoted reply detection: "> text" or "On ... wrote:" pattern
_QUOTED_LINE_RE = re.compile(r"^>+\s?", re.MULTILINE)
_ON_WROTE_RE = re.compile(
    r"^On\s+.{10,80}\s+wrote:\s*$",
    re.MULTILINE | re.IGNORECASE,
)

# Signature markers
_SIGNATURE_RE = re.compile(
    r"^(--|___+|---+|\*\*\*+|~~+|Best\s+regards|Kind\s+regards|Regards|Sincerely|Thanks|Thank\s+you|Cheers|Sent\s+from)",
    re.MULTILINE | re.IGNORECASE,
)

# List patterns: numbered lists (1. / a. / i.) and bullet lists (- / * / •)
_LIST_LINE_RE = re.compile(
    r"^(\s*[\-\*\u2022]\s+|\s*\d+[\.\)]\s+|\s*[a-z][\.\)]\s+|\s*[ivxlcdm]+[\.\)]\s+)",
    re.MULTILINE | re.IGNORECASE,
)

# Placeholder token used during table extraction
_TABLE_PLACEHOLDER = "__TABLE_{idx}__"


# ── Content density classification ───────────────────────────────────

def _classify_content_density(text: str) -> str:
    """Classify text as 'dense', 'normal', or 'conversational'.

    Dense: legal/compliance text with long sentences, formal language.
    Conversational: short sentences, informal, email-style.
    Normal: everything else.
    """
    if not text or len(text) < 50:
        return "normal"

    lines = text.strip().splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return "normal"

    avg_line_len = sum(len(l) for l in non_empty) / len(non_empty)

    # Count formal/legal indicators
    legal_terms = len(re.findall(
        r'\b(pursuant|herein|thereof|whereas|notwithstanding|shall|hereby|'
        r'compliance|regulation|statute|provision|amendment|section\s+\d|'
        r'article\s+\d|subsection|paragraph|clause|stipulate|obligation)\b',
        text, re.IGNORECASE,
    ))
    legal_density = legal_terms / max(len(text.split()), 1)

    # Count conversational indicators
    conv_indicators = len(re.findall(
        r'\b(hi|hey|hello|thanks|cheers|btw|fyi|asap|lol|ok|okay|sure|'
        r'let me know|sounds good|no worries|got it)\b',
        text, re.IGNORECASE,
    ))

    # Heuristic classification
    if legal_density > 0.03 or avg_line_len > 120:
        return "dense"
    elif conv_indicators >= 2 or avg_line_len < 50:
        return "conversational"
    return "normal"


def _get_dynamic_chunk_size(text: str, base_size: int) -> int:
    """Return chunk size adjusted for content density.

    Only active when ``settings.email_dynamic_chunk_size`` is True.
    """
    if not settings.email_dynamic_chunk_size:
        return base_size

    density = _classify_content_density(text)
    if density == "dense":
        return settings.email_chunk_size_dense
    elif density == "conversational":
        return settings.email_chunk_size_conversational
    return settings.email_chunk_size_normal


# ── Email-specific segmentation ─────────────────────────────────────

def _extract_message_id(text: str) -> str:
    """Extract Message-ID from email headers, or generate a deterministic one."""
    m = re.search(r"Message-ID:\s*<?([^>\s]+)>?", text, re.IGNORECASE)
    if m:
        return m.group(1)
    # Generate deterministic ID from content hash
    return f"gen-{hashlib.sha256(text[:500].encode()).hexdigest()[:16]}"


def _extract_thread_id(text: str) -> str:
    """Extract thread ID from In-Reply-To or References headers."""
    m = re.search(r"(?:In-Reply-To|References):\s*<?([^>\s]+)>?", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _is_quoted_block(text: str) -> bool:
    """Check if a paragraph is primarily quoted text (> lines)."""
    lines = text.strip().splitlines()
    if not lines:
        return False
    quoted = sum(1 for l in lines if l.strip().startswith(">"))
    return quoted / len(lines) > 0.5


def _is_signature_block(text: str) -> bool:
    """Check if a paragraph is an email signature."""
    stripped = text.strip()
    if _SIGNATURE_RE.match(stripped):
        return True
    # Short block after a signature marker
    if stripped.startswith("--") and len(stripped) < 500:
        return True
    return False


def _is_list_block(text: str) -> bool:
    """Check if a paragraph is a list (bullet or numbered)."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False
    list_lines = sum(1 for l in lines if _LIST_LINE_RE.match(l))
    return list_lines / len(lines) > 0.5


def _split_email_into_logical_segments(text: str) -> list[Segment]:
    """Split email text into logical segments: header, body parts,
    quoted replies, and signatures.

    Preserves quoted reply blocks and signature markers as atomic units.
    """
    segments: list[Segment] = []

    # First, detect the header block (From/To/Subject/Date at the start)
    lines = text.split("\n")
    header_end = 0
    for i, line in enumerate(lines):
        if _EMAIL_HEADER_RE.match(line.strip()):
            header_end = i + 1
        elif line.strip() == "" and header_end > 0:
            # Blank line after headers = end of header block
            header_end = i
            break
        elif header_end > 0 and not _EMAIL_HEADER_RE.match(line.strip()):
            break

    if header_end > 0:
        header_text = "\n".join(lines[:header_end]).strip()
        if header_text:
            segments.append(Segment(type=SegmentType.EMAIL_HEADER, content=header_text))

    # Process remaining text
    remaining = "\n".join(lines[header_end:]).strip()
    if not remaining:
        return segments

    # Split remaining on blank lines
    paragraphs = re.split(r"\n\s*\n", remaining)

    in_quoted_block = False
    quoted_parts: list[str] = []
    signature_started = False
    signature_parts: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check for "On ... wrote:" pattern — starts a quoted block
        if _ON_WROTE_RE.search(para):
            # Flush any pending body content
            in_quoted_block = True
            quoted_parts.append(para)
            continue

        if in_quoted_block and (_is_quoted_block(para) or para.strip().startswith(">")):
            quoted_parts.append(para)
            continue
        elif in_quoted_block:
            # End of quoted block — flush
            if quoted_parts:
                segments.append(Segment(
                    type=SegmentType.QUOTED_REPLY,
                    content="\n\n".join(quoted_parts),
                ))
                quoted_parts = []
            in_quoted_block = False

        # Check for signature
        if _is_signature_block(para):
            signature_started = True
            signature_parts.append(para)
            continue

        if signature_started:
            signature_parts.append(para)
            continue

        # Check for list block
        if _is_list_block(para):
            segments.append(Segment(type=SegmentType.LIST, content=para))
            continue

        # Regular body paragraph
        segments.append(Segment(type=SegmentType.EMAIL_BODY, content=para))

    # Flush remaining quoted block
    if quoted_parts:
        segments.append(Segment(
            type=SegmentType.QUOTED_REPLY,
            content="\n\n".join(quoted_parts),
        ))

    # Flush signature
    if signature_parts:
        segments.append(Segment(
            type=SegmentType.SIGNATURE,
            content="\n\n".join(signature_parts),
        ))

    return segments


# ── Phase 1: Segment detection ──────────────────────────────────────


def segment_text(text: str, *, is_email: bool = False) -> list[Segment]:
    """Parse raw document text into structural segments.

    When ``is_email=True`` and ``settings.email_chunk_rules`` is enabled,
    uses email-specific segmentation that preserves quoted replies,
    signatures, and header/body boundaries.

    Strategy:
      1. Extract [TABLE]...[/TABLE] blocks, replace with placeholders.
      2. Split remaining text on blank lines (paragraph boundaries).
      3. Classify each paragraph as heading / email_header / paragraph.
      4. Re-insert table segments at their placeholder positions.
    """
    if not text or not text.strip():
        return [Segment(type=SegmentType.TEXT, content=text or "")]

    # Use email-specific segmentation if applicable
    if is_email and settings.email_chunk_rules:
        return _split_email_into_logical_segments(text)

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

        # Check for list blocks (never split mid-structure)
        if _is_list_block(para):
            segments.append(Segment(type=SegmentType.LIST, content=para))
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
      1. Never split a segment across chunks (incl. tables, lists).
      2. A heading segment starts a new chunk (unless current chunk is empty).
      3. Oversized segments get their own chunk (no truncation).
      4. Overlap: repeat trailing text from previous chunk.
      5. Tables and lists are never split mid-structure.
      6. Email headers always start a new chunk.
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

        # Rule 6: email headers start a new chunk
        if segment.type == SegmentType.EMAIL_HEADER and current_size > 0:
            result_chunks.append(_join_parts(current_parts))
            current_parts = []
            current_size = 0

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


def _detect_chunk_type(segments: list[Segment], chunk_text: str) -> str:
    """Determine the dominant segment type for a chunk."""
    for seg in segments:
        if seg.content and seg.content[:50] in chunk_text:
            return seg.type.value
    return SegmentType.PARAGRAPH.value


def _is_email_document(doc: Document) -> bool:
    """Heuristic: is the document an email (eml/msg/pst)?"""
    ft = doc.metadata.get("file_type", "")
    if ft in ("eml", "msg", "pst"):
        return True
    # Check for email headers in the content
    content_start = doc.page_content[:500]
    header_count = len(_EMAIL_HEADER_RE.findall(content_start))
    return header_count >= 2


def chunk_documents_structured(
    docs: list[Document],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[Document]:
    """Drop-in replacement for ``chunk_documents()`` in ingest.py.

    Each returned Document inherits the parent's metadata plus:
      - ``chunk_index``: position within the parent document
      - ``chunk_type``: segment type (heading, paragraph, email_header, etc.)
      - ``message_id``: unique email identifier (for email docs)
      - ``thread_id``: email thread identifier (for email docs)
      - ``total_chunks``: total chunks from this parent document

    When ``email_chunk_rules`` is enabled, email documents get
    specialized segmentation that preserves headers, quoted replies,
    and signatures as atomic units, with dynamic chunk sizing.
    """
    chunks: list[Document] = []

    for doc in docs:
        is_email = _is_email_document(doc)

        # Dynamic chunk size for emails
        effective_chunk_size = chunk_size
        if is_email and settings.email_chunk_rules:
            effective_chunk_size = _get_dynamic_chunk_size(
                doc.page_content, chunk_size,
            )

        segments = segment_text(doc.page_content, is_email=is_email)
        chunk_texts = assemble_chunks(segments, effective_chunk_size, chunk_overlap)

        # Extract email identifiers for linking
        message_id = ""
        thread_id = ""
        if is_email and settings.email_chunk_rules:
            message_id = _extract_message_id(doc.page_content)
            thread_id = _extract_thread_id(doc.page_content)
            # Also check metadata for message_id from the loader
            if not message_id or message_id.startswith("gen-"):
                meta_mid = doc.metadata.get("message_id", "")
                if meta_mid:
                    message_id = meta_mid

        total_chunks = len(chunk_texts)

        for idx, text in enumerate(chunk_texts):
            # Determine chunk_type from the first matching segment
            chunk_type = _detect_chunk_type(segments, text)

            chunk_meta = {
                **doc.metadata,
                "chunk_index": idx,
                "chunk_type": chunk_type,
                "total_chunks": total_chunks,
            }

            # Email-specific metadata
            if is_email and settings.email_chunk_rules:
                chunk_meta["message_id"] = message_id
                chunk_meta["thread_id"] = thread_id
                chunk_meta["content_density"] = _classify_content_density(text)

            chunks.append(Document(
                page_content=text,
                metadata=chunk_meta,
            ))

    logger.info(
        "Structure-aware chunking: %d docs -> %d chunks (base_size=%d, overlap=%d, email_rules=%s)",
        len(docs), len(chunks), chunk_size, chunk_overlap, settings.email_chunk_rules,
    )
    return chunks
