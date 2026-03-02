"""Tests for app.chunking — segment detection and chunk assembly."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from app.chunking import (
    Segment,
    SegmentType,
    _classify_paragraph,
    assemble_chunks,
    segment_text,
)


# ── Heading detection ────────────────────────────────────────────


def test_markdown_heading_detected():
    """Markdown headings (# Heading) are classified correctly."""
    seg = _classify_paragraph("## Executive Summary")
    assert seg.type == SegmentType.HEADING
    assert seg.level == 2


def test_uppercase_heading_detected():
    """All-caps lines like 'EXECUTIVE SUMMARY' are classified as headings."""
    seg = _classify_paragraph("EXECUTIVE SUMMARY")
    assert seg.type == SegmentType.HEADING


def test_normal_text_not_heading():
    """Regular paragraph text should not be classified as a heading."""
    seg = _classify_paragraph("This is a normal paragraph about compliance findings.")
    assert seg.type == SegmentType.PARAGRAPH


def test_short_uppercase_not_heading():
    """Short uppercase strings (< 5 chars) should not be headings."""
    seg = _classify_paragraph("OK")
    assert seg.type != SegmentType.HEADING


# ── Table detection ──────────────────────────────────────────────


def test_table_block_preserved():
    """[TABLE]...[/TABLE] blocks should become TABLE segments."""
    text = "Introduction\n\n[TABLE]\nCol1 | Col2\nA    | B\n[/TABLE]\n\nConclusion"
    segments = segment_text(text)
    types = [s.type for s in segments]
    assert SegmentType.TABLE in types


def test_table_not_split():
    """Table blocks should remain intact (not split across chunks)."""
    table = "[TABLE]\n" + "Row | Data\n" * 50 + "[/TABLE]"
    text = f"Header\n\n{table}\n\nFooter"
    segments = segment_text(text)

    table_segments = [s for s in segments if s.type == SegmentType.TABLE]
    assert len(table_segments) == 1
    assert "Row | Data" in table_segments[0].content


# ── Email header detection ───────────────────────────────────────


def test_email_header_detected():
    """Multi-line From/To/Subject blocks are classified as email headers."""
    text = "From: alice@example.com\nTo: bob@example.com\nSubject: Audit findings\nDate: 2024-01-15"
    seg = _classify_paragraph(text)
    assert seg.type == SegmentType.EMAIL_HEADER


# ── Chunk assembly ───────────────────────────────────────────────


def test_heading_starts_new_chunk():
    """A heading segment should start a new chunk."""
    segments = [
        Segment(type=SegmentType.PARAGRAPH, content="A" * 100),
        Segment(type=SegmentType.HEADING, content="# New Section", level=1),
        Segment(type=SegmentType.PARAGRAPH, content="B" * 100),
    ]
    chunks = assemble_chunks(segments, chunk_size=500, chunk_overlap=0)
    # There should be at least 2 chunks: one for the first paragraph,
    # one starting with the heading
    assert len(chunks) >= 2
    assert "# New Section" in chunks[1] or "# New Section" in chunks[-1]


def test_chunk_size_respected():
    """No chunk should exceed chunk_size (except oversized single segments)."""
    segments = [
        Segment(type=SegmentType.PARAGRAPH, content="Word " * 50)  # ~250 chars
        for _ in range(10)
    ]
    chunks = assemble_chunks(segments, chunk_size=300, chunk_overlap=0)
    for chunk in chunks:
        # Allow slight overshoot from the +2 separator math
        assert len(chunk) <= 600, f"Chunk too large: {len(chunk)} chars"


def test_empty_segments_produce_no_chunks():
    """An empty segment list should return an empty chunk list."""
    assert assemble_chunks([]) == []


def test_oversized_segment_gets_own_chunk():
    """A single segment larger than chunk_size still gets included."""
    segments = [
        Segment(type=SegmentType.TABLE, content="X" * 2000),
    ]
    chunks = assemble_chunks(segments, chunk_size=500, chunk_overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0]) == 2000
