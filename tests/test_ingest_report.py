"""Tests for IngestionReport in app.ingest."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from app.ingest import IngestionReport


def test_report_initial_state():
    """Fresh report has zero counts."""
    r = IngestionReport()
    assert r.total_files_found == 0
    assert r.files_loaded == 0
    assert r.files_failed == 0
    assert r.failed_files == []


def test_record_failure():
    """record_failure increments counter and appends to failed_files."""
    r = IngestionReport()
    r.record_failure("bad_file.pdf", "Corrupt PDF")

    assert r.files_failed == 1
    assert len(r.failed_files) == 1
    assert r.failed_files[0]["file"] == "bad_file.pdf"
    assert r.failed_files[0]["error"] == "Corrupt PDF"


def test_summary_format():
    """summary() returns a human-readable string with counts."""
    r = IngestionReport()
    r.total_files_found = 10
    r.files_loaded = 8
    r.files_failed = 2
    r.chunks_created = 45
    r.chunks_stored = 45

    s = r.summary()
    assert "loaded=8" in s
    assert "failed=2" in s
    assert "chunks=45" in s


def test_summary_partial_flag():
    """When there are failures, summary includes PARTIAL."""
    r = IngestionReport()
    r.files_loaded = 5
    r.files_failed = 1

    s = r.summary()
    assert "PARTIAL" in s


def test_summary_ok_flag():
    """When there are no failures, summary includes OK."""
    r = IngestionReport()
    r.files_loaded = 5
    r.files_failed = 0

    s = r.summary()
    assert "OK" in s
