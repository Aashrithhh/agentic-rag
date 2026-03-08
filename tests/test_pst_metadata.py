"""Tests for PST email metadata extraction and persistence.

Covers:
- compute_hallucination_score was already tested elsewhere; this file
  focuses on the new pst_email_metadata pipeline.
- Unit: _derive_message_id fallbacks, _derive_sender_email
- Unit: upsert_pst_email_metadata idempotency (ON CONFLICT DO UPDATE)
- Integration: _extract_pst_folder populates _pst_metadata_rows
- Integration: ingest() feature gate — only case_id == "demo-pst" persists rows
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ── Unit: _derive_message_id fallbacks ──────────────────────────────


class TestDeriveMessageId:
    """Test the three-level fallback for MessageId derivation."""

    def test_internet_message_id_preferred(self):
        """When PR_INTERNET_MESSAGE_ID is available, use it."""
        from app.ingest import _derive_message_id

        item = MagicMock()
        pa = MagicMock()
        pa.GetProperty.return_value = "<abc123@mail.example.com>"
        item.PropertyAccessor = pa
        item.EntryID = "ENTRY123"

        result = _derive_message_id(item, "test.pst", 1)
        assert result == "<abc123@mail.example.com>"

    def test_entry_id_fallback(self):
        """When internet message id is missing, fall back to EntryID."""
        from app.ingest import _derive_message_id

        item = MagicMock()
        pa = MagicMock()
        pa.GetProperty.side_effect = Exception("not available")
        item.PropertyAccessor = pa
        item.EntryID = "ENTRY456"

        result = _derive_message_id(item, "test.pst", 1)
        assert result == "ENTRY456"

    def test_hash_fallback(self):
        """When both internet id and EntryID are empty, use deterministic hash."""
        from app.ingest import _derive_message_id

        item = MagicMock()
        pa = MagicMock()
        pa.GetProperty.side_effect = Exception("not available")
        item.PropertyAccessor = pa
        item.EntryID = ""
        item.Subject = "Test Subject"
        item.SenderName = "Alice"
        item.ReceivedTime = "2024-10-03 09:15:00"

        result = _derive_message_id(item, "test.pst", 1)
        expected_raw = "test.pst|Test Subject|Alice|2024-10-03 09:15:00|1"
        expected = hashlib.sha256(expected_raw.encode()).hexdigest()
        assert result == expected

    def test_hash_is_deterministic(self):
        """Same inputs produce same hash — needed for idempotent upsert."""
        from app.ingest import _derive_message_id

        item = MagicMock()
        pa = MagicMock()
        pa.GetProperty.side_effect = Exception("nope")
        item.PropertyAccessor = pa
        item.EntryID = ""
        item.Subject = "Repeat"
        item.SenderName = "Bob"
        item.ReceivedTime = "2024-01-01"

        r1 = _derive_message_id(item, "a.pst", 5)
        r2 = _derive_message_id(item, "a.pst", 5)
        assert r1 == r2


# ── Unit: _derive_sender_email ──────────────────────────────────────


class TestDeriveSenderEmail:
    def test_returns_email_when_present(self):
        from app.ingest import _derive_sender_email

        item = MagicMock()
        item.SenderEmailAddress = "alice@example.com"
        assert _derive_sender_email(item) == "alice@example.com"

    def test_returns_empty_on_missing(self):
        from app.ingest import _derive_sender_email

        item = MagicMock(spec=[])  # no attributes
        assert _derive_sender_email(item) == ""

    def test_returns_empty_on_none(self):
        from app.ingest import _derive_sender_email

        item = MagicMock()
        item.SenderEmailAddress = None
        assert _derive_sender_email(item) == ""


# ── Unit: upsert_pst_email_metadata idempotency ────────────────────


class TestUpsertPstEmailMetadata:
    """Test the upsert helper using a real in-process SQLite-like mock
    is impractical (needs PG dialect). Instead we verify the function
    handles empty input and correct return value.
    """

    def test_empty_rows_returns_zero(self):
        from app.db import upsert_pst_email_metadata

        engine = MagicMock()
        assert upsert_pst_email_metadata(engine, []) == 0


# ── Integration: _pst_metadata_rows accumulation ───────────────────


class TestPstMetadataAccumulation:
    """Test that _extract_pst_folder populates _pst_metadata_rows."""

    def test_extract_populates_metadata_rows(self):
        from app.ingest import _extract_pst_folder, _pst_metadata_rows

        _pst_metadata_rows.clear()

        # Build a mock Outlook folder with one mail item
        item = MagicMock()
        item.Class = 43  # olMail
        item.SenderName = "James Martinez"
        item.SenderEmailAddress = "james@techcorp.com"
        item.To = "robert@techcorp.com"
        item.CC = ""
        item.Subject = "Investigation Update"
        item.ReceivedTime = "2024-10-10 15:30:00"
        item.Body = "Week 1 findings..."
        item.HTMLBody = "<html><body>Week 1 findings...</body></html>"
        item.EntryID = "ENTRY_001"
        item.ConversationID = "CONV_001"
        item.Attachments = MagicMock()
        item.Attachments.Count = 0

        # PropertyAccessor for _derive_message_id: simulate no internet msg id
        pa = MagicMock()
        pa.GetProperty.side_effect = Exception("not available")
        item.PropertyAccessor = pa

        items_mock = MagicMock()
        items_mock.Count = 1
        items_mock.Item.return_value = item

        folder = MagicMock()
        folder.Items = items_mock
        folder.Name = "Inbox"
        folder.Folders.Count = 0

        docs: list = []
        _extract_pst_folder(folder, "test_case.pst", docs)

        # Should have one document AND one metadata row
        assert len(docs) == 1
        assert len(_pst_metadata_rows) == 1

        row = _pst_metadata_rows[0]
        assert row["PstFileId"] == "test_case.pst"
        assert row["Subject"] == "Investigation Update"
        assert row["SenderName"] == "James Martinez"
        assert row["SenderEmail"] == "james@techcorp.com"
        assert row["RecipientTo"] == "robert@techcorp.com"
        assert row["HasAttachments"] is False
        assert row["FolderPath"] == "Inbox"
        assert row["MessageId"] == "ENTRY_001"  # EntryID fallback
        assert row["BodyText"] == "Week 1 findings..."
        assert "Week 1 findings" in row["BodyHtml"]

        _pst_metadata_rows.clear()

    def test_extract_with_attachments(self):
        from app.ingest import _extract_pst_folder, _pst_metadata_rows

        _pst_metadata_rows.clear()

        item = MagicMock()
        item.Class = 43
        item.SenderName = "Amanda Foster"
        item.SenderEmailAddress = "amanda@techcorp.com"
        item.To = "james@techcorp.com"
        item.CC = "sarah@techcorp.com"
        item.Subject = "Vendor Records"
        item.ReceivedTime = "2024-10-04"
        item.Body = "See attached records."
        item.HTMLBody = ""
        item.EntryID = "ENTRY_002"
        item.ConversationID = ""

        att = MagicMock()
        att.FileName = "records.pdf"
        item.Attachments = MagicMock()
        item.Attachments.Count = 1
        item.Attachments.Item.return_value = att

        pa = MagicMock()
        pa.GetProperty.side_effect = Exception("nope")
        item.PropertyAccessor = pa

        items_mock = MagicMock()
        items_mock.Count = 1
        items_mock.Item.return_value = item

        folder = MagicMock()
        folder.Items = items_mock
        folder.Name = "Sent Items"
        folder.Folders.Count = 0

        # Patch settings to disable attachment processing for this test
        with patch("app.ingest.settings") as mock_settings:
            mock_settings.process_attachments = False
            mock_settings.attachment_context_link = False
            docs: list = []
            _extract_pst_folder(folder, "records.pst", docs)

        assert len(_pst_metadata_rows) == 1
        row = _pst_metadata_rows[0]
        assert row["HasAttachments"] is True
        assert row["RecipientCc"] == "sarah@techcorp.com"
        assert row["FolderPath"] == "Sent Items"

        _pst_metadata_rows.clear()


# ── Integration: feature gate in ingest() ──────────────────────────


class TestIngestFeatureGate:
    """Verify that PST metadata is only persisted when case_id == 'demo-pst'."""

    @patch("app.ingest.upsert_pst_email_metadata")
    @patch("app.ingest.load_documents")
    @patch("app.ingest.init_db")
    @patch("app.ingest.settings")
    @patch("app.ingest.is_blob_mode", return_value=True)
    def test_demo_pst_case_persists_metadata(
        self, _blob, mock_settings, mock_init, mock_load, mock_upsert
    ):
        from app.ingest import _pst_metadata_rows, ingest

        mock_settings.require_blob_source = False
        mock_settings.use_structure_aware_chunking = False
        mock_settings.pii_redaction_enabled = False
        mock_settings.enable_metadata_enrichment = False
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50

        engine = MagicMock()
        mock_init.return_value = engine
        mock_upsert.return_value = 3

        # load_documents is called AFTER _pst_metadata_rows.clear(),
        # so we simulate the PST loader populating rows during load.
        fake_rows = [
            {"PstFileId": "test.pst", "Subject": "Subj1", "MessageId": "M1",
             "SenderName": "", "SenderEmail": "", "RecipientTo": "",
             "RecipientCc": "", "SentDate": "", "BodyText": "", "BodyHtml": "",
             "HasAttachments": False, "FolderPath": "Inbox"},
            {"PstFileId": "test.pst", "Subject": "Subj2", "MessageId": "M2",
             "SenderName": "", "SenderEmail": "", "RecipientTo": "",
             "RecipientCc": "", "SentDate": "", "BodyText": "", "BodyHtml": "",
             "HasAttachments": False, "FolderPath": "Inbox"},
            {"PstFileId": "test.pst", "Subject": "Subj3", "MessageId": "M3",
             "SenderName": "", "SenderEmail": "", "RecipientTo": "",
             "RecipientCc": "", "SentDate": "", "BodyText": "", "BodyHtml": "",
             "HasAttachments": False, "FolderPath": "Inbox"},
        ]

        def fake_load_documents(source):
            _pst_metadata_rows.extend(fake_rows)
            return []  # No docs — just testing metadata path

        mock_load.side_effect = fake_load_documents

        ingest("some-source", case_id="demo-pst")

        # Should have called upsert with case_id injected
        mock_upsert.assert_called_once()
        rows_arg = mock_upsert.call_args[0][1]
        assert len(rows_arg) == 3
        assert all(r["case_id"] == "demo-pst" for r in rows_arg)

    @patch("app.ingest.upsert_pst_email_metadata")
    @patch("app.ingest.load_documents")
    @patch("app.ingest.init_db")
    @patch("app.ingest.settings")
    @patch("app.ingest.is_blob_mode", return_value=True)
    def test_non_demo_case_skips_metadata(
        self, _blob, mock_settings, mock_init, mock_load, mock_upsert
    ):
        from app.ingest import _pst_metadata_rows, ingest

        mock_settings.require_blob_source = False
        mock_settings.use_structure_aware_chunking = False
        mock_settings.pii_redaction_enabled = False
        mock_settings.enable_metadata_enrichment = False
        mock_settings.chunk_size = 512
        mock_settings.chunk_overlap = 50

        engine = MagicMock()
        mock_init.return_value = engine

        def fake_load_documents(source):
            _pst_metadata_rows.append({
                "PstFileId": "test.pst", "Subject": "Subj1", "MessageId": "M1",
                "SenderName": "", "SenderEmail": "", "RecipientTo": "",
                "RecipientCc": "", "SentDate": "", "BodyText": "", "BodyHtml": "",
                "HasAttachments": False, "FolderPath": "Inbox",
            })
            return []

        mock_load.side_effect = fake_load_documents

        ingest("some-source", case_id="big-thorium")

        # Should NOT call upsert_pst_email_metadata for non-demo-pst case
        mock_upsert.assert_not_called()
