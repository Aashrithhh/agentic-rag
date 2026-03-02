"""Unit tests for app.security — API key validation and error sanitization."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.security import _valid_api_keys, sanitize_error

pytestmark = pytest.mark.unit


# ── sanitize_error ───────────────────────────────────────────────────


class TestSanitizeError:
    def test_value_error_gets_safe_message(self):
        msg = sanitize_error(ValueError("bad input"))
        assert "invalid input" in msg.lower()
        assert "bad input" not in msg

    def test_connection_error_gets_safe_message(self):
        msg = sanitize_error(ConnectionError("psycopg2 failed"))
        assert "unavailable" in msg.lower()
        assert "psycopg2" not in msg

    def test_timeout_error_gets_safe_message(self):
        msg = sanitize_error(TimeoutError("read timed out"))
        assert "timed out" in msg.lower()
        assert "read timed out" not in msg

    def test_unknown_type_gets_generic(self):
        msg = sanitize_error(RuntimeError("sqlalchemy.engine url parse error at 0x7f"))
        assert "internal error" in msg.lower()
        assert "sqlalchemy" not in msg
        assert "0x7f" not in msg


# ── _valid_api_keys ──────────────────────────────────────────────────


class TestApiKeyParsing:
    @patch("app.security.settings")
    def test_empty_string_returns_empty_set(self, mock_settings):
        mock_settings.api_keys = ""
        assert _valid_api_keys() == set()

    @patch("app.security.settings")
    def test_single_key(self, mock_settings):
        mock_settings.api_keys = "abc123"
        assert _valid_api_keys() == {"abc123"}

    @patch("app.security.settings")
    def test_multiple_keys_with_whitespace(self, mock_settings):
        mock_settings.api_keys = " key1 , key2 , key3 "
        assert _valid_api_keys() == {"key1", "key2", "key3"}

    @patch("app.security.settings")
    def test_whitespace_only_returns_empty(self, mock_settings):
        mock_settings.api_keys = "   ,  , "
        assert _valid_api_keys() == set()
