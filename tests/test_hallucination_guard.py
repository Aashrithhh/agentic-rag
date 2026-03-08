"""Tests for the Hallucination Guard — scoring, decisions, flags, node, and defaults."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ── Score computation ────────────────────────────────────────────────


class TestComputeHallucinationScore:
    """Verify the weighted score formula and flag generation."""

    def test_all_clean_signals_yield_zero(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 10, "invalid_citations": 0},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[
                {"is_sufficiently_supported": True},
                {"is_sufficiently_supported": True},
            ],
        )
        assert score == 0.0
        assert flags == []

    def test_all_bad_signals_yield_high_score(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 8, "invalid_citations": 8},
            post_check_result={
                "unsupported_claims": [
                    {"severity": "critical"},
                    {"severity": "critical"},
                ]
            },
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": False},
            ],
        )
        assert score == pytest.approx(1.0)
        assert len(flags) == 3
        flag_types = {f["type"] for f in flags}
        assert flag_types == {"citation", "unsupported_claim", "critical_fact"}

    def test_partial_citations_score(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 8, "invalid_citations": 3},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[],
        )
        expected = 0.35 * (3 / 8)
        assert score == pytest.approx(expected, abs=0.001)
        assert len(flags) == 1
        assert flags[0]["type"] == "citation"
        assert "3/8" in flags[0]["detail"]

    def test_mixed_severity_claims(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 0, "invalid_citations": 0},
            post_check_result={
                "unsupported_claims": [
                    {"severity": "critical"},
                    {"severity": "moderate"},
                    {"severity": "minor"},
                ]
            },
            critical_fact_assessments=[],
        )
        severity_sum = 1.0 + 0.6 + 0.3
        max_possible = 3.0
        claims_score = severity_sum / max_possible
        expected = 0.35 * claims_score
        assert score == pytest.approx(expected, abs=0.001)

    def test_weak_critical_facts_only(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 0, "invalid_citations": 0},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
            ],
        )
        expected = 0.30 * (3 / 6)
        assert score == pytest.approx(expected, abs=0.001)
        assert len(flags) == 1
        assert flags[0]["type"] == "critical_fact"
        assert "3/6" in flags[0]["detail"]

    def test_score_clamped_to_one(self):
        """Even with extreme input the score should never exceed 1.0."""
        from app.hallucination_guard import compute_hallucination_score

        score, _ = compute_hallucination_score(
            citation_report={"total_citations": 1, "invalid_citations": 100},
            post_check_result={
                "unsupported_claims": [{"severity": "critical"}] * 50
            },
            critical_fact_assessments=[{"is_sufficiently_supported": False}] * 50,
        )
        assert score <= 1.0


# ── Decision thresholds ──────────────────────────────────────────────


class TestDecideHallucination:
    def test_pass_below_warn(self):
        from app.hallucination_guard import decide_hallucination

        assert decide_hallucination(0.0) == "pass"
        assert decide_hallucination(0.34) == "pass"

    def test_warn_at_threshold(self):
        from app.hallucination_guard import decide_hallucination

        assert decide_hallucination(0.35) == "warn"
        assert decide_hallucination(0.50) == "warn"
        assert decide_hallucination(0.64) == "warn"

    def test_block_at_threshold(self):
        from app.hallucination_guard import decide_hallucination

        assert decide_hallucination(0.65) == "block"
        assert decide_hallucination(1.0) == "block"


# ── run_hallucination_guard (answer modification) ────────────────────


class TestRunHallucinationGuard:
    def test_pass_preserves_answer(self):
        from app.hallucination_guard import run_hallucination_guard

        result = run_hallucination_guard(
            citation_report={"total_citations": 5, "invalid_citations": 0},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[],
            answer="Original answer",
        )
        assert result["hallucination_decision"] == "pass"
        assert "answer" not in result  # answer unchanged — not in result dict

    def test_warn_prepends_warning(self):
        from app.hallucination_guard import run_hallucination_guard

        result = run_hallucination_guard(
            citation_report={"total_citations": 8, "invalid_citations": 5},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": False},
            ],
            answer="Original answer",
        )
        assert result["hallucination_decision"] == "warn"
        assert "Hallucination Warning" in result["answer"]
        assert "Original answer" in result["answer"]

    def test_block_replaces_answer(self):
        from app.hallucination_guard import run_hallucination_guard

        result = run_hallucination_guard(
            citation_report={"total_citations": 8, "invalid_citations": 8},
            post_check_result={
                "unsupported_claims": [{"severity": "critical"}] * 3
            },
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": False},
            ],
            answer="Original answer",
        )
        assert result["hallucination_decision"] == "block"
        assert "Insufficient evidence" in result["answer"]
        assert "Original answer" not in result["answer"]


# ── No-signal safe defaults ──────────────────────────────────────────


class TestSafeDefaults:
    def test_empty_inputs_yield_pass(self):
        from app.hallucination_guard import run_hallucination_guard

        result = run_hallucination_guard(
            citation_report={},
            post_check_result={},
            critical_fact_assessments=[],
            answer="Some answer",
        )
        assert result["hallucination_score"] == 0.0
        assert result["hallucination_decision"] == "pass"
        assert result["hallucination_flags"] == []

    def test_none_like_values(self):
        from app.hallucination_guard import compute_hallucination_score

        score, flags = compute_hallucination_score(
            citation_report={"total_citations": 0, "invalid_citations": 0},
            post_check_result={},
            critical_fact_assessments=[],
        )
        assert score == 0.0
        assert flags == []


# ── Flag formatting ──────────────────────────────────────────────────


class TestFlagFormatting:
    def test_citation_flag_format(self):
        from app.hallucination_guard import compute_hallucination_score

        _, flags = compute_hallucination_score(
            citation_report={"total_citations": 8, "invalid_citations": 3},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[],
        )
        assert len(flags) == 1
        assert flags[0] == {"type": "citation", "detail": "3/8 invalid citations"}

    def test_unsupported_claim_flag_format(self):
        from app.hallucination_guard import compute_hallucination_score

        _, flags = compute_hallucination_score(
            citation_report={"total_citations": 0, "invalid_citations": 0},
            post_check_result={
                "unsupported_claims": [
                    {"severity": "critical"},
                    {"severity": "critical"},
                ]
            },
            critical_fact_assessments=[],
        )
        assert len(flags) == 1
        assert flags[0] == {
            "type": "unsupported_claim",
            "detail": "2 critical unsupported claims",
        }

    def test_critical_fact_flag_format(self):
        from app.hallucination_guard import compute_hallucination_score

        _, flags = compute_hallucination_score(
            citation_report={"total_citations": 0, "invalid_citations": 0},
            post_check_result={"unsupported_claims": []},
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
            ],
        )
        assert len(flags) == 1
        assert flags[0] == {
            "type": "critical_fact",
            "detail": "3/6 weak critical facts",
        }

    def test_all_flag_types_present(self):
        from app.hallucination_guard import compute_hallucination_score

        _, flags = compute_hallucination_score(
            citation_report={"total_citations": 8, "invalid_citations": 3},
            post_check_result={
                "unsupported_claims": [{"severity": "critical"}]
            },
            critical_fact_assessments=[
                {"is_sufficiently_supported": False},
                {"is_sufficiently_supported": True},
            ],
        )
        flag_types = [f["type"] for f in flags]
        assert "citation" in flag_types
        assert "unsupported_claim" in flag_types
        assert "critical_fact" in flag_types


# ── Node integration (disabled guard) ────────────────────────────────


class TestHallucinationGuardNode:
    @patch("app.nodes.hallucination_guard.settings")
    def test_disabled_returns_defaults(self, mock_settings):
        mock_settings.hallucination_guard_enabled = False

        from app.nodes.hallucination_guard import hallucination_guard_node

        result = hallucination_guard_node({})
        assert result["hallucination_score"] == 0.0
        assert result["hallucination_decision"] == "pass"
        assert result["hallucination_flags"] == []

    @patch("app.nodes.hallucination_guard.settings")
    def test_enabled_processes_state(self, mock_settings):
        mock_settings.hallucination_guard_enabled = True
        mock_settings.hallucination_warn_threshold = 0.35
        mock_settings.hallucination_block_threshold = 0.65
        mock_settings.hallucination_invalid_citation_weight = 0.35
        mock_settings.hallucination_unsupported_claim_weight = 0.35
        mock_settings.hallucination_critical_fact_weight = 0.30

        from app.nodes.hallucination_guard import hallucination_guard_node

        state = {
            "citation_report": {"total_citations": 5, "invalid_citations": 0},
            "post_check_result": {"unsupported_claims": []},
            "critical_fact_assessments": [],
            "answer": "Test answer",
        }
        result = hallucination_guard_node(state)
        assert result["hallucination_decision"] == "pass"
        assert result["hallucination_score"] == 0.0
