"""Tests for app.graph — routing logic after the grader node."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from app.graph import _route_after_grader, build_graph


def test_route_yes_goes_to_validator():
    """When grader says 'yes', route to validator."""
    state = {"grader_score": "yes", "loop_count": 0}
    assert _route_after_grader(state) == "validator"


def test_route_no_goes_to_rewriter():
    """When grader says 'no' and loop count is below max, route to rewriter."""
    state = {"grader_score": "no", "loop_count": 0}
    with patch("app.graph.settings") as mock_settings:
        mock_settings.max_rewrite_loops = 3
        assert _route_after_grader(state) == "rewriter"


def test_route_max_loops_goes_to_validator():
    """When loop count reaches max, route to validator regardless of score."""
    state = {"grader_score": "no", "loop_count": 3}
    with patch("app.graph.settings") as mock_settings:
        mock_settings.max_rewrite_loops = 3
        assert _route_after_grader(state) == "validator"


def test_route_default_score_is_no():
    """Missing grader_score defaults to 'no'."""
    state = {"loop_count": 0}
    with patch("app.graph.settings") as mock_settings:
        mock_settings.max_rewrite_loops = 3
        assert _route_after_grader(state) == "rewriter"


def test_graph_includes_hallucination_guard_node():
    """Graph should contain the hallucination_guard node."""
    graph = build_graph()
    assert "hallucination_guard" in graph.nodes


def test_graph_generator_routes_to_hallucination_guard():
    """Generator should route to hallucination_guard, not directly to END."""
    graph = build_graph()
    gen_edges = graph.edges
    assert ("generator", "hallucination_guard") in gen_edges
