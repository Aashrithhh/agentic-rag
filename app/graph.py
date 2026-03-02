"""LangGraph — Self-Correction Audit Agent graph definition.

                ┌────────────┐
                │   START     │
                └─────┬──────┘
                      ▼
            ┌──────────────────┐
            │  Intent Router   │  ← classifies query intent
            └────────┬─────────┘
                     ▼
              ┌───────────────┐
              │   Retriever   │  ← hybrid HNSW + GIN search + reranking
              └───────┬───────┘
                      ▼
              ┌───────────────┐
              │    Grader     │  ← structured yes/no relevance
              └───────┬───────┘
                      │
              ┌───────┴───────┐
         yes  │               │  no  (& loop_count < MAX)
              ▼               ▼
     ┌────────────┐   ┌──────────────┐
     │  Validator  │   │  Rewriter    │──┐
     └─────┬──────┘   └──────────────┘  │
           ▼                loops ───────┘
     ┌────────────┐
     │  Generator  │  ← citation validation + HITL flagging
     └─────┬──────┘
           ▼
       ┌───────┐
       │  END  │
       └───────┘
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.config import settings
from app.nodes.generator import generator_node
from app.nodes.grader import grader_node
from app.nodes.retriever import retriever_node
from app.nodes.rewriter import rewriter_node
from app.nodes.validator import validator_node
from app.state import AgentState

logger = logging.getLogger(__name__)


# ── Intent Router Node ───────────────────────────────────────────────

def intent_router_node(state: AgentState) -> dict:
    """Classify query intent and set retrieval parameters."""
    if not settings.intent_routing_enabled:
        return {}

    from app.intent_router import route_query
    query = state.get("rewritten_query") or state["query"]
    try:
        intent_result, _params = route_query(query)
        logger.info(
            "Intent router — intent=%s confidence=%.2f",
            intent_result.intent,
            intent_result.confidence,
        )
        return {
            "intent": intent_result.intent,
            "intent_confidence": intent_result.confidence,
        }
    except Exception as exc:
        logger.warning("Intent routing failed: %s", exc)
        return {"intent": "fact_lookup", "intent_confidence": 0.0}


def _route_after_grader(state: AgentState) -> str:
    score = state.get("grader_score", "no")
    loop = state.get("loop_count", 0)

    if score == "yes":
        return "validator"
    if loop >= settings.max_rewrite_loops:
        return "validator"
    return "rewriter"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_router", intent_router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("validator", validator_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("intent_router")
    graph.add_edge("intent_router", "retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges("grader", _route_after_grader, {
        "validator": "validator",
        "rewriter": "rewriter",
    })
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("validator", "generator")
    graph.add_edge("generator", END)

    return graph


def compile_graph():
    return build_graph().compile()
