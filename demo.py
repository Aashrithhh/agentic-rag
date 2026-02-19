"""Demo script — runs without database or external API.

This demonstrates the graph structure and node flow without requiring:
  • Postgres + pgvector
  • Anthropic API key
  • External validation endpoint

Run with: python demo.py
"""

from langgraph.graph import END, StateGraph
from app.state import AgentState

# ── Mock nodes ───────────────────────────────────────────────────────────

def mock_retriever(state: AgentState) -> dict:
    from langchain_core.documents import Document
    print("\n[RETRIEVER] Executing hybrid search...")
    query = state.get("rewritten_query") or state["query"]
    print(f"  Query: {query}")
    
    # Mock documents
    docs = [
        Document(
            page_content="Acme Corp disclosed material weaknesses in SOX 404 controls for FY2024.",
            metadata={"source": "10K_2024.pdf", "page": 42}
        ),
        Document(
            page_content="Internal audit identified deficiencies in revenue recognition processes.",
            metadata={"source": "audit_report.pdf", "page": 15}
        ),
    ]
    print(f"  Retrieved {len(docs)} documents")
    return {"retrieved_docs": docs}


def mock_grader(state: AgentState) -> dict:
    print("\n[GRADER] Evaluating relevance...")
    loop = state.get("loop_count", 0)
    
    # Simulate: fail first time, pass on rewrite
    if loop == 0:
        score = "no"
        reasoning = "Documents discuss controls but don't specifically address material weaknesses."
    else:
        score = "yes"
        reasoning = "Documents now contain relevant information about material weaknesses."
    
    print(f"  Score: {score}")
    print(f"  Reasoning: {reasoning}")
    return {"grader_score": score, "grader_reasoning": reasoning}


def mock_rewriter(state: AgentState) -> dict:
    print("\n[REWRITER] Improving query...")
    loop = state.get("loop_count", 0)
    original = state["query"]
    
    new_query = f"{original} SOX 404 material weaknesses internal controls"
    print(f"  Original: {original}")
    print(f"  Rewritten: {new_query}")
    return {"rewritten_query": new_query, "loop_count": loop + 1}


def mock_validator(state: AgentState) -> dict:
    print("\n[VALIDATOR] Cross-verifying claims...")
    print("  Simulating parallel API calls to /api/validate-all/...")
    
    results = [
        {
            "claim": "Acme Corp disclosed material weaknesses",
            "endpoint": "http://localhost:8000/api/validate-all/filings",
            "status_code": 200,
            "api_response": {"value": [{"id": 1, "filing_type": "10-K"}]},
            "is_valid": True,
        }
    ]
    print("  [OK] Verified 1/1 claims")
    return {"validation_results": results, "validation_status": "pass"}


def mock_generator(state: AgentState) -> dict:
    print("\n[GENERATOR] Synthesizing cited answer...")
    answer = """**Material Weaknesses Disclosure**

Acme Corp disclosed material weaknesses in internal controls over financial 
reporting related to SOX 404 compliance for FY2024 [Source: 10K_2024.pdf, p.42].

Specifically, the internal audit identified deficiencies in revenue recognition 
processes that did not adequately ensure accurate reporting [Source: audit_report.pdf, p.15].

[PASS] Validation Status: All claims verified against external systems."""
    
    print("  Generated answer with citations")
    return {"answer": answer}


# ── Routing ──────────────────────────────────────────────────────────────

def route_after_grader(state: AgentState) -> str:
    score = state.get("grader_score", "no")
    loop = state.get("loop_count", 0)
    
    if score == "yes":
        return "validator"
    if loop >= 3:
        return "validator"
    return "rewriter"


# ── Build graph ──────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Self-Correction Audit Agent — DEMO MODE")
    print("=" * 72)
    
    graph = StateGraph(AgentState)
    
    graph.add_node("retriever", mock_retriever)
    graph.add_node("grader", mock_grader)
    graph.add_node("rewriter", mock_rewriter)
    graph.add_node("validator", mock_validator)
    graph.add_node("generator", mock_generator)
    
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges("grader", route_after_grader, {
        "validator": "validator",
        "rewriter": "rewriter",
    })
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("validator", "generator")
    graph.add_edge("generator", END)
    
    agent = graph.compile()
    
    initial_state: AgentState = {
        "query": "Did Acme Corp disclose any material weaknesses in FY2024?",
        "loop_count": 0,
    }
    
    result = agent.invoke(initial_state)
    
    print("\n" + "=" * 72)
    print("FINAL ANSWER")
    print("=" * 72)
    print(result.get("answer", "(no answer)"))
    print("\n[OK] Demo completed successfully!")
    print(f"  Loop count: {result.get('loop_count', 0)}")
    print(f"  Validation status: {result.get('validation_status', 'N/A')}")


if __name__ == "__main__":
    main()
