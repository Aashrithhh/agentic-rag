"""Self-Correction Audit Agent — entry point.

Usage:
    python main.py "Did Acme Corp disclose any material weaknesses in FY2024?"
    python main.py "What are the revenue recognition policies?" --doc-type "10-K" --entity "Acme Corp" --verbose
"""

from __future__ import annotations

import argparse
import json
import logging

from app.graph import compile_graph
from app.state import AgentState


def run_query(
    query: str,
    *,
    case_id: str | None = None,
    doc_type: str | None = None,
    entity_name: str | None = None,
    verbose: bool = False,
) -> AgentState:
    initial_state: AgentState = {"query": query, "loop_count": 0}

    # Case-level isolation takes priority over loose filters
    if case_id:
        initial_state["case_id"] = case_id
    elif doc_type or entity_name:
        meta_filter: dict = {}
        if doc_type: meta_filter["doc_type"] = doc_type
        if entity_name: meta_filter["entity_name"] = entity_name
        initial_state["metadata_filter"] = meta_filter

    agent = compile_graph()

    if verbose:
        print("=" * 72)
        print(f"QUERY: {query}")
        print("=" * 72)
        final_state = None
        for step in agent.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            print(f"\n{'-' * 36} {node_name} {'-' * 36}")
            for key, val in node_output.items():
                if key == "retrieved_docs":
                    print(f"  {key}: [{len(val)} documents]")
                elif key == "answer":
                    answer_str = str(val)
                    print(f"  {key}: {answer_str[:200]}..." if len(answer_str) > 200 else f"  {key}: {val}")
                else:
                    print(f"  {key}: {val}")
            final_state = node_output
        return final_state
    else:
        return agent.invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Correction Audit Agent")
    parser.add_argument("query", help="Compliance question to answer")
    parser.add_argument("--case", default=None, help="Case ID for isolated retrieval (e.g. big-thorium, purview-exchange)")
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true", dest="output_json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)-8s | %(name)-30s | %(message)s")
    result = run_query(args.query, case_id=args.case, doc_type=args.doc_type, entity_name=args.entity, verbose=args.verbose)

    if args.output_json:
        out = {}
        for k, v in result.items():
            out[k] = [{"content": d.page_content[:300], "metadata": d.metadata} for d in v] if k == "retrieved_docs" else v
        print(json.dumps(out, indent=2, default=str))
    else:
        print("\n" + "=" * 72)
        print("ANSWER")
        print("=" * 72)
        print(result.get("answer", "(no answer generated)"))
        if result.get("validation_status") in ("partial", "fail"):
            print(f"\n⚠  VALIDATION STATUS: {result['validation_status']}")
        if result.get("loop_count", 0) > 0:
            print(f"\n🔄 Query was rewritten {result['loop_count']} time(s).")


if __name__ == "__main__":
    main()
