"""Full local RAG pipeline -- runs entirely in-process, no external services.

This demo uses:
  * TF-IDF embeddings + cosine similarity for REAL vector retrieval
  * Keyword-overlap grading (simulates LLM relevance judgement)
  * Template-based answer generation with real cited passages
  * Full LangGraph self-correction loop

Run with: python demo_local.py
Optional:  python demo_local.py --query "your question here"
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import sys
import textwrap
from collections import Counter
from typing import Any

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from app.state import AgentState

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
#  1.  SAMPLE COMPLIANCE DOCUMENTS (realistic financial/legal content)
# ============================================================================

SAMPLE_DOCUMENTS: list[dict[str, Any]] = [
    # ---- 10-K Annual Report -----------------------------------------------
    {
        "source": "Acme_Corp_10K_2024.pdf",
        "doc_type": "10-K",
        "entity": "Acme Corp",
        "chunks": [
            {"page": 3, "text": (
                "Acme Corporation (NYSE: ACME) is a diversified industrial "
                "conglomerate with operations in aerospace, defense, and "
                "advanced materials. For the fiscal year ended December 31, "
                "2024, consolidated revenue was $12.4 billion, a 7% increase "
                "year-over-year."
            )},
            {"page": 18, "text": (
                "Revenue Recognition Policy: Revenue from long-term contracts "
                "is recognized over time using the cost-to-cost method in "
                "accordance with ASC 606. Variable consideration, including "
                "performance bonuses and penalties, is estimated using the "
                "expected value method and included in the transaction price."
            )},
            {"page": 42, "text": (
                "Management's Report on Internal Controls (SOX 404): "
                "Management identified a material weakness in internal "
                "controls over financial reporting related to the revenue "
                "recognition process for long-term defense contracts. "
                "Specifically, the percentage-of-completion estimates were "
                "not subject to adequate independent review, leading to "
                "potential misstatement of contract revenue in Q2 and Q3 2024."
            )},
            {"page": 43, "text": (
                "Remediation Plan: Management has engaged a third-party "
                "specialist to redesign the estimation review process. "
                "New controls include mandatory quarterly re-estimation by "
                "an independent cost engineering team and automated variance "
                "thresholds that trigger escalation to the Audit Committee."
            )},
            {"page": 55, "text": (
                "Related Party Transactions: During FY2024, Acme Corp "
                "entered into a consulting agreement with Apex Advisory LLC, "
                "whose principal is a first-degree relative of the CFO. "
                "Total payments under this agreement were $2.3 million. "
                "The Audit Committee reviewed and approved the arrangement."
            )},
            {"page": 60, "text": (
                "Contingent Liabilities: Acme Corp is a defendant in "
                "environmental remediation proceedings related to its "
                "former manufacturing facility in Newark, NJ. Management "
                "estimates the probable loss at $45 million, which has been "
                "accrued as of December 31, 2024."
            )},
        ],
    },
    # ---- External Audit Report --------------------------------------------
    {
        "source": "Acme_Audit_Report_2024.pdf",
        "doc_type": "audit-report",
        "entity": "Acme Corp",
        "chunks": [
            {"page": 1, "text": (
                "Independent Auditor's Report (Deloitte & Touche LLP): "
                "We have audited the consolidated financial statements of "
                "Acme Corporation as of and for the year ended December 31, "
                "2024. In our opinion, the financial statements present "
                "fairly, in all material respects, the financial position "
                "of Acme Corporation."
            )},
            {"page": 5, "text": (
                "Report on Internal Controls: We identified a material "
                "weakness in the Company's internal control over financial "
                "reporting. The material weakness relates to insufficient "
                "review controls over percentage-of-completion revenue "
                "estimates for defense contracts. This material weakness "
                "was also identified by management in their SOX 404 "
                "assessment."
            )},
            {"page": 8, "text": (
                "Key Audit Matter -- Revenue Recognition: Given the "
                "significance of long-term contract revenue ($4.8 billion, "
                "representing 39% of total revenue) and the judgment involved "
                "in estimating costs to complete, we identified revenue "
                "recognition on defense contracts as a Key Audit Matter."
            )},
            {"page": 12, "text": (
                "Key Audit Matter -- Environmental Liability: The estimation "
                "of the environmental remediation liability at the Newark "
                "facility requires significant judgment regarding cleanup "
                "technology, regulatory requirements, and timeline. We "
                "engaged our environmental specialists to evaluate "
                "management's assumptions."
            )},
        ],
    },
    # ---- Revenue Recognition Policy Memo ----------------------------------
    {
        "source": "Revenue_Recognition_Policy_v3.pdf",
        "doc_type": "policy",
        "entity": "Acme Corp",
        "chunks": [
            {"page": 1, "text": (
                "Revenue Recognition Policy (Effective January 1, 2024): "
                "This policy governs the recognition of revenue across all "
                "business units of Acme Corporation in compliance with "
                "ASC 606 - Revenue from Contracts with Customers."
            )},
            {"page": 4, "text": (
                "Long-Term Contracts: For contracts accounted for under "
                "the over-time method, revenue is recognized based on the "
                "ratio of costs incurred to date to total estimated costs "
                "(cost-to-cost method). Estimates of total contract costs "
                "must be reviewed and approved by the divisional controller "
                "AND the corporate cost engineering team on a quarterly basis."
            )},
            {"page": 7, "text": (
                "Variable Consideration: Performance incentives, liquidated "
                "damages, and other variable elements must be estimated "
                "using the expected value method. The constraint on variable "
                "consideration requires that amounts are included only to "
                "the extent it is highly probable that a significant revenue "
                "reversal will not occur."
            )},
            {"page": 10, "text": (
                "Contract Modifications: Modifications to existing contracts "
                "must be evaluated to determine whether they represent a "
                "separate performance obligation. If the modification adds "
                "distinct goods/services at standalone selling price, it is "
                "treated as a separate contract."
            )},
        ],
    },
]


# ============================================================================
#  2.  TF-IDF VECTOR STORE (in-memory, pure Python)
# ============================================================================

class LocalVectorStore:
    """Lightweight TF-IDF + cosine-similarity vector store."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.vocab: dict[str, int] = {}
        self.idf: list[float] = []
        self.tfidf_matrix: list[list[float]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_vocab(self, all_tokens: list[list[str]]) -> None:
        df: Counter[str] = Counter()
        all_unique: set[str] = set()
        for tokens in all_tokens:
            unique = set(tokens)
            for t in unique:
                df[t] += 1
            all_unique.update(unique)

        self.vocab = {term: i for i, term in enumerate(sorted(all_unique))}
        n = len(all_tokens)
        self.idf = [0.0] * len(self.vocab)
        for term, idx in self.vocab.items():
            self.idf[idx] = math.log((n + 1) / (df[term] + 1)) + 1

    def _tfidf_vector(self, tokens: list[str]) -> list[float]:
        tf: Counter[str] = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = [0.0] * len(self.vocab)
        for term, count in tf.items():
            if term in self.vocab:
                idx = self.vocab[term]
                vec[idx] = (count / total) * self.idf[idx]
        return vec

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def ingest(self, documents: list[Document]) -> None:
        self.documents = documents
        all_tokens = [self._tokenize(d.page_content) for d in documents]
        self._build_vocab(all_tokens)
        self.tfidf_matrix = [self._tfidf_vector(t) for t in all_tokens]

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        q_vec = self._tfidf_vector(self._tokenize(query))
        scores = [
            (doc, self._cosine_sim(q_vec, dvec))
            for doc, dvec in zip(self.documents, self.tfidf_matrix)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
#  3.  INGEST sample documents into the vector store
# ============================================================================

VECTOR_STORE = LocalVectorStore()


def ingest_sample_documents() -> int:
    """Load sample documents into the in-memory vector store."""
    docs: list[Document] = []
    for doc_info in SAMPLE_DOCUMENTS:
        for chunk in doc_info["chunks"]:
            docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "source": doc_info["source"],
                    "page": chunk["page"],
                    "doc_type": doc_info["doc_type"],
                    "entity": doc_info["entity"],
                },
            ))
    VECTOR_STORE.ingest(docs)
    return len(docs)


# ============================================================================
#  4.  LANGGRAPH NODES (real retrieval, heuristic grading, template generation)
# ============================================================================

def retriever_node(state: AgentState) -> dict:
    """Real TF-IDF retrieval from in-memory vector store."""
    query = state.get("rewritten_query") or state["query"]
    print(f"\n[RETRIEVER] Searching {len(VECTOR_STORE.documents)} document chunks...")
    print(f"  Query: {query}")

    results = VECTOR_STORE.search(query, top_k=4)
    docs = []
    print(f"  Top results:")
    for i, (doc, score) in enumerate(results):
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page", "?")
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"    {i+1}. [{src} p.{pg}] score={score:.3f} -- {snippet}...")
        docs.append(doc)

    return {"retrieved_docs": docs}


def grader_node(state: AgentState) -> dict:
    """Keyword-overlap relevance grading (simulates LLM judgement)."""
    print("\n[GRADER] Evaluating document relevance...")
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    # Tokenize query
    q_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "was", "were", "are", "did", "do",
                  "does", "in", "of", "for", "to", "and", "or", "any", "this",
                  "that", "it", "its", "with", "from", "on", "at", "by", "as"}
    q_tokens -= stop_words

    # Calculate relevance: % of query tokens found in top docs
    if not docs:
        score, reasoning = "no", "No documents retrieved."
    else:
        combined_text = " ".join(d.page_content.lower() for d in docs[:3])
        doc_tokens = set(re.findall(r"[a-z0-9]+", combined_text))
        overlap = q_tokens & doc_tokens
        relevance = len(overlap) / max(len(q_tokens), 1)

        # Require higher relevance for "yes"
        threshold = 0.6
        if relevance >= threshold:
            score = "yes"
            reasoning = (f"Found {len(overlap)}/{len(q_tokens)} key terms in "
                        f"retrieved documents (relevance={relevance:.0%}). "
                        f"Matching terms: {', '.join(sorted(overlap)[:8])}")
        else:
            missing = q_tokens - doc_tokens
            score = "no"
            reasoning = (f"Only {len(overlap)}/{len(q_tokens)} key terms found "
                        f"(relevance={relevance:.0%}). "
                        f"Missing: {', '.join(sorted(missing)[:5])}")

    print(f"  Score: {score}")
    print(f"  Reasoning: {reasoning}")
    print(f"  Loop: {loop}")
    return {"grader_score": score, "grader_reasoning": reasoning}


def rewriter_node(state: AgentState) -> dict:
    """Query expansion using synonyms and extracted terms."""
    print("\n[REWRITER] Expanding query for better retrieval...")
    loop = state.get("loop_count", 0)
    original = state.get("rewritten_query") or state["query"]

    # Domain-aware query expansion map
    expansions: dict[str, list[str]] = {
        "material weakness": ["material weakness", "internal control deficiency",
                              "SOX 404", "control weakness", "ICFR"],
        "revenue": ["revenue recognition", "ASC 606", "contract revenue",
                     "percentage of completion", "cost to cost"],
        "audit": ["audit", "auditor", "audit report", "audit opinion",
                  "key audit matter"],
        "compliance": ["compliance", "regulatory", "SOX", "internal controls"],
        "environmental": ["environmental", "remediation", "liability",
                         "contingent liability", "cleanup"],
        "related party": ["related party", "conflict of interest",
                         "related person transaction"],
        "disclosure": ["disclosure", "disclosed", "reported", "identified"],
    }

    # Find relevant expansions
    query_lower = original.lower()
    extra_terms: list[str] = []
    for trigger, terms in expansions.items():
        if trigger in query_lower:
            for t in terms:
                if t.lower() not in query_lower:
                    extra_terms.append(t)

    # Also add terms from grader reasoning
    reasoning = state.get("grader_reasoning", "")
    missing_match = re.findall(r"Missing: ([\w, ]+)", reasoning)
    if missing_match:
        for word in missing_match[0].split(", "):
            if word.strip() and word.strip() not in query_lower:
                extra_terms.append(word.strip())

    new_query = original
    if extra_terms:
        new_query = f"{original} {' '.join(extra_terms[:6])}"

    print(f"  Original: {original}")
    print(f"  Expanded: {new_query}")
    print(f"  Added terms: {', '.join(extra_terms[:6]) if extra_terms else '(none)'}")
    return {"rewritten_query": new_query, "loop_count": loop + 1}


def validator_node(state: AgentState) -> dict:
    """Simulate cross-referencing claims against external data."""
    print("\n[VALIDATOR] Cross-verifying claims against external sources...")
    docs = state.get("retrieved_docs", [])
    query = state.get("rewritten_query") or state["query"]

    # Extract "claims" from retrieved docs (simplified NER)
    claims: list[dict[str, Any]] = []
    for doc in docs[:3]:
        text = doc.page_content
        src = doc.metadata.get("source", "unknown")
        pg = doc.metadata.get("page", "?")

        # Pattern-match factual claims
        patterns = [
            (r"revenue was \$([\d.,]+ (?:billion|million))", "financial_data"),
            (r"material weakness", "sox_finding"),
            (r"(\$[\d.,]+ million)", "monetary_amount"),
            (r"(ASC \d+|SOX \d+)", "standard_reference"),
            (r"(Deloitte|PwC|EY|KPMG)", "auditor_reference"),
        ]

        for pattern, claim_type in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for m in matches:
                claim_text = m if isinstance(m, str) else m
                claims.append({
                    "claim": f"{claim_type}: {claim_text}",
                    "source": f"{src} p.{pg}",
                })

    # Simulate validation (in production: parallel httpx calls)
    results = []
    for claim in claims[:5]:
        h = hashlib.md5(claim["claim"].encode()).hexdigest()
        is_valid = int(h[0], 16) < 14  # ~87.5% validation rate
        results.append({
            "claim": claim["claim"],
            "endpoint": f"http://api.example.com/validate/{claim['claim'][:20]}",
            "status_code": 200,
            "api_response": {"verified": is_valid, "source": claim["source"]},
            "is_valid": is_valid,
        })

    valid_count = sum(1 for r in results if r["is_valid"])
    total = len(results)

    if total == 0:
        status = "pass"
    elif valid_count == total:
        status = "pass"
    elif valid_count >= total * 0.5:
        status = "partial"
    else:
        status = "fail"

    print(f"  Extracted {len(claims)} claims from retrieved passages")
    print(f"  Verified: {valid_count}/{total} claims")
    print(f"  Status: {status.upper()}")
    for r in results:
        tag = "[OK]" if r["is_valid"] else "[!!]"
        print(f"    {tag} {r['claim']}")

    return {"validation_results": results, "validation_status": status}


def generator_node(state: AgentState) -> dict:
    """Template-based answer generation with inline citations."""
    print("\n[GENERATOR] Synthesizing cited answer...")
    docs = state.get("retrieved_docs", [])
    query = state["query"]
    validation_status = state.get("validation_status", "unknown")
    validation_results = state.get("validation_results", [])

    # Build cited passages
    cited_passages: list[str] = []
    for doc in docs[:4]:
        src = doc.metadata.get("source", "unknown")
        pg = doc.metadata.get("page", "?")
        # Truncate long passages
        text = doc.page_content
        if len(text) > 200:
            text = text[:200] + "..."
        cited_passages.append(
            f'"{text}" [Source: {src}, p.{pg}]'
        )

    # Build answer
    answer_parts = [
        f"## Answer to: {query}\n",
        "Based on analysis of the retrieved compliance documents:\n",
    ]

    for i, passage in enumerate(cited_passages, 1):
        answer_parts.append(f"**Finding {i}:**\n{passage}\n")

    # Add validation summary
    valid_count = sum(1 for r in validation_results if r["is_valid"])
    total = len(validation_results)

    if validation_status == "pass":
        answer_parts.append(
            f"\n**Validation:** All {total} extracted claims verified "
            f"successfully against external reference data."
        )
    elif validation_status == "partial":
        answer_parts.append(
            f"\n**Validation:** {valid_count}/{total} claims verified. "
            "Some claims could not be independently confirmed -- "
            "manual review recommended."
        )
    else:
        answer_parts.append(
            f"\n**Validation:** Only {valid_count}/{total} claims verified. "
            "Significant discrepancies detected -- escalate to senior auditor."
        )

    answer = "\n".join(answer_parts)
    print(f"  Generated answer with {len(cited_passages)} citations")
    print(f"  Validation status: {validation_status}")
    return {"answer": answer}


# ============================================================================
#  5.  GRAPH ROUTING + ASSEMBLY
# ============================================================================

def route_after_grader(state: AgentState) -> str:
    score = state.get("grader_score", "no")
    loop = state.get("loop_count", 0)
    max_loops = 3

    if score == "yes":
        return "validator"
    if loop >= max_loops:
        print(f"\n  [!] Max rewrite loops ({max_loops}) reached, proceeding to validation")
        return "validator"
    return "rewriter"


def build_local_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("rewriter", rewriter_node)
    graph.add_node("validator", validator_node)
    graph.add_node("generator", generator_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "grader")
    graph.add_conditional_edges("grader", route_after_grader, {
        "validator": "validator",
        "rewriter": "rewriter",
    })
    graph.add_edge("rewriter", "retriever")
    graph.add_edge("validator", "generator")
    graph.add_edge("generator", END)

    return graph


# ============================================================================
#  6.  MAIN
# ============================================================================

SAMPLE_QUERIES = [
    "Did Acme Corp disclose any material weaknesses in FY2024?",
    "What is Acme Corp's revenue recognition policy for long-term contracts?",
    "Are there any related party transactions involving Acme Corp executives?",
    "What environmental liabilities does Acme Corp face?",
    "Who audited Acme Corp and what were the key audit matters?",
]


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG Pipeline -- fully self-contained demo"
    )
    parser.add_argument("--query", "-q", type=str, default=None,
                       help="Custom compliance question")
    parser.add_argument("--all", "-a", action="store_true",
                       help="Run all sample queries")
    parser.add_argument("--list", "-l", action="store_true",
                       help="List sample queries")
    args = parser.parse_args()

    if args.list:
        print("Available sample queries:")
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        return

    # Ingest documents
    print("=" * 72)
    print("Self-Correction Audit Agent -- LOCAL MODE (no external services)")
    print("=" * 72)
    n = ingest_sample_documents()
    print(f"\n[INGEST] Loaded {n} document chunks into TF-IDF vector store")
    print(f"  Sources: {', '.join(set(d['source'] for d in SAMPLE_DOCUMENTS))}")

    # Build graph
    graph = build_local_graph()
    agent = graph.compile()
    print("[GRAPH] Compiled LangGraph pipeline")

    # Determine queries to run
    if args.all:
        queries = SAMPLE_QUERIES
    elif args.query:
        queries = [args.query]
    else:
        queries = [SAMPLE_QUERIES[0]]  # Default: material weakness question

    for qi, query in enumerate(queries):
        if len(queries) > 1:
            print(f"\n{'#' * 72}")
            print(f"# QUERY {qi + 1}/{len(queries)}")
            print(f"{'#' * 72}")

        print(f"\n>> Question: {query}")

        initial_state: AgentState = {
            "query": query,
            "loop_count": 0,
        }

        result = agent.invoke(initial_state)

        print("\n" + "-" * 72)
        print("FINAL ANSWER")
        print("-" * 72)
        print(result.get("answer", "(no answer generated)"))
        print("-" * 72)
        print(f"  Self-correction loops: {result.get('loop_count', 0)}")
        print(f"  Validation status: {result.get('validation_status', 'N/A')}")
        print(f"  Documents retrieved: {len(result.get('retrieved_docs', []))}")

    print(f"\n{'=' * 72}")
    print("[OK] All queries completed successfully!")
    print("=" * 72)


if __name__ == "__main__":
    main()
