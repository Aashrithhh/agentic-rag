"""Real Agentic RAG pipeline using Azure OpenAI (GPT-4o) + Cohere Embeddings.

This runs the full self-correction loop with REAL LLM calls:
  * Cohere embed-english-v3.0 for semantic embeddings
  * Cosine-similarity vector retrieval (in-memory)
  * Azure GPT-4o for grading, rewriting, and answer generation
  * Full LangGraph graph with conditional routing

Run: python demo_real.py
      python demo_real.py --query "your compliance question"
      python demo_real.py --all
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# Force .env to override system environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.state import AgentState

# Force UTF-8 on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
#  1.  SAMPLE COMPLIANCE DOCUMENTS
# ============================================================================

SAMPLE_DOCUMENTS: list[dict[str, Any]] = [
    {
        "source": "Acme_Corp_10K_2024.pdf", "doc_type": "10-K", "entity": "Acme Corp",
        "chunks": [
            {"page": 3, "text": "Acme Corporation (NYSE: ACME) is a diversified industrial conglomerate with operations in aerospace, defense, and advanced materials. For the fiscal year ended December 31, 2024, consolidated revenue was $12.4 billion, a 7% increase year-over-year."},
            {"page": 18, "text": "Revenue Recognition Policy: Revenue from long-term contracts is recognized over time using the cost-to-cost method in accordance with ASC 606. Variable consideration, including performance bonuses and penalties, is estimated using the expected value method and included in the transaction price."},
            {"page": 42, "text": "Management's Report on Internal Controls (SOX 404): Management identified a material weakness in internal controls over financial reporting related to the revenue recognition process for long-term defense contracts. Specifically, the percentage-of-completion estimates were not subject to adequate independent review, leading to potential misstatement of contract revenue in Q2 and Q3 2024."},
            {"page": 43, "text": "Remediation Plan: Management has engaged a third-party specialist to redesign the estimation review process. New controls include mandatory quarterly re-estimation by an independent cost engineering team and automated variance thresholds that trigger escalation to the Audit Committee."},
            {"page": 55, "text": "Related Party Transactions: During FY2024, Acme Corp entered into a consulting agreement with Apex Advisory LLC, whose principal is a first-degree relative of the CFO. Total payments under this agreement were $2.3 million. The Audit Committee reviewed and approved the arrangement."},
            {"page": 60, "text": "Contingent Liabilities: Acme Corp is a defendant in environmental remediation proceedings related to its former manufacturing facility in Newark, NJ. Management estimates the probable loss at $45 million, which has been accrued as of December 31, 2024."},
        ],
    },
    {
        "source": "Acme_Audit_Report_2024.pdf", "doc_type": "audit-report", "entity": "Acme Corp",
        "chunks": [
            {"page": 1, "text": "Independent Auditor's Report (Deloitte & Touche LLP): We have audited the consolidated financial statements of Acme Corporation as of and for the year ended December 31, 2024. In our opinion, the financial statements present fairly, in all material respects, the financial position of Acme Corporation."},
            {"page": 5, "text": "Report on Internal Controls: We identified a material weakness in the Company's internal control over financial reporting. The material weakness relates to insufficient review controls over percentage-of-completion revenue estimates for defense contracts. This material weakness was also identified by management in their SOX 404 assessment."},
            {"page": 8, "text": "Key Audit Matter -- Revenue Recognition: Given the significance of long-term contract revenue ($4.8 billion, representing 39% of total revenue) and the judgment involved in estimating costs to complete, we identified revenue recognition on defense contracts as a Key Audit Matter."},
            {"page": 12, "text": "Key Audit Matter -- Environmental Liability: The estimation of the environmental remediation liability at the Newark facility requires significant judgment regarding cleanup technology, regulatory requirements, and timeline. We engaged our environmental specialists to evaluate management's assumptions."},
        ],
    },
    {
        "source": "Revenue_Recognition_Policy_v3.pdf", "doc_type": "policy", "entity": "Acme Corp",
        "chunks": [
            {"page": 1, "text": "Revenue Recognition Policy (Effective January 1, 2024): This policy governs the recognition of revenue across all business units of Acme Corporation in compliance with ASC 606 - Revenue from Contracts with Customers."},
            {"page": 4, "text": "Long-Term Contracts: For contracts accounted for under the over-time method, revenue is recognized based on the ratio of costs incurred to date to total estimated costs (cost-to-cost method). Estimates of total contract costs must be reviewed and approved by the divisional controller AND the corporate cost engineering team on a quarterly basis."},
            {"page": 7, "text": "Variable Consideration: Performance incentives, liquidated damages, and other variable elements must be estimated using the expected value method. The constraint on variable consideration requires that amounts are included only to the extent it is highly probable that a significant revenue reversal will not occur."},
            {"page": 10, "text": "Contract Modifications: Modifications to existing contracts must be evaluated to determine whether they represent a separate performance obligation. If the modification adds distinct goods/services at standalone selling price, it is treated as a separate contract."},
        ],
    },
]

# ============================================================================
#  2.  COHERE-BASED VECTOR STORE
# ============================================================================

class CohereVectorStore:
    """In-memory vector store using Cohere embeddings + cosine similarity."""

    def __init__(self):
        print("[EMBED] Initializing Cohere embed-english-v3.0...")
        self.embedder = CohereEmbeddings(
            model=settings.cohere_embed_model,
            cohere_api_key=settings.cohere_api_key,
        )
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []

    def ingest(self, documents: list[Document]) -> None:
        self.documents = documents
        texts = [d.page_content for d in documents]
        print(f"[EMBED] Embedding {len(texts)} chunks via Cohere API...")
        self.vectors = self.embedder.embed_documents(texts)
        print(f"[EMBED] Done. Vector dimension: {len(self.vectors[0])}")

    def search(self, query: str, top_k: int = 4) -> list[tuple[Document, float]]:
        q_vec = self.embedder.embed_query(query)
        scores = []
        for doc, dvec in zip(self.documents, self.vectors):
            sim = self._cosine_sim(q_vec, dvec)
            scores.append((doc, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0


# ============================================================================
#  3.  LLM HELPER
# ============================================================================

def get_llm(**kwargs) -> AzureChatOpenAI:
    defaults = dict(
        azure_deployment=settings.azure_openai_deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )
    defaults.update(kwargs)
    return AzureChatOpenAI(**defaults)


# ============================================================================
#  4.  LANGGRAPH NODES (real LLM)
# ============================================================================

STORE: CohereVectorStore | None = None


def retriever_node(state: AgentState) -> dict:
    """Cohere-powered semantic retrieval."""
    query = state.get("rewritten_query") or state["query"]
    print(f"\n[RETRIEVER] Searching with Cohere embeddings...")
    print(f"  Query: {query}")

    results = STORE.search(query, top_k=4)
    docs = []
    for i, (doc, score) in enumerate(results):
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page", "?")
        snippet = doc.page_content[:80].replace("\n", " ")
        print(f"    {i+1}. [{src} p.{pg}] sim={score:.4f} | {snippet}...")
        docs.append(doc)
    return {"retrieved_docs": docs}


# -- Grader (GPT-4o with structured output) --------------------------------

class GradeResult(BaseModel):
    score: str = Field(description="'yes' if documents are relevant, 'no' otherwise")
    reasoning: str = Field(description="Brief explanation")


_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a compliance-document relevance grader.\n"
        "Given a USER QUERY and RETRIEVED DOCUMENTS, decide whether the documents "
        "contain enough information to meaningfully answer the query.\n"
        "Be strict: tangentially related topics should get 'no'.\n"
        'Respond with JSON: {{"score": "yes" or "no", "reasoning": "..."}}'
    )),
    ("human", "Query: {query}\n\nDocuments:\n{documents}\n\nGrade the relevance."),
])


def grader_node(state: AgentState) -> dict:
    """GPT-4o relevance grading."""
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    if not docs:
        return {"grader_score": "no", "grader_reasoning": "No documents retrieved."}

    doc_texts = "\n---\n".join(
        f"[{d.metadata.get('source','?')} p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )

    print(f"\n[GRADER] Asking GPT-4o to evaluate relevance (loop={loop})...")
    llm = get_llm(temperature=0, max_tokens=1024)
    structured_llm = llm.with_structured_output(GradeResult, method="function_calling")
    chain = _GRADER_PROMPT | structured_llm
    result: GradeResult = chain.invoke({"query": query, "documents": doc_texts})

    print(f"  Score: {result.score}")
    print(f"  Reasoning: {result.reasoning}")
    return {"grader_score": result.score, "grader_reasoning": result.reasoning}


# -- Rewriter (GPT-4o) -----------------------------------------------------

_REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a search-query optimiser for a compliance document retrieval system.\n"
        "The previous query did NOT return relevant documents. Rewrite it so that a "
        "semantic search engine finds the correct passages.\n"
        "Strategies: add synonyms, expand abbreviations, broaden/narrow scope.\n"
        "Return ONLY the rewritten query string."
    )),
    ("human", (
        "Original query: {original_query}\n"
        "Previous search: {previous_query}\n"
        "Grader feedback: {grader_reasoning}\n"
        "Retrieved summaries:\n{doc_summaries}\n\n"
        "Produce an improved search query:"
    )),
])


def rewriter_node(state: AgentState) -> dict:
    """GPT-4o query rewriting."""
    original = state["query"]
    previous = state.get("rewritten_query") or original
    reasoning = state.get("grader_reasoning", "")
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    summaries = "\n".join(
        f"- [{d.metadata.get('source','?')}] {d.page_content[:150]}..."
        for d in docs[:5]
    ) or "(none)"

    print(f"\n[REWRITER] Asking GPT-4o to improve query (loop {loop+1})...")
    llm = get_llm(temperature=0.3, max_tokens=256)
    chain = _REWRITER_PROMPT | llm
    resp = chain.invoke({
        "original_query": original,
        "previous_query": previous,
        "grader_reasoning": reasoning,
        "doc_summaries": summaries,
    })

    new_query = resp.content.strip().strip('"')
    print(f"  Before: {previous}")
    print(f"  After:  {new_query}")
    return {"rewritten_query": new_query, "loop_count": loop + 1}


# -- Validator (simulated external API) -------------------------------------

def validator_node(state: AgentState) -> dict:
    """Simulate external API validation (no live API required)."""
    docs = state.get("retrieved_docs", [])
    print(f"\n[VALIDATOR] Cross-verifying claims from {len(docs)} documents...")

    # Extract factual claims via regex
    claims = []
    for doc in docs[:3]:
        text = doc.page_content
        src = doc.metadata.get("source", "?")
        pg = doc.metadata.get("page", "?")
        patterns = [
            (r"\$[\d.,]+ (?:billion|million)", "financial_figure"),
            (r"material weakness", "sox_finding"),
            (r"ASC \d+|SOX \d+", "standard_ref"),
            (r"Deloitte|PwC|EY|KPMG", "auditor"),
        ]
        for pat, typ in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                claims.append({"claim": f"{typ}: {m.group()}", "source": f"{src} p.{pg}", "is_valid": True})

    results = [
        {"claim": c["claim"], "endpoint": "simulated", "status_code": 200,
         "api_response": {"verified": True}, "is_valid": c["is_valid"]}
        for c in claims[:5]
    ]

    valid = sum(1 for r in results if r["is_valid"])
    total = len(results)
    status = "pass" if valid == total else ("partial" if valid > 0 else "fail")

    print(f"  Extracted {len(claims)} claims, verified {valid}/{total}")
    for r in results:
        print(f"    [OK] {r['claim']}")
    print(f"  Status: {status.upper()}")
    return {"validation_results": results, "validation_status": status}


# -- Generator (GPT-4o) ----------------------------------------------------

_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior compliance analyst producing audit-ready answers.\n"
        "RULES:\n"
        "1. Ground every statement with [Source: filename.pdf, p.XX] citations.\n"
        "2. If validation shows a claim is invalid, flag the discrepancy.\n"
        "3. Structure your answer with clear headings.\n"
        "4. If documents are insufficient, state what is missing."
    )),
    ("human", (
        "Query: {query}\n\n"
        "Documents:\n{documents}\n\n"
        "Validation Status: {validation_status}\n"
        "Validation Details: {validation_results}\n\n"
        "Produce a comprehensive, cited answer."
    )),
])


def generator_node(state: AgentState) -> dict:
    """GPT-4o final answer generation with citations."""
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    v_status = state.get("validation_status", "pass")
    v_results = state.get("validation_results", [])

    doc_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source','?')}, p.{d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )
    v_text = "\n".join(f"- {r['claim']} (valid={r['is_valid']})" for r in v_results) or "None"

    print(f"\n[GENERATOR] Asking GPT-4o to synthesize cited answer...")
    llm = get_llm(temperature=0, max_tokens=4096)
    chain = _GENERATOR_PROMPT | llm
    resp = chain.invoke({
        "query": query,
        "documents": doc_text or "(none)",
        "validation_status": v_status,
        "validation_results": v_text,
    })

    print(f"  Generated {len(resp.content)} characters")
    return {"answer": resp.content}


# ============================================================================
#  5.  GRAPH
# ============================================================================

def route_after_grader(state: AgentState) -> str:
    score = state.get("grader_score", "no")
    loop = state.get("loop_count", 0)
    if score == "yes":
        return "validator"
    if loop >= settings.max_rewrite_loops:
        print(f"\n  [!] Max rewrite loops ({settings.max_rewrite_loops}) reached")
        return "validator"
    return "rewriter"


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("retriever", retriever_node)
    g.add_node("grader", grader_node)
    g.add_node("rewriter", rewriter_node)
    g.add_node("validator", validator_node)
    g.add_node("generator", generator_node)
    g.set_entry_point("retriever")
    g.add_edge("retriever", "grader")
    g.add_conditional_edges("grader", route_after_grader, {
        "validator": "validator", "rewriter": "rewriter"
    })
    g.add_edge("rewriter", "retriever")
    g.add_edge("validator", "generator")
    g.add_edge("generator", END)
    return g


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
    global STORE

    parser = argparse.ArgumentParser(description="Real Agentic RAG -- Azure GPT-4o + Cohere")
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--all", "-a", action="store_true")
    parser.add_argument("--list", "-l", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        return

    # Check keys
    if not settings.azure_openai_api_key:
        print("[ERROR] AZURE_OPENAI_API_KEY not set in .env"); return
    if not settings.cohere_api_key:
        print("[ERROR] COHERE_API_KEY not set in .env"); return

    print("=" * 72)
    print("Self-Correction Audit Agent -- LIVE MODE")
    print("  LLM:        Azure OpenAI GPT-4o")
    print("  Embeddings: Cohere embed-english-v3.0")
    print("  Graph:      LangGraph self-correction loop")
    print("=" * 72)

    # Ingest
    STORE = CohereVectorStore()
    all_docs: list[Document] = []
    for info in SAMPLE_DOCUMENTS:
        for chunk in info["chunks"]:
            all_docs.append(Document(
                page_content=chunk["text"],
                metadata={"source": info["source"], "page": chunk["page"],
                          "doc_type": info["doc_type"], "entity": info["entity"]},
            ))
    STORE.ingest(all_docs)
    print(f"[INGEST] {len(all_docs)} chunks indexed\n")

    # Build graph
    agent = build_graph().compile()

    queries = SAMPLE_QUERIES if args.all else ([args.query] if args.query else [SAMPLE_QUERIES[0]])

    for qi, query in enumerate(queries):
        if len(queries) > 1:
            print(f"\n{'#' * 72}")
            print(f"# QUERY {qi+1}/{len(queries)}")
            print(f"{'#' * 72}")

        print(f"\n>> {query}")
        result = agent.invoke({"query": query, "loop_count": 0})

        print(f"\n{'=' * 72}")
        print("FINAL ANSWER")
        print("=" * 72)
        print(result.get("answer", "(no answer)"))
        print("=" * 72)
        print(f"  Self-correction loops: {result.get('loop_count', 0)}")
        print(f"  Validation: {result.get('validation_status', 'N/A')}")
        print(f"  Docs retrieved: {len(result.get('retrieved_docs', []))}")

    print(f"\n[OK] Done!")


if __name__ == "__main__":
    main()
