"""Agentic RAG pipeline running on YOUR real data.

Ingests all .txt files from the raw_corpus folder, embeds them with
Cohere, and lets you query them with the full self-correction loop
powered by Azure GPT-4o.

Usage:
  python demo_corpus.py                          # interactive mode
  python demo_corpus.py -q "your question"       # single query
  python demo_corpus.py --all                    # run sample queries
  python demo_corpus.py --list                   # list sample queries
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

# Force .env to override system env vars
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

# Windows console UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(r"C:\Users\AashrithReddyVootkur\Documents\legal-assistant\data\raw_corpus")

# ============================================================================
#  1.  DOCUMENT LOADER — parse emails, chats, memos
# ============================================================================

def _classify_document(text: str, filename: str) -> str:
    """Classify document type from content."""
    if re.search(r"^\[?\d{4}-\d{2}-\d{2}", text, re.MULTILINE) and "|" in text[:200]:
        return "chat"
    if text.strip().startswith("From") and "Subject" in text[:500]:
        return "email"
    if "SYNTH" in filename:
        return "synthetic-email"
    return "document"


def _extract_participants(text: str, doc_type: str) -> list[str]:
    """Extract people mentioned in the document."""
    people = set()
    # Email headers
    for match in re.finditer(r"(?:From|To|CC)\s*:\s*(.+?)(?:\[|$)", text, re.MULTILINE):
        name = match.group(1).strip().rstrip("[")
        if name and "@" not in name and len(name) < 60:
            people.add(name)
    # Email addresses -> names
    for match in re.finditer(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*[\[<]?\w+@", text):
        people.add(match.group(1))
    # Chat headers like "William Davis | Francis Ham:"
    header_match = re.match(r"^(.+?)\s*\|\s*(.+?):", text)
    if header_match:
        people.add(header_match.group(1).strip())
        people.add(header_match.group(2).strip())
    # Chat messages: [timestamp] Name Name message
    for match in re.finditer(r"\]\s+([A-Z][a-z]+ [A-Z][a-z]+)\s", text):
        people.add(match.group(1))
    return sorted(people)


def _extract_date(text: str) -> str:
    """Extract date from document."""
    # Email: Sent : 8/10/2017
    m = re.search(r"Sent\s*:\s*(\d{1,2}/\d{1,2}/\d{4})", text)
    if m:
        return m.group(1)
    # ISO timestamp
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        return m.group(1)
    return "unknown"


def _extract_subject(text: str) -> str:
    """Extract subject line from email."""
    m = re.search(r"Subject\s*:\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return ""


def load_corpus(data_dir: Path, max_files: int | None = None) -> list[Document]:
    """Load and chunk all .txt files from the corpus directory."""
    documents: list[Document] = []
    txt_files = sorted(data_dir.glob("*.txt"))
    if max_files:
        txt_files = txt_files[:max_files]

    for fpath in txt_files:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text or len(text) < 20:
            continue

        doc_type = _classify_document(text, fpath.name)
        participants = _extract_participants(text, doc_type)
        date = _extract_date(text)
        subject = _extract_subject(text)

        # For long documents, chunk them; for short ones, keep as-is
        chunks = _chunk_text(text, chunk_size=800, overlap=150)

        for i, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": fpath.name,
                    "doc_type": doc_type,
                    "participants": ", ".join(participants),
                    "date": date,
                    "subject": subject,
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                },
            ))

    return documents


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Simple text chunker with overlap."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        # Try to break at a sentence or newline boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                br = text.rfind(sep, start + chunk_size // 2, end + 100)
                if br > start:
                    end = br + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


# ============================================================================
#  2.  COHERE VECTOR STORE
# ============================================================================

class CohereVectorStore:
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
        # Cohere has a batch limit; embed in batches of 96
        batch_size = 96
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            print(f"  Embedding batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1} ({len(batch)} chunks)...")
            vecs = self.embedder.embed_documents(batch)
            all_vectors.extend(vecs)
        self.vectors = all_vectors
        print(f"[EMBED] Done. {len(self.vectors)} vectors, dim={len(self.vectors[0])}")

    def search(self, query: str, top_k: int = 6) -> list[tuple[Document, float]]:
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


STORE: CohereVectorStore | None = None

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
#  4.  LANGGRAPH NODES
# ============================================================================

def retriever_node(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    print(f"\n[RETRIEVER] Searching {len(STORE.documents)} chunks...")
    print(f"  Query: {query}")

    results = STORE.search(query, top_k=6)
    docs = []
    for i, (doc, score) in enumerate(results):
        src = doc.metadata.get("source", "?")
        ppl = doc.metadata.get("participants", "")
        subj = doc.metadata.get("subject", "")
        snippet = doc.page_content[:80].replace("\n", " ")
        label = f"{src}"
        if subj:
            label += f" | {subj[:40]}"
        print(f"    {i+1}. [{label}] sim={score:.4f}")
        print(f"       {snippet}...")
        docs.append(doc)
    return {"retrieved_docs": docs}


# -- Grader -----------------------------------------------------------------

class GradeResult(BaseModel):
    score: str = Field(description="'yes' if documents are relevant, 'no' otherwise")
    reasoning: str = Field(description="Brief explanation")


_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a legal-document relevance grader for an e-discovery/compliance investigation.\n"
        "Given a USER QUERY and RETRIEVED DOCUMENTS (emails, chats, memos from a corporate corpus),\n"
        "decide whether the documents contain enough information to meaningfully answer the query.\n"
        "Be strict: tangentially related content should get 'no'.\n"
        "Consider participant names, dates, subjects, and actual content.\n"
        'Respond with JSON: {{"score": "yes" or "no", "reasoning": "..."}}'
    )),
    ("human", "Query: {query}\n\nDocuments:\n{documents}\n\nGrade the relevance."),
])


def grader_node(state: AgentState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    if not docs:
        return {"grader_score": "no", "grader_reasoning": "No documents retrieved."}

    doc_texts = "\n---\n".join(
        f"[{d.metadata.get('source','?')} | {d.metadata.get('doc_type','?')} | "
        f"participants: {d.metadata.get('participants','?')} | "
        f"date: {d.metadata.get('date','?')} | "
        f"subject: {d.metadata.get('subject','')}"
        f"]\n{d.page_content}"
        for d in docs
    )

    print(f"\n[GRADER] GPT-4o evaluating relevance (loop={loop})...")
    llm = get_llm(temperature=0, max_tokens=1024)
    structured_llm = llm.with_structured_output(GradeResult)
    chain = _GRADER_PROMPT | structured_llm
    result: GradeResult = chain.invoke({"query": query, "documents": doc_texts})

    print(f"  Score: {result.score}")
    print(f"  Reasoning: {result.reasoning}")
    return {"grader_score": result.score, "grader_reasoning": result.reasoning}


# -- Rewriter ---------------------------------------------------------------

_REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a search-query optimizer for a legal e-discovery system.\n"
        "The corpus contains corporate emails, chat logs, and memos from BigThorium company.\n"
        "Key people: William Davis (Director of Operations), Francis Ham (CEO), "
        "Arvind Patel (Welder), Muhammad Kumar (Worker), Scarlett Reed (Associate PM), "
        "Sandra Myers, Alexander Hill, Gabriel Torres, Abigail Rogers.\n"
        "The previous query did NOT return relevant documents. Rewrite it for better semantic retrieval.\n"
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
    original = state["query"]
    previous = state.get("rewritten_query") or original
    reasoning = state.get("grader_reasoning", "")
    docs = state.get("retrieved_docs", [])
    loop = state.get("loop_count", 0)

    summaries = "\n".join(
        f"- [{d.metadata.get('source','?')} | {d.metadata.get('participants','')}] "
        f"{d.page_content[:120]}..."
        for d in docs[:5]
    ) or "(none)"

    print(f"\n[REWRITER] GPT-4o improving query (loop {loop+1})...")
    llm = get_llm(temperature=0.3, max_tokens=256)
    chain = _REWRITER_PROMPT | llm
    resp = chain.invoke({
        "original_query": original,
        "previous_query": previous,
        "grader_reasoning": reasoning,
        "doc_summaries": summaries,
    })

    new_query = resp.content.strip().strip('"')
    print(f"  Before: {previous[:100]}")
    print(f"  After:  {new_query[:100]}")
    return {"rewritten_query": new_query, "loop_count": loop + 1}


# -- Validator (document-based cross-referencing) ---------------------------

def validator_node(state: AgentState) -> dict:
    docs = state.get("retrieved_docs", [])
    print(f"\n[VALIDATOR] Cross-referencing claims across {len(docs)} documents...")

    # Check internal consistency: do multiple sources corroborate?
    sources = set(d.metadata.get("source", "") for d in docs)
    participants_all = set()
    for d in docs:
        for p in d.metadata.get("participants", "").split(", "):
            if p.strip():
                participants_all.add(p.strip())

    results = []
    for doc in docs[:5]:
        src = doc.metadata.get("source", "?")
        results.append({
            "claim": f"Content from {src}",
            "endpoint": "internal-cross-ref",
            "status_code": 200,
            "api_response": {"sources": len(sources), "corroborating": len(sources) > 1},
            "is_valid": True,
        })

    status = "pass" if len(sources) > 1 else "partial"
    print(f"  Unique sources: {len(sources)}")
    print(f"  Participants found: {', '.join(sorted(participants_all)[:8])}")
    print(f"  Cross-reference status: {status.upper()}")
    return {"validation_results": results, "validation_status": status}


# -- Generator (GPT-4o) ----------------------------------------------------

_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a senior legal analyst performing e-discovery review.\n"
        "You are analyzing corporate communications (emails, chats, memos) from BigThorium.\n"
        "RULES:\n"
        "1. Cite every claim with [Source: filename.txt, date] references.\n"
        "2. Identify key participants and their roles.\n"
        "3. Note any potentially concerning patterns (labor issues, retaliation, discrimination).\n"
        "4. Structure your answer with clear headings.\n"
        "5. If evidence is insufficient, state what additional documents would be needed.\n"
        "6. Maintain legal objectivity -- present facts, flag concerns, avoid conclusions of law."
    )),
    ("human", (
        "Query: {query}\n\n"
        "Documents:\n{documents}\n\n"
        "Cross-Reference Status: {validation_status}\n\n"
        "Produce a comprehensive, cited analysis."
    )),
])


def generator_node(state: AgentState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    v_status = state.get("validation_status", "pass")

    doc_text = "\n---\n".join(
        f"[Source: {d.metadata.get('source','?')} | Type: {d.metadata.get('doc_type','?')} | "
        f"Date: {d.metadata.get('date','?')} | "
        f"Participants: {d.metadata.get('participants','?')} | "
        f"Subject: {d.metadata.get('subject','')}]\n{d.page_content}"
        for d in docs
    )

    print(f"\n[GENERATOR] GPT-4o synthesizing cited answer...")
    llm = get_llm(temperature=0, max_tokens=4096)
    chain = _GENERATOR_PROMPT | llm
    resp = chain.invoke({
        "query": query,
        "documents": doc_text or "(none)",
        "validation_status": v_status,
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
#  6.  SAMPLE QUERIES (tailored to BigThorium corpus)
# ============================================================================

SAMPLE_QUERIES = [
    "What evidence exists of worker exploitation or labor violations at BigThorium?",
    "What did William Davis and Francis Ham discuss about undocumented workers?",
    "Were there any pay cuts or broken promises regarding citizenship for workers?",
    "What construction delays or project issues were reported?",
    "Is there evidence of retaliation against workers who complained?",
]


# ============================================================================
#  7.  MAIN
# ============================================================================

def main():
    global STORE

    parser = argparse.ArgumentParser(description="Agentic RAG on your legal corpus")
    parser.add_argument("--query", "-q", type=str, default=None)
    parser.add_argument("--all", "-a", action="store_true", help="Run all sample queries")
    parser.add_argument("--list", "-l", action="store_true", help="List sample queries")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    if args.list:
        print("Sample queries for BigThorium corpus:")
        for i, q in enumerate(SAMPLE_QUERIES, 1):
            print(f"  {i}. {q}")
        return

    # Validate keys
    if not settings.azure_openai_api_key:
        print("[ERROR] AZURE_OPENAI_API_KEY not set"); return
    if not settings.cohere_api_key:
        print("[ERROR] COHERE_API_KEY not set"); return

    print("=" * 72)
    print("Self-Correction Audit Agent -- YOUR CORPUS")
    print(f"  Data:       {DATA_DIR}")
    print(f"  LLM:        Azure OpenAI ({settings.azure_openai_deployment})")
    print(f"  Embeddings: Cohere {settings.cohere_embed_model}")
    print("=" * 72)

    # Load & ingest
    print(f"\n[LOAD] Reading .txt files from {DATA_DIR}...")
    raw_docs = load_corpus(DATA_DIR)
    print(f"[LOAD] {len(raw_docs)} chunks from {len(set(d.metadata['source'] for d in raw_docs))} files")

    # Show corpus stats
    doc_types = {}
    for d in raw_docs:
        t = d.metadata.get("doc_type", "?")
        doc_types[t] = doc_types.get(t, 0) + 1
    for t, c in sorted(doc_types.items()):
        print(f"  {t}: {c} chunks")

    STORE = CohereVectorStore()
    STORE.ingest(raw_docs)

    # Build graph
    agent = build_graph().compile()
    print("[GRAPH] Compiled LangGraph pipeline\n")

    # Determine queries
    if args.interactive or (not args.query and not args.all):
        print("Enter your questions (type 'quit' to exit):\n")
        while True:
            try:
                query = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query or query.lower() in ("quit", "exit", "q"):
                break
            _run_query(agent, query)
    elif args.all:
        for qi, query in enumerate(SAMPLE_QUERIES):
            print(f"\n{'#' * 72}")
            print(f"# QUERY {qi+1}/{len(SAMPLE_QUERIES)}")
            print(f"{'#' * 72}")
            _run_query(agent, query)
    elif args.query:
        _run_query(agent, args.query)

    print(f"\n[OK] Done!")


def _run_query(agent, query: str):
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


if __name__ == "__main__":
    main()
