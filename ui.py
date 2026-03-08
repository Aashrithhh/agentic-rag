"""Streamlit Demo UI for the Self-Correction Audit Agent.

Launch:
    streamlit run ui.py
"""

from __future__ import annotations

import os
import threading
import time
import logging
import shlex
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Agentic RAG - Audit Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Auth gate (optional password protection) ─────────────────────────


FLAGGED_EXPORT_DIR = Path("exports/flagged")


def _export_flagged_pdf(
    case_id: str,
    filename: str,
    query_text: str,
    docs: list[dict],
) -> Path:
    """Write a readable PDF of flagged documents to the local exports folder.

    Returns the Path to the written PDF file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        )
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
    except ImportError as exc:
        raise ImportError(
            "The 'reportlab' package is required for PDF export. "
            "Install it with: pip install reportlab"
        ) from exc

    out_dir = FLAGGED_EXPORT_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{filename}.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["Normal"]
    meta_style = ParagraphStyle(
        "Meta", parent=normal_style, fontSize=8, textColor=colors.grey,
    )
    content_style = ParagraphStyle(
        "Content", parent=normal_style, fontSize=9, leading=13,
        spaceAfter=6, alignment=TA_LEFT,
    )

    story: list = []

    # ── Title page ──
    story.append(Paragraph(f"Flagged Documents — {filename}", title_style))
    story.append(Spacer(1, 12))

    summary_data = [
        ["Case", case_id],
        ["Saved set", filename],
        ["Query", query_text[:200] if query_text else "(none)"],
        ["Document count", str(len(docs))],
        ["Exported at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    t = Table(summary_data, colWidths=[1.5 * inch, 4.5 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 24))

    # ── Each flagged document ──
    for idx, d in enumerate(docs, 1):
        meta = d.get("metadata", {})
        source = d.get("source", meta.get("source", "unknown"))
        subject = meta.get("email_subject", "")
        sender = meta.get("email_from", "")
        date = meta.get("email_date", "")

        header = f"Document {idx}"
        if subject:
            header += f" — {subject}"
        story.append(Paragraph(header, heading_style))

        # Metadata block
        meta_lines = []
        if sender:
            meta_lines.append(f"<b>From:</b> {sender}")
        if meta.get("email_to"):
            meta_lines.append(f"<b>To:</b> {meta['email_to']}")
        if date:
            meta_lines.append(f"<b>Date:</b> {date}")
        meta_lines.append(f"<b>Source:</b> {source}")
        if meta.get("chunk_index") is not None:
            meta_lines.append(f"<b>Chunk:</b> {meta['chunk_index']}")
        if meta.get("page") is not None:
            meta_lines.append(f"<b>Page:</b> {meta['page']}")

        for ml in meta_lines:
            story.append(Paragraph(ml, meta_style))
        story.append(Spacer(1, 8))

        # Content — escape XML entities for reportlab
        content = d.get("content", "(no content)")
        content_safe = (
            content.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )
        story.append(Paragraph(content_safe, content_style))
        story.append(Spacer(1, 16))

        if idx < len(docs):
            story.append(PageBreak())

    doc.build(story)
    return pdf_path


# ── Natural-language save command parsing ────────────────────────────────

import re as _re_cmd
from collections import OrderedDict as _OD_cmd

_SAVE_CMD_RE = _re_cmd.compile(
    r"^\s*save\s+"
    r"(?:docs?(?:uments?)?)\s+"
    r"(.+?)"
    r"\s+as\s+"
    r"([\w\-. ]+?)"
    r"\s*$",
    _re_cmd.IGNORECASE,
)
_RANGE_RE = _re_cmd.compile(r"(\d+)\s*[-\u2013\u2014]\s*(\d+)")
_SINGLE_RE = _re_cmd.compile(r"\d+")


def _parse_save_command(text: str, max_card: int):
    """Regex parser for save commands. Returns (doc_numbers, filename) or None."""
    m = _SAVE_CMD_RE.match(text)
    if not m:
        return None
    spec, filename = m.group(1), m.group(2).strip()

    nums: list[int] = []
    remainder = spec
    for rm in _RANGE_RE.finditer(spec):
        lo, hi = int(rm.group(1)), int(rm.group(2))
        if lo <= hi:
            nums.extend(range(lo, hi + 1))
        remainder = remainder.replace(rm.group(0), " ")
    for sm in _SINGLE_RE.finditer(remainder):
        nums.append(int(sm.group(0)))

    nums = sorted(set(nums))
    if not nums or not filename:
        return None
    return (nums, filename)


def _llm_parse_save_command(text: str, max_card: int):
    """LLM fallback for parsing save commands regex cannot handle."""
    from pydantic import BaseModel, Field
    from app.llm import get_chat_llm

    class SaveCommandExtraction(BaseModel):
        doc_numbers: list[int] = Field(description="Document numbers to save (1-based)")
        filename: str = Field(description="Filename to save as")
        is_save_command: bool = Field(description="Whether this is a save command")

    llm = get_chat_llm(temperature=0, max_tokens=256)
    chain = llm.with_structured_output(SaveCommandExtraction)
    prompt = (
        f"The user typed a command in a document-saving interface. "
        f"There are {max_card} documents numbered 1 to {max_card}. "
        f"Extract the document numbers and filename from this command. "
        f"If this is not a save command, set is_save_command to false.\n\n"
        f"Command: {text}"
    )
    try:
        result = chain.invoke(prompt)
        if not result.is_save_command:
            return None
        nums = sorted(set(result.doc_numbers))
        if not nums or not result.filename.strip():
            return None
        return (nums, result.filename.strip())
    except Exception:
        return None


def _build_card_docs_map(retrieved_docs: list) -> dict[int, dict]:
    """Build card_num → payload dict mapping from retrieved documents.

    Uses the same ordering as the UI: email groups first, then standalone docs.
    """
    email_grps: OrderedDict[str, list] = OrderedDict()
    standalone: list[tuple[int, object]] = []
    for di, d in enumerate(retrieved_docs):
        ef = d.metadata.get("email_from", "")
        es = d.metadata.get("email_subject", "")
        mid = d.metadata.get("message_id", "")
        if ef or es:
            gk = mid or f"{es}||{ef}"
            if gk not in email_grps:
                email_grps[gk] = []
            email_grps[gk].append((di, d))
        else:
            standalone.append((di, d))

    card_docs: dict[int, dict] = {}
    cn = 0
    for _gk, cl in email_grps.items():
        cn += 1
        cl.sort(key=lambda t: t[1].metadata.get("chunk_index") or 0)
        merged_content = "\n\n".join(c.page_content for _, c in cl)
        first = cl[0][1]
        card_docs[cn] = {
            "source": first.metadata.get("source", ""),
            "page": first.metadata.get("page"),
            "chunk_index": first.metadata.get("chunk_index"),
            "content": merged_content[:5000],
            "metadata": {k: str(v) if v is not None else None for k, v in first.metadata.items()},
        }
    for _, sd in standalone:
        cn += 1
        card_docs[cn] = {
            "source": sd.metadata.get("source", ""),
            "page": sd.metadata.get("page"),
            "chunk_index": sd.metadata.get("chunk_index"),
            "content": sd.page_content[:5000],
            "metadata": {k: str(v) if v is not None else None for k, v in sd.metadata.items()},
        }
    return card_docs


def _check_auth() -> None:
    """Block the UI with a password prompt when auth is enabled."""
    import hashlib as _hashlib
    import hmac as _hmac
    import time as _time

    from app.config import settings as _cfg
    if not _cfg.ui_auth_enabled:
        return

    _SESSION_TTL = 4 * 3600  # 4-hour session expiry

    if st.session_state.get("authenticated"):
        login_time = st.session_state.get("auth_time", 0)
        if (_time.time() - login_time) < _SESSION_TTL:
            return
        st.session_state["authenticated"] = False
        st.warning("Session expired. Please log in again.")

    st.title("Login Required")
    pwd = st.text_input("Password", type="password", key="auth_pwd")
    if st.button("Login"):
        import logging as _log
        _auth_logger = _log.getLogger("app.ui.auth")
        # Timing-safe comparison via HMAC on SHA-256 hashes
        pwd_hash = _hashlib.sha256(pwd.encode()).hexdigest()
        expected_hash = _hashlib.sha256(_cfg.ui_auth_password.encode()).hexdigest()
        if _hmac.compare_digest(pwd_hash, expected_hash):
            st.session_state["authenticated"] = True
            st.session_state["auth_time"] = _time.time()
            _auth_logger.info("UI login successful")
            st.rerun()
        else:
            _auth_logger.warning("UI login failed — invalid password")
            st.error("Invalid password.")
    st.stop()


_check_auth()

# ── Imports from the app package ────────────────────────────────────
from app.graph import compile_graph
from app.cache import cache as app_cache
from app.config import settings
from app.db import init_db
from app.state import AgentState
from app.cases import CASES, list_cases, get_case, metadata_filter_for_case, auto_ingest_cases
from app.integrations.mcp_client import MCPOpsClient, MCPClientError


# ── Logging (suppress noisy libs in the UI) ─────────────────────────
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Auto-ingest disabled (large blob cases block the UI) ────────────
# Ingestion is now handled via the Data Management tab below.
if "auto_ingested" not in st.session_state:
    st.session_state["auto_ingested"] = True


# ─────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
.block-container { max-width: 1100px; }

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    text-align: center;
}
.hero h1 { margin: 0; font-size: 2.4rem; font-weight: 700; letter-spacing: -0.5px; }
.hero p  { margin: 0.4rem 0 0; font-size: 1.05rem; opacity: 0.85; }

/* ── Pipeline step cards ── */
.step-card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    background: #fafafa;
    transition: box-shadow 0.2s;
}
.step-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,0.08); }
.step-card.active {
    border-color: #4a9eff;
    background: #f0f7ff;
    box-shadow: 0 0 0 2px rgba(74,158,255,0.25);
}
.step-card.done {
    border-color: #34c759;
    background: #f0faf3;
}
.step-card.fail {
    border-color: #ff3b30;
    background: #fff5f5;
}
.step-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.4rem;
}
.step-detail { font-size: 0.88rem; color: #555; line-height: 1.5; }

/* ── Answer box ── */
.answer-box {
    border: 2px solid #4a9eff;
    border-radius: 14px;
    padding: 1.5rem;
    margin-top: 1rem;
    background: linear-gradient(160deg, #f6faff 0%, #ffffff 100%);
}

/* ── Metric pill ── */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 0.8rem 0; }
.metric-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: #f0f0f0;
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.85rem;
    font-weight: 500;
}

/* ── Source chip ── */
.source-chip {
    display: inline-block;
    background: #e8f4fd;
    border: 1px solid #b8ddf0;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.8rem;
    margin: 0.15rem 0.2rem;
    color: #1a6fa0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🔍 Agentic RAG &mdash; Audit Agent</h1>
    <p>Self-correcting retrieval-augmented generation for compliance & legal discovery</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────


def _call_mcp_tool(tool_name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    if not settings.mcp_ops_enabled:
        return {"error": "MCP Ops is disabled by configuration."}

    try:
        cmd = settings.mcp_ops_command
        tool_args = shlex.split(settings.mcp_ops_args, posix=False)
        client = MCPOpsClient(command=cmd, args=tool_args)
        return client.call_tool(tool_name, args)
    except MCPClientError as exc:
        return {"error": str(exc)}
    except Exception as exc:
        return {"error": f"MCP call failed: {exc}"}

with st.sidebar:
    # ── Case selector ──────────────────────────────────────────────
    st.markdown("### 📂 Case Selector")
    case_options = list_cases()
    case_labels = [c.display_name for c in case_options]
    case_ids    = [c.case_id for c in case_options]

    selected_idx = st.radio(
        "Choose a case context",
        range(len(case_options)),
        format_func=lambda i: f"{case_labels[i]}",
        key="case_radio",
        label_visibility="collapsed",
    )
    selected_case = case_options[selected_idx]

    st.info(f"**{selected_case.display_name}**  \n{selected_case.description}", icon="🔒")
    st.caption(
        f"Isolated DB: `audit_rag_{selected_case.case_id.replace('-', '_')}`  \n"
        f"Filter: `doc_type={selected_case.doc_type}`"
        + (f", `entity={selected_case.entity_name}`" if selected_case.entity_name else "")
    )

    st.markdown("---")
    st.markdown("### 📊 Pipeline Architecture")
    st.markdown("""
    ```
    START
      │
      ▼
    Retriever   ← hybrid HNSW + GIN
      │
      ▼
    Grader      ← structured yes / no
      │
      ├─ yes ──► Validator
      │              │
      │              ▼
      │          Generator ──► END
      │
      └─ no ──► Rewriter ──► Retriever
                 (loops up to 3×)
    ```
    """)

    st.markdown("---")
    st.markdown("### 💡 Sample questions")
    for q in selected_case.sample_questions:
        if st.button(q, key=f"sample_{hash(q)}_{selected_case.case_id}", use_container_width=True):
            st.session_state["prefill_query"] = q
            st.rerun()


    st.markdown("---")
    st.markdown("### MCP Ops Diagnostics")

    if settings.mcp_ops_enabled:
        if st.button("Run DB Health", use_container_width=True):
            db_status = _call_mcp_tool("case_db_status", {"case_id": selected_case.case_id})
            vec_status = _call_mcp_tool("pgvector_status", {"case_id": selected_case.case_id})
            st.session_state["mcp_diag_db"] = {"db": db_status, "pgvector": vec_status}

        if st.button("Check pgvector + indexes", use_container_width=True):
            idx = _call_mcp_tool("vector_index_status", {"case_id": selected_case.case_id})
            st.session_state["mcp_diag_idx"] = idx

        mcp_preview_query = st.text_input("Preview Query", value=st.session_state.get("query_text", ""), key="mcp_preview_query")
        if st.button("Retrieve Preview", use_container_width=True):
            st.session_state["mcp_diag_preview"] = _call_mcp_tool(
                "retrieve_preview",
                {"case_id": selected_case.case_id, "query": mcp_preview_query or "", "top_k": 6},
            )

        if st.button("Trace Query", use_container_width=True):
            st.session_state["mcp_diag_trace"] = _call_mcp_tool(
                "trace_query",
                {"case_id": selected_case.case_id, "query": mcp_preview_query or "", "max_steps": 20},
            )

        sql_default = "SELECT source, COUNT(*) AS cnt FROM document_chunks GROUP BY source ORDER BY cnt DESC"
        mcp_sql = st.text_area("Read-only SQL", value=sql_default, key="mcp_readonly_sql", height=90)
        if st.button("Run Read-only SQL", use_container_width=True):
            st.session_state["mcp_diag_sql"] = _call_mcp_tool(
                "execute_readonly_postgres_sql",
                {"case_id": selected_case.case_id, "sql": mcp_sql, "limit": 200},
            )

        if "mcp_diag_db" in st.session_state:
            with st.expander("DB Health Result", expanded=False):
                st.json(st.session_state["mcp_diag_db"])
        if "mcp_diag_idx" in st.session_state:
            with st.expander("Index Result", expanded=False):
                st.json(st.session_state["mcp_diag_idx"])
        if "mcp_diag_preview" in st.session_state:
            with st.expander("Retrieve Preview Result", expanded=False):
                st.json(st.session_state["mcp_diag_preview"])
        if "mcp_diag_trace" in st.session_state:
            with st.expander("Trace Result", expanded=False):
                st.json(st.session_state["mcp_diag_trace"])
        if "mcp_diag_sql" in st.session_state:
            with st.expander("Read-only SQL Result", expanded=False):
                st.json(st.session_state["mcp_diag_sql"])
    else:
        st.caption("MCP Ops diagnostics disabled via settings.")

    # ── Cache Diagnostics ──────────────────────────────────────
    st.markdown("---")
    st.markdown("### Cache Diagnostics")
    cache_stats = app_cache.stats()

    if cache_stats["enabled"]:
        ttl_label = "process-lifetime" if cache_stats["ttl_seconds"] == 0 else f"{cache_stats['ttl_seconds']}s"
        st.caption(f"TTL: {ttl_label} | Max/ns: {cache_stats['max_entries_per_namespace']}")

        col_h, col_m, col_r = st.columns(3)
        col_h.metric("Hits", cache_stats["total_hits"])
        col_m.metric("Misses", cache_stats["total_misses"])
        col_r.metric("Hit Rate", cache_stats["total_hit_rate"])

        with st.expander("Per-namespace details", expanded=False):
            for ns_name, ns_stats in cache_stats.get("namespaces", {}).items():
                st.markdown(
                    f"**{ns_name}** — {ns_stats['size']}/{ns_stats['max_entries']} entries, "
                    f"{ns_stats['hits']} hits / {ns_stats['misses']} misses "
                    f"({ns_stats['hit_rate']})"
                )

        if st.button("Clear all caches", use_container_width=True):
            removed = app_cache.invalidate_all()
            st.success(f"Cleared {removed} cache entries.")
    else:
        st.caption("Caching is disabled (CACHE_ENABLED=false).")


# ─────────────────────────────────────────────────────────────────────
# Ingestion helpers (threaded background ingestion for Streamlit)
# ─────────────────────────────────────────────────────────────────────

# Thread-safe store for ingestion results (background thread writes here,
# main Streamlit thread reads via _get_ingest_result).
import threading as _threading

_ingest_results_lock = _threading.Lock()
_ingest_results: dict[str, dict] = {}


def _get_chunk_count(case_id: str) -> int:
    """Return number of chunks in a case's database."""
    try:
        from sqlalchemy import text, select, func
        eng = init_db(case_id)
        from app.db import document_chunks
        with eng.connect() as conn:
            return conn.execute(
                select(func.count()).select_from(document_chunks)
            ).scalar() or 0
    except Exception:
        return 0


def _run_ingest_thread(case_id: str, source_path: str, result_key: str) -> None:
    """Run ingestion in a background thread so Streamlit doesn't freeze.

    Writes outcome to a thread-safe store — never touches st.session_state directly.
    """
    with _ingest_results_lock:
        _ingest_results[result_key] = {"status": "running", "message": "Ingestion in progress..."}
    try:
        from app.ingest import ingest_case
        result = ingest_case(case_id, source_path)
        with _ingest_results_lock:
            _ingest_results[result_key] = {"status": "completed", "result": result}
    except Exception as exc:
        with _ingest_results_lock:
            _ingest_results[result_key] = {"status": "failed", "error": str(exc)}


def _get_ingest_result(result_key: str) -> dict | None:
    """Poll the thread-safe store and transfer result to session_state.

    Called from the main Streamlit thread only.
    """
    with _ingest_results_lock:
        result = _ingest_results.pop(result_key, None)
    if result is not None:
        st.session_state[result_key] = result
    return result


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


NODE_META: dict[str, dict[str, str]] = {
    "retriever":           {"icon": "📥", "label": "Retriever",           "desc": "Hybrid semantic + keyword search via Cohere & pgvector"},
    "grader":              {"icon": "⚖️",  "label": "Grader",             "desc": "LLM-based binary relevance check (GPT-4o)"},
    "rewriter":            {"icon": "✏️",  "label": "Rewriter",           "desc": "Query reformulation for better recall"},
    "validator":           {"icon": "🛡️", "label": "Validator",          "desc": "Cross-verify extracted claims against external APIs"},
    "generator":           {"icon": "📝", "label": "Generator",          "desc": "Audit-ready answer synthesis with citations"},
    "hallucination_guard": {"icon": "🔬", "label": "Hallucination Guard", "desc": "Risk scoring & pass/warn/block policy"},
}


def render_step_card(node: str, data: dict[str, Any], css_class: str = "done"):
    """Render a pipeline step as a styled card."""
    import html as _html
    _esc = _html.escape

    meta = NODE_META.get(node, {"icon": "🔹", "label": node.title(), "desc": ""})
    detail_html = ""

    if node == "retriever":
        docs = data.get("retrieved_docs", [])
        n = len(docs)
        sources = sorted({d.metadata.get("source", "?") for d in docs})
        chips = "".join(f'<span class="source-chip">{_esc(s)}</span>' for s in sources[:8])
        detail_html = f"Retrieved <strong>{n}</strong> document chunks<br/>{chips}"

    elif node == "grader":
        score = data.get("grader_score", "?")
        reasoning = data.get("grader_reasoning", "")
        icon = "✅" if score == "yes" else "❌"
        detail_html = (
            f"Relevance: {icon} <strong>{_esc(score.upper())}</strong><br/>"
            f"<em>{_esc(reasoning[:200])}</em>"
        )
        if score == "no":
            css_class = "fail"

    elif node == "rewriter":
        rq = data.get("rewritten_query", "")
        loop = data.get("loop_count", 0)
        detail_html = f"Loop <strong>{loop}</strong> &mdash; rewritten to:<br/><em>{_esc(rq[:200])}</em>"

    elif node == "validator":
        status = data.get("validation_status", "pass")
        results = data.get("validation_results", [])
        n_claims = len(results)
        n_valid = sum(1 for r in results if r.get("is_valid"))
        badge = {"pass": "✅ PASS", "partial": "⚠️ PARTIAL", "fail": "❌ FAIL"}.get(status, _esc(status))
        detail_html = f"Status: <strong>{badge}</strong> &mdash; {n_valid}/{n_claims} claims verified"
        if status == "fail":
            css_class = "fail"

    elif node == "generator":
        answer = data.get("answer", "")
        detail_html = f"Produced <strong>{len(answer):,}</strong>-character answer"

    elif node == "hallucination_guard":
        h_decision = data.get("hallucination_decision", "pass")
        h_score = data.get("hallucination_score", 0.0)
        badge_map = {"pass": ("✅ PASS", "done"), "warn": ("⚠️ WARN", "fail"), "block": ("🚫 BLOCK", "fail")}
        badge_text, css_class = badge_map.get(h_decision, ("✅ PASS", "done"))
        detail_html = f"Decision: <strong>{badge_text}</strong> &mdash; score: {h_score:.2f}"
        h_flags = data.get("hallucination_flags", [])
        if h_flags:
            detail_html += "<br/>" + "<br/>".join(
                f"• {_esc(f.get('detail', ''))}" for f in h_flags
            )

    st.markdown(f"""
    <div class="step-card {css_class}">
        <div class="step-header">{meta['icon']} {meta['label']}</div>
        <div class="step-detail">{meta['desc']}<br/>{detail_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Tabbed layout — Query | Data Management
# ─────────────────────────────────────────────────────────────────────
tab_query, tab_ingest = st.tabs(["🔎 Query Pipeline", "📦 Data Management"])


# ═════════════════════════════════════════════════════════════════════
# TAB 1: QUERY PIPELINE
# ═════════════════════════════════════════════════════════════════════
with tab_query:
    # Handle prefill from sample-question buttons (must happen BEFORE the widget)
    if "prefill_query" in st.session_state:
        st.session_state["query_text"] = st.session_state.pop("prefill_query")

    query = st.text_area(
        "🔎 Enter your compliance / audit question",
        key="query_text",
        height=90,
        placeholder="e.g. What were the issues with the Indian workers at Big Thorium?",
    )

    # ── Email metadata filters (collapsible) ──────────────
    with st.expander("📧 Email Filters (optional)", expanded=False):
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            email_from_filter = st.text_input(
                "From (sender)", key="filter_email_from",
                placeholder="e.g. sarah.johnson",
            )
            email_date_from = st.date_input(
                "Date from", value=None, key="filter_date_from",
            )
        with filter_col2:
            email_to_filter = st.text_input(
                "To (recipient)", key="filter_email_to",
                placeholder="e.g. compliance@techcorp.com",
            )
            email_date_to = st.date_input(
                "Date to", value=None, key="filter_date_to",
            )
        source_file_filter = st.text_input(
            "Source file", key="filter_source_file",
            placeholder="e.g. email_fraud_investigation.txt",
        )

    col_run, col_clear = st.columns([1, 5])
    with col_run:
        run_clicked = st.button("▶  Run Query", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑  Clear history"):
            st.session_state.pop("steps", None)
            st.session_state.pop("final_answer", None)
            st.session_state.pop("final_state", None)
            # Clear stale flag checkbox state
            for _fk in [k for k in st.session_state if k.startswith("flag_doc_")]:
                del st.session_state[_fk]
            st.session_state.pop("_flag_card_count", None)
            st.session_state.pop("flag_save_filename", None)
            st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # Execute pipeline
    # ─────────────────────────────────────────────────────────────────────
    if run_clicked and query.strip():
        # ── Clear stale flag state from previous query ─────────
        for _fk in [k for k in st.session_state if k.startswith("flag_doc_")]:
            del st.session_state[_fk]
        st.session_state.pop("_flag_card_count", None)
        st.session_state.pop("flag_save_filename", None)

        # ── Input validation ───────────────────────────────────────
        if len(query.strip()) > settings.ui_max_query_length:
            st.error(f"Query too long ({len(query.strip())} chars). Max is {settings.ui_max_query_length}.")
            st.stop()

        # Ensure the per-case database is ready
        init_db(selected_case.case_id)

        # ── Build email metadata filters ─────────────────────
        email_filters: dict[str, str] = {}
        if email_from_filter:
            email_filters["email_from"] = email_from_filter
        if email_to_filter:
            email_filters["email_to"] = email_to_filter
        if email_date_from:
            email_filters["email_date_from"] = str(email_date_from)
        if email_date_to:
            email_filters["email_date_to"] = str(email_date_to)
        if source_file_filter:
            email_filters["source_file"] = source_file_filter

        # Build initial state with case-level isolation
        initial_state: AgentState = {
            "query": query.strip(),
            "loop_count": 0,
            "case_id": selected_case.case_id,
        }
        if email_filters:
            initial_state["metadata_filter"] = email_filters

        agent = compile_graph()

        steps: list[tuple[str, dict]] = []
        final_state: dict = {}

        # ── Streaming execution with live progress ──
        pipeline_container = st.container()
        progress_bar = st.progress(0, text="Starting pipeline...")
        status_placeholder = st.empty()

        step_count = 0
        t_start = time.time()

        for step in agent.stream(initial_state):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            steps.append((node_name, node_output))
            final_state.update(node_output)
            step_count += 1

            # Update progress (estimate ~5 steps typical)
            progress_bar.progress(
                min(step_count / 5, 0.95),
                text=f"Running **{NODE_META.get(node_name, {}).get('label', node_name)}**..."
            )

        elapsed = time.time() - t_start
        progress_bar.progress(1.0, text="Pipeline complete!")
        time.sleep(0.3)
        progress_bar.empty()

        # Store in session
        st.session_state["steps"] = steps
        st.session_state["final_answer"] = final_state.get("answer", "")
        st.session_state["elapsed"] = elapsed
        st.session_state["final_state"] = final_state

    # ─────────────────────────────────────────────────────────────────────
    # Display results
    # ─────────────────────────────────────────────────────────────────────
    if "steps" in st.session_state:
        steps = st.session_state["steps"]
        final_state = st.session_state.get("final_state", {})
        elapsed = st.session_state.get("elapsed", 0)

        # ── Metrics row ──
        n_docs = len(final_state.get("retrieved_docs", []))
        score = final_state.get("grader_score", "?")
        v_status = final_state.get("validation_status", "?")
        loops = final_state.get("loop_count", 0)
        answer_len = len(final_state.get("answer", ""))

        h_decision = final_state.get("hallucination_decision", "pass")
        h_score = final_state.get("hallucination_score", 0.0)
        h_badge_color = {"pass": "#34c759", "warn": "#ff9500", "block": "#ff3b30"}.get(h_decision, "#34c759")
        h_badge_label = h_decision.upper()

        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-pill">📥 {n_docs} docs retrieved</span>
            <span class="metric-pill">⚖️ Grader: {score.upper()}</span>
            <span class="metric-pill">🛡️ Validation: {v_status.upper()}</span>
            <span class="metric-pill">🔄 Rewrites: {loops}</span>
            <span class="metric-pill">⏱️ {elapsed:.1f}s</span>
            <span class="metric-pill">📏 {answer_len:,} chars</span>
            <span class="metric-pill" style="background:{h_badge_color}22;border:1px solid {h_badge_color}">🔬 Risk: {h_badge_label} ({h_score:.2f})</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Two-column layout: pipeline steps + answer ──
        col_pipeline, col_answer = st.columns([2, 3])

        with col_pipeline:
            st.markdown("#### Pipeline Trace")
            for node_name, node_data in steps:
                render_step_card(node_name, {**final_state, **node_data})

        with col_answer:
            st.markdown("#### Answer")
            answer = st.session_state.get("final_answer", "")

            h_decision = final_state.get("hallucination_decision", "pass")
            h_score_val = final_state.get("hallucination_score", 0.0)
            h_flags = final_state.get("hallucination_flags", [])

            if h_decision == "block":
                st.error(
                    f"**Answer Blocked** (hallucination risk score: {h_score_val:.2f})\n\n"
                    "The system detected a high risk of hallucination. "
                    "Please refine your query or consult the source documents directly."
                )
                if h_flags:
                    with st.expander("🔬 Hallucination Risk Details", expanded=True):
                        for hf in h_flags:
                            st.warning(f"**{hf.get('type', 'unknown')}:** {hf.get('detail', '')}")
            elif answer:
                if h_decision == "warn":
                    st.warning(f"**Hallucination Warning** (risk score: {h_score_val:.2f})")

                st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(answer)
                st.markdown('</div>', unsafe_allow_html=True)

                if h_flags:
                    with st.expander("🔬 Hallucination Risk Details", expanded=False):
                        for hf in h_flags:
                            st.info(f"**{hf.get('type', 'unknown')}:** {hf.get('detail', '')}")

                if final_state.get("validation_status") in ("partial", "fail"):
                    st.warning(
                        f"Validation status: **{final_state['validation_status'].upper()}** "
                        "— some claims could not be verified. See discrepancy notes above."
                    )

                critical_facts = final_state.get("critical_fact_assessments", [])
                weak_facts = [a for a in critical_facts if not a.get("is_sufficiently_supported")]
                if weak_facts:
                    with st.expander("⚠️ Insufficient Evidence Warnings", expanded=True):
                        for wf in weak_facts:
                            values = ", ".join(str(v) for v in wf.get("critical_values", []))
                            st.warning(
                                f"**Claim:** {wf['claim']}  \n"
                                f"Critical values: `{values}`  \n"
                                f"Supporting docs: {wf['support_count']}/{wf['min_required']} required  \n"
                                f"Sources: {', '.join(wf.get('supporting_sources', [])) or 'none'}"
                            )
            else:
                st.info("No answer was generated.")

        # ── Expandable: retrieved documents with flag checkboxes ──
        with st.expander("📄 Retrieved Documents", expanded=False):
            import html as _html
            import re as _re
            from collections import OrderedDict

            docs = final_state.get("retrieved_docs", [])

            # ── Group email chunks by message_id (or subject+from fallback) ──
            # Non-email docs get their own group (keyed by index)
            email_groups: OrderedDict[str, list] = OrderedDict()
            standalone: list[tuple[int, object]] = []  # (original_index, doc)

            for idx, doc in enumerate(docs):
                email_from = doc.metadata.get("email_from", "")
                email_subject = doc.metadata.get("email_subject", "")
                message_id = doc.metadata.get("message_id", "")
                is_email = bool(email_from or email_subject)

                if is_email:
                    # Group key: prefer message_id, fall back to subject+from
                    group_key = message_id or f"{email_subject}||{email_from}"
                    if group_key not in email_groups:
                        email_groups[group_key] = []
                    email_groups[group_key].append((idx, doc))
                else:
                    standalone.append((idx, doc))

            # ── Render grouped emails ──
            card_num = 0
            for group_key, chunk_list in email_groups.items():
                card_num += 1
                # Sort chunks by chunk_index so they assemble in order
                chunk_list.sort(key=lambda t: t[1].metadata.get("chunk_index") or 0)

                # Use metadata from first chunk for the email header
                first_doc = chunk_list[0][1]
                src = first_doc.metadata.get("source", "unknown")
                page = first_doc.metadata.get("page", "?")
                email_from = first_doc.metadata.get("email_from", "")
                email_to = first_doc.metadata.get("email_to", "")
                email_subject = first_doc.metadata.get("email_subject", "")
                email_date = first_doc.metadata.get("email_date", "")

                # Collect badges and flags across all chunks in this group
                has_stitched = any(d.metadata.get("is_stitched") for _, d in chunk_list)
                has_att = any(d.metadata.get("is_attachment_context") for _, d in chunk_list)
                max_boost = max((d.metadata.get("metadata_boost", 0) for _, d in chunk_list), default=0)
                chunk_types = list(OrderedDict.fromkeys(
                    d.metadata.get("chunk_type", "") for _, d in chunk_list if d.metadata.get("chunk_type")
                ))
                num_chunks = len(chunk_list)

                badges = f'<span style="background:#e8f4fd;border:1px solid #b8ddf0;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">📄 {_html.escape(str(src))}</span>'
                badges += f'<span style="background:#f0faf3;border:1px solid #a8ddb8;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">p.{page}</span>'
                if num_chunks > 1:
                    badges += f'<span style="background:#e3f2fd;border:1px solid #90caf9;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">📧 {num_chunks} chunks assembled</span>'
                for ct in chunk_types:
                    chunk_color = {"email_header": "#fff3cd", "email_body": "#f0f7ff", "list": "#f5f0ff"}.get(ct, "#f8f8f8")
                    chunk_border = {"email_header": "#ffc107", "email_body": "#4a9eff", "list": "#9b59b6"}.get(ct, "#ccc")
                    badges += f'<span style="background:{chunk_color};border:1px solid {chunk_border};border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">{ct}</span>'
                if has_stitched:
                    badges += '<span style="background:#fff0f5;border:1px solid #ffb3cc;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">🔗 stitched</span>'
                if has_att:
                    badges += '<span style="background:#fff8e1;border:1px solid #ffcc02;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">📎 attachment</span>'
                if max_boost > 0:
                    badges += f'<span style="background:#e8f5e9;border:1px solid #81c784;border-radius:5px;padding:2px 8px;font-size:0.78rem">⬆ boost={max_boost:.2f}</span>'

                # ── Flag checkbox for this email group ──
                st.checkbox(
                    f"🚩 Flag Document {card_num}",
                    key=f"flag_doc_{card_num}",
                    value=st.session_state.get(f"flag_doc_{card_num}", False),
                )

                # ── Assemble full email body from all chunks ──
                body_parts: list[str] = []
                for _, chunk_doc in chunk_list:
                    _raw = chunk_doc.page_content
                    _lines = _raw.splitlines()
                    _clean_lines: list[str] = []
                    for _ln in _lines:
                        _s = _ln.strip().rstrip('\r')
                        if _re.match(
                            r'^(From|To|Cc|CC|Bcc|BCC|Date|Subject|Sent|Importance|Attachments)\s*:',
                            _s,
                        ):
                            continue
                        _clean_lines.append(_ln)
                    cleaned = "\n".join(_clean_lines).strip()
                    if cleaned:
                        body_parts.append(cleaned)

                display_body = "\n\n".join(body_parts)

                header_rows = ""
                if email_from:
                    header_rows += f'<tr><td style="color:#888;font-size:0.82rem;padding:2px 8px 2px 0;white-space:nowrap;font-weight:600">From</td><td style="font-size:0.85rem;padding:2px 0">{_html.escape(email_from)}</td></tr>'
                if email_to:
                    header_rows += f'<tr><td style="color:#888;font-size:0.82rem;padding:2px 8px 2px 0;white-space:nowrap;font-weight:600">To</td><td style="font-size:0.85rem;padding:2px 0">{_html.escape(str(email_to)[:120])}</td></tr>'
                if email_date:
                    header_rows += f'<tr><td style="color:#888;font-size:0.82rem;padding:2px 8px 2px 0;white-space:nowrap;font-weight:600">Date</td><td style="font-size:0.85rem;padding:2px 0">{_html.escape(str(email_date)[:40])}</td></tr>'
                if email_subject:
                    header_rows += f'<tr><td style="color:#888;font-size:0.82rem;padding:2px 8px 2px 0;white-space:nowrap;font-weight:600">Subject</td><td style="font-size:0.85rem;padding:2px 0;font-weight:600">{_html.escape(email_subject)}</td></tr>'

                body_html = _html.escape(display_body).replace("\n", "<br>")

                st.markdown(f"""
<div style="margin-bottom:1rem">
  <div style="margin-bottom:6px"><strong>Document {card_num}</strong> &nbsp; {badges}</div>
  <div style="border:1px solid #d0d7de;border-radius:10px;overflow:hidden;font-family:sans-serif">
    <div style="background:#f6f8fa;padding:10px 14px;border-bottom:1px solid #d0d7de">
      <table style="border-collapse:collapse;width:100%">{header_rows}</table>
    </div>
    <div style="padding:12px 14px;background:#ffffff;font-size:0.88rem;line-height:1.6;color:#24292f;white-space:pre-wrap">{body_html}</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Render standalone (non-email) documents ──
            for _, doc in standalone:
                card_num += 1
                src = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                badges = f'<span style="background:#e8f4fd;border:1px solid #b8ddf0;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">📄 {_html.escape(str(src))}</span>'
                badges += f'<span style="background:#f0faf3;border:1px solid #a8ddb8;border-radius:5px;padding:2px 8px;font-size:0.78rem;margin-right:4px">p.{page}</span>'
                body_html = _html.escape(doc.page_content[:500]).replace("\n", "<br>")

                # ── Flag checkbox for this standalone doc ──
                st.checkbox(
                    f"🚩 Flag Document {card_num}",
                    key=f"flag_doc_{card_num}",
                    value=st.session_state.get(f"flag_doc_{card_num}", False),
                )

                st.markdown(f"""
<div style="margin-bottom:1rem">
  <div style="margin-bottom:6px"><strong>Document {card_num}</strong> &nbsp; {badges}</div>
  <div style="border:1px solid #d0d7de;border-radius:10px;padding:12px 14px;background:#fafafa;font-size:0.88rem;line-height:1.6;color:#24292f">{body_html}</div>
</div>
""", unsafe_allow_html=True)

            # Stash total card count for the save section below
            st.session_state["_flag_card_count"] = card_num

            # ── Quick Save Command Bar ──────────────────────────────────
            st.markdown("---")
            st.markdown("**Quick Save Command**")
            _cmd_text = st.text_input(
                "Type a save command",
                key="save_command_bar",
                placeholder='e.g. "Save doc 1 and 3 as suspicious_batch"',
                label_visibility="collapsed",
            )
            if st.button("⚡ Run Command", key="run_save_cmd"):
                if not _cmd_text or not _cmd_text.strip():
                    st.warning("Please type a command first.")
                else:
                    _max_card = card_num
                    _parsed = _parse_save_command(_cmd_text.strip(), _max_card)

                    if _parsed is None:
                        with st.spinner("Parsing command with LLM..."):
                            _parsed = _llm_parse_save_command(_cmd_text.strip(), _max_card)

                    if _parsed is None:
                        st.error(
                            'Could not parse your command. Try: **save doc 1, 2 as filename**'
                        )
                    else:
                        _cmd_nums, _cmd_fname = _parsed

                        # Validate doc numbers
                        _bad = [n for n in _cmd_nums if n < 1 or n > _max_card]
                        if _bad:
                            st.error(
                                f"Document(s) {_bad} out of range. There are {_max_card} documents."
                            )
                        elif not _re_cmd.match(r'^[\w\-. ]+$', _cmd_fname):
                            st.error(
                                "Filename may only contain letters, digits, hyphens, underscores, dots, and spaces."
                            )
                        else:
                            _all_docs = final_state.get("retrieved_docs", [])
                            _cmd_card_map = _build_card_docs_map(_all_docs)
                            _cmd_payloads = [_cmd_card_map[n] for n in _cmd_nums if n in _cmd_card_map]

                            if not _cmd_payloads:
                                st.error("No documents could be collected for those numbers.")
                            else:
                                try:
                                    from app.db import save_flagged_doc_set
                                    _cmd_engine = init_db(selected_case.case_id)
                                    _cmd_query = st.session_state.get("query_text", "")
                                    _cmd_id = save_flagged_doc_set(
                                        _cmd_engine,
                                        case_id=selected_case.case_id,
                                        filename=_cmd_fname,
                                        query_text=_cmd_query,
                                        docs_json=_cmd_payloads,
                                    )
                                    try:
                                        _cmd_pdf = _export_flagged_pdf(
                                            case_id=selected_case.case_id,
                                            filename=_cmd_fname,
                                            query_text=_cmd_query,
                                            docs=_cmd_payloads,
                                        )
                                        st.success(
                                            f"✅ Saved doc(s) {_cmd_nums} as **{_cmd_fname}** (id={_cmd_id})\n\n"
                                            f"📄 PDF exported to `{_cmd_pdf}`"
                                        )
                                    except Exception as _pdf_e:
                                        st.success(
                                            f"✅ Saved doc(s) {_cmd_nums} as **{_cmd_fname}** (id={_cmd_id})"
                                        )
                                        st.warning(f"⚠️ PDF export failed: {_pdf_e}")
                                except Exception as _cmd_exc:
                                    if "uq_saved_flagged_docs_case_filename" in str(_cmd_exc):
                                        st.error(f"A file named **{_cmd_fname}** already exists for this case.")
                                    else:
                                        st.error(f"Save failed: {_cmd_exc}")

        # ── Save Flagged Documents section ──────────────────────────────
        _total_cards = st.session_state.get("_flag_card_count", 0)
        _flagged_indices = [
            i for i in range(1, _total_cards + 1)
            if st.session_state.get(f"flag_doc_{i}", False)
        ]

        if _flagged_indices:
            st.markdown(f"**🚩 {len(_flagged_indices)} document(s) flagged**")

            _flag_filename = st.text_input(
                "Filename for saved set",
                key="flag_save_filename",
                placeholder="e.g. suspicious_emails_batch1",
            )

            if st.button("💾 Save Flagged Documents", type="primary"):
                if not _flag_filename or not _flag_filename.strip():
                    st.error("Please enter a filename.")
                else:
                    import re as _re_val
                    _fname = _flag_filename.strip()
                    if not _re_val.match(r'^[\w\-. ]+$', _fname):
                        st.error("Filename may only contain letters, digits, hyphens, underscores, dots, and spaces.")
                    else:
                        # Build flagged doc payloads using shared helper
                        _all_docs = final_state.get("retrieved_docs", [])
                        _card_docs = _build_card_docs_map(_all_docs)
                        _flagged_payloads = [_card_docs[i] for i in _flagged_indices if i in _card_docs]

                        if not _flagged_payloads:
                            st.error("No flagged documents could be collected.")
                        else:
                            try:
                                from app.db import save_flagged_doc_set
                                _save_engine = init_db(selected_case.case_id)
                                _query_text = st.session_state.get("query_text", "")
                                _new_id = save_flagged_doc_set(
                                    _save_engine,
                                    case_id=selected_case.case_id,
                                    filename=_fname,
                                    query_text=_query_text,
                                    docs_json=_flagged_payloads,
                                )
                                # Also export as local PDF
                                try:
                                    _pdf_path = _export_flagged_pdf(
                                        case_id=selected_case.case_id,
                                        filename=_fname,
                                        query_text=_query_text,
                                        docs=_flagged_payloads,
                                    )
                                    st.success(
                                        f"✅ Saved {len(_flagged_payloads)} flagged doc(s) as **{_fname}** (id={_new_id})\n\n"
                                        f"📄 PDF exported to `{_pdf_path}`"
                                    )
                                except Exception as _pdf_exc:
                                    st.success(
                                        f"✅ Saved {len(_flagged_payloads)} flagged doc(s) as **{_fname}** (id={_new_id})"
                                    )
                                    st.warning(f"⚠️ PDF export failed: {_pdf_exc}")
                            except Exception as _save_exc:
                                if "uq_saved_flagged_docs_case_filename" in str(_save_exc):
                                    st.error(f"A file named **{_fname}** already exists for this case. Choose a different name.")
                                else:
                                    st.error(f"Save failed: {_save_exc}")

        # ── View saved flagged doc sets ──────────────────────────────
        with st.expander("📂 Saved Flagged Documents", expanded=False):
            try:
                from app.db import list_flagged_doc_sets, get_flagged_doc_set
                _view_engine = init_db(selected_case.case_id)
                _saved_sets = list_flagged_doc_sets(_view_engine, selected_case.case_id)
                if _saved_sets:
                    for _ss in _saved_sets:
                        _scol1, _scol2, _scol3 = st.columns([3, 1, 1])
                        with _scol1:
                            st.markdown(f"**{_ss['filename']}** — {_ss['doc_count']} docs — {str(_ss['saved_at'])[:19]}")
                        with _scol2:
                            if st.button("View", key=f"view_saved_{_ss['id']}"):
                                _detail = get_flagged_doc_set(_view_engine, _ss["id"])
                                if _detail:
                                    st.session_state[f"_saved_detail_{_ss['id']}"] = _detail
                        with _scol3:
                            st.caption(f"Query: {_ss['query_text'][:60]}...")
                        # Show detail if loaded
                        if f"_saved_detail_{_ss['id']}" in st.session_state:
                            _det = st.session_state[f"_saved_detail_{_ss['id']}"]
                            st.json(_det["docs_json"])
                else:
                    st.caption("No saved flagged document sets for this case.")
            except Exception as _view_exc:
                st.warning(f"Could not load saved sets: {_view_exc}")

        # ── Expandable: validation details ──
        v_results = final_state.get("validation_results", [])
        if v_results:
            with st.expander("🛡️ Validation Details", expanded=False):
                for r in v_results:
                    icon = "✅" if r.get("is_valid") else "❌"
                    st.markdown(f"{icon} **Claim:** {r.get('claim', '?')}")
                    st.markdown(f"   Endpoint: `{r.get('endpoint', '?')}` — Valid: **{r.get('is_valid')}**")
                    st.markdown("---")


# ═════════════════════════════════════════════════════════════════════
# TAB 2: DATA MANAGEMENT
# ═════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.markdown("### 📦 Data Ingestion & Management")
    st.markdown(
        "Manage document ingestion for each case. "
        "Ingestion runs in the background — you can switch to the Query tab while it runs."
    )

    # ── Show case status overview ────────────────────────────────
    st.markdown("#### Case Status")

    all_cases = list_cases()
    case_status_data = []
    for c in all_cases:
        chunks = _get_chunk_count(c.case_id)
        case_status_data.append({
            "Case": c.display_name,
            "Case ID": c.case_id,
            "Doc Type": c.doc_type,
            "Chunks": chunks,
            "Status": "✅ Ingested" if chunks > 0 else "⚠️ Empty",
            "Source": c.blob_prefix or c.data_dir or "—",
        })

    st.dataframe(case_status_data, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Ingest selected case ─────────────────────────────────────
    st.markdown(f"#### Ingest: **{selected_case.display_name}**")

    current_chunks = _get_chunk_count(selected_case.case_id)
    if current_chunks > 0:
        st.success(f"This case already has **{current_chunks}** chunks in the database.")

    # Determine source
    from app.blob_storage import is_blob_mode
    blob_mode = is_blob_mode()

    if blob_mode and selected_case.blob_prefix:
        source_display = f"Azure Blob: `{selected_case.blob_prefix}`"
        source_path = selected_case.blob_prefix
    elif selected_case.data_dir:
        source_display = f"Local: `{selected_case.data_dir}`"
        source_path = selected_case.data_dir
    else:
        source_display = "No data source configured"
        source_path = ""

    st.info(f"**Source:** {source_display}", icon="☁️" if blob_mode else "📁")

    # Custom source override
    with st.expander("Custom source (advanced)", expanded=False):
        custom_source = st.text_input(
            "Override source path (blob prefix or local directory)",
            value="",
            key="custom_source_path",
            placeholder="e.g. demo-pst or data/sample_corpus",
        )
        if custom_source.strip():
            source_path = custom_source.strip()

    # Ingestion state key
    ingest_key = f"ingest_result_{selected_case.case_id}"

    # ── Start / Status buttons ───────────────────────────────────
    col_ingest, col_status = st.columns([1, 2])

    with col_ingest:
        ingest_disabled = not source_path
        if st.button(
            "▶ Start Ingestion",
            type="primary",
            use_container_width=True,
            disabled=ingest_disabled,
            key="btn_start_ingest",
        ):
            if source_path:
                st.session_state[ingest_key] = {"status": "running", "message": "Starting ingestion..."}
                t = threading.Thread(
                    target=_run_ingest_thread,
                    args=(selected_case.case_id, source_path, ingest_key),
                    daemon=True,
                )
                t.start()
                st.rerun()

    with col_status:
        if st.button("🔄 Refresh Status", use_container_width=True, key="btn_refresh_ingest"):
            st.rerun()

    # ── Show ingestion result / progress ─────────────────────────
    # Transfer any background-thread result into session_state
    _get_ingest_result(ingest_key)
    ingest_state = st.session_state.get(ingest_key, None)

    if ingest_state:
        status = ingest_state.get("status", "unknown")

        if status == "running":
            st.warning("⏳ **Ingestion in progress…** Click **Refresh Status** to check for completion.", icon="⏳")

        elif status == "completed":
            result = ingest_state.get("result", {})
            st.success(
                f"✅ **Ingestion complete!**  \n"
                f"- **Chunks inserted:** {result.get('chunks_inserted', '?')}  \n"
                f"- **Summary:** {result.get('summary', 'N/A')}",
                icon="✅",
            )
            if result.get("files_failed", 0) > 0:
                st.warning(
                    f"⚠️ {result['files_failed']} file(s) failed to load:  \n"
                    + "\n".join(f"- {f['file']}: {f['error']}" for f in result.get("failed_files", []))
                )

        elif status == "failed":
            st.error(f"❌ **Ingestion failed:** {ingest_state.get('error', 'Unknown error')}", icon="❌")

    # ── CLI reference ────────────────────────────────────────────
    st.markdown("---")
    with st.expander("CLI Reference", expanded=False):
        st.markdown(
            f"""
You can also run ingestion from the command line:

```powershell
# Ingest from blob storage
python -m app.ingest --case {selected_case.case_id} --blob-prefix {selected_case.blob_prefix or 'YOUR-PREFIX'}

# Ingest from local directory
python -m app.ingest --case {selected_case.case_id} --dir {selected_case.data_dir or 'data/YOUR-FOLDER'}

# Specify doc type explicitly
python -m app.ingest --case {selected_case.case_id} --blob-prefix {selected_case.blob_prefix or 'PREFIX'} --doc-type {selected_case.doc_type}
```

**REST API** (requires API key):

```bash
# Submit async ingestion job
curl -X POST http://localhost:8080/api/v1/jobs/submit \\
  -H "X-API-Key: YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"job_type": "ingest", "case_id": "{selected_case.case_id}", "source_path": "{source_path}"}}'

# Check job status
curl http://localhost:8080/api/v1/jobs/JOB_ID -H "X-API-Key: YOUR_KEY"
```
"""
        )
