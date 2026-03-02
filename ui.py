"""Streamlit Demo UI for the Self-Correction Audit Agent.

Launch:
    streamlit run ui.py
"""

from __future__ import annotations

import time
import logging
import shlex
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
# Data is already ingested – use CLI for manual ingestion if needed:
#   python -m app.ingest --case <case_id>
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
# Helpers
# ─────────────────────────────────────────────────────────────────────


NODE_META: dict[str, dict[str, str]] = {
    "retriever": {"icon": "📥", "label": "Retriever",  "desc": "Hybrid semantic + keyword search via Cohere & pgvector"},
    "grader":    {"icon": "⚖️",  "label": "Grader",     "desc": "LLM-based binary relevance check (GPT-4o)"},
    "rewriter":  {"icon": "✏️",  "label": "Rewriter",   "desc": "Query reformulation for better recall"},
    "validator": {"icon": "🛡️", "label": "Validator",  "desc": "Cross-verify extracted claims against external APIs"},
    "generator": {"icon": "📝", "label": "Generator",  "desc": "Audit-ready answer synthesis with citations"},
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

    st.markdown(f"""
    <div class="step-card {css_class}">
        <div class="step-header">{meta['icon']} {meta['label']}</div>
        <div class="step-detail">{meta['desc']}<br/>{detail_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Main query area
# ─────────────────────────────────────────────────────────────────────
# Handle prefill from sample-question buttons (must happen BEFORE the widget)
if "prefill_query" in st.session_state:
    st.session_state["query_text"] = st.session_state.pop("prefill_query")

query = st.text_area(
    "🔎 Enter your compliance / audit question",
    key="query_text",
    height=90,
    placeholder="e.g. What were the issues with the Indian workers at Big Thorium?",
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_clicked = st.button("▶  Run Query", type="primary", use_container_width=True)
with col_clear:
    if st.button("🗑  Clear history"):
        st.session_state.pop("steps", None)
        st.session_state.pop("final_answer", None)
        st.rerun()

# ─────────────────────────────────────────────────────────────────────
# Execute pipeline
# ─────────────────────────────────────────────────────────────────────
if run_clicked and query.strip():
    # ── Input validation ───────────────────────────────────────
    if len(query.strip()) > settings.ui_max_query_length:
        st.error(f"Query too long ({len(query.strip())} chars). Max is {settings.ui_max_query_length}.")
        st.stop()

    # Ensure the per-case database is ready
    init_db(selected_case.case_id)

    # Build initial state with case-level isolation
    initial_state: AgentState = {
        "query": query.strip(),
        "loop_count": 0,
        "case_id": selected_case.case_id,
    }

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

    st.markdown(f"""
    <div class="metric-row">
        <span class="metric-pill">📥 {n_docs} docs retrieved</span>
        <span class="metric-pill">⚖️ Grader: {score.upper()}</span>
        <span class="metric-pill">🛡️ Validation: {v_status.upper()}</span>
        <span class="metric-pill">🔄 Rewrites: {loops}</span>
        <span class="metric-pill">⏱️ {elapsed:.1f}s</span>
        <span class="metric-pill">📏 {answer_len:,} chars</span>
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
        if answer:
            st.markdown(f'<div class="answer-box">', unsafe_allow_html=True)
            st.markdown(answer)
            st.markdown('</div>', unsafe_allow_html=True)

            # Validation warning
            if final_state.get("validation_status") in ("partial", "fail"):
                st.warning(
                    f"Validation status: **{final_state['validation_status'].upper()}** "
                    "— some claims could not be verified. See discrepancy notes above."
                )
        else:
            st.info("No answer was generated.")

    # ── Expandable: retrieved documents ──
    with st.expander("📄 Retrieved Documents", expanded=False):
        docs = final_state.get("retrieved_docs", [])
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**Document {i}** — `{src}` (page {page})")
            st.text(doc.page_content[:500])
            st.markdown("---")

    # ── Expandable: validation details ──
    v_results = final_state.get("validation_results", [])
    if v_results:
        with st.expander("🛡️ Validation Details", expanded=False):
            for r in v_results:
                icon = "✅" if r.get("is_valid") else "❌"
                st.markdown(f"{icon} **Claim:** {r.get('claim', '?')}")
                st.markdown(f"   Endpoint: `{r.get('endpoint', '?')}` — Valid: **{r.get('is_valid')}**")
                st.markdown("---")
