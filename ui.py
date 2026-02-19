"""Streamlit Demo UI for the Self-Correction Audit Agent.

Launch:
    streamlit run ui.py
"""

from __future__ import annotations

import time
import logging
from typing import Any

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="Agentic RAG - Audit Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports from the app package ────────────────────────────────────
from app.graph import compile_graph
from app.state import AgentState
from app.cases import CASES, list_cases, get_case, metadata_filter_for_case


# ── Logging (suppress noisy libs in the UI) ─────────────────────────
logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


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
    meta = NODE_META.get(node, {"icon": "🔹", "label": node.title(), "desc": ""})
    detail_html = ""

    if node == "retriever":
        docs = data.get("retrieved_docs", [])
        n = len(docs)
        sources = sorted({d.metadata.get("source", "?") for d in docs})
        chips = "".join(f'<span class="source-chip">{s}</span>' for s in sources[:8])
        detail_html = f"Retrieved <strong>{n}</strong> document chunks<br/>{chips}"

    elif node == "grader":
        score = data.get("grader_score", "?")
        reasoning = data.get("grader_reasoning", "")
        icon = "✅" if score == "yes" else "❌"
        detail_html = (
            f"Relevance: {icon} <strong>{score.upper()}</strong><br/>"
            f"<em>{reasoning[:200]}</em>"
        )
        if score == "no":
            css_class = "fail"

    elif node == "rewriter":
        rq = data.get("rewritten_query", "")
        loop = data.get("loop_count", 0)
        detail_html = f"Loop <strong>{loop}</strong> &mdash; rewritten to:<br/><em>{rq[:200]}</em>"

    elif node == "validator":
        status = data.get("validation_status", "pass")
        results = data.get("validation_results", [])
        n_claims = len(results)
        n_valid = sum(1 for r in results if r.get("is_valid"))
        badge = {"pass": "✅ PASS", "partial": "⚠️ PARTIAL", "fail": "❌ FAIL"}.get(status, status)
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
