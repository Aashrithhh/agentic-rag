"""RAG Evaluation Framework — retrieval quality, answer quality, pipeline metrics.

Three measurement layers:
  1. Retrieval: precision@k, recall@k, MRR (requires ground-truth test set)
  2. Answer quality: faithfulness & relevance via LLM-as-judge (always available)
  3. Pipeline: per-node latency, grader pass rate, rewrite rate

Usage:
    python -m app.evaluation --case big-thorium
    python -m app.evaluation --case big-thorium --test-set eval/big_thorium_test.jsonl
    python -m app.evaluation --case big-thorium --output results.json --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import settings
from app.db import init_db
from app.graph import compile_graph
from app.state import AgentState

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────────────────


@dataclass
class TestQuestion:
    """A single evaluation question, optionally with ground truth."""

    query: str
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_keywords: list[str] = field(default_factory=list)
    expected_answer: str = ""


@dataclass
class NodeTiming:
    """Timing for a single pipeline node execution."""

    node: str
    duration_seconds: float
    output_keys: list[str] = field(default_factory=list)


@dataclass
class QuestionResult:
    """Full evaluation result for a single question."""

    query: str
    case_id: str
    # Retrieval metrics
    retrieved_sources: list[str] = field(default_factory=list)
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    reciprocal_rank: float = 0.0
    has_ground_truth: bool = False
    # Answer quality
    faithfulness_score: float = 0.0
    faithfulness_reasoning: str = ""
    relevance_score: float = 0.0
    relevance_reasoning: str = ""
    # Pipeline metrics
    total_latency_seconds: float = 0.0
    node_timings: list[NodeTiming] = field(default_factory=list)
    loop_count: int = 0
    grader_score: str = ""
    validation_status: str = ""
    answer_length: int = 0
    answer: str = ""


# ── LLM-as-judge models ─────────────────────────────────────────────


class FaithfulnessGrade(BaseModel):
    """LLM-as-judge score for answer faithfulness."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = completely unfaithful, 1.0 = fully grounded in documents",
    )
    reasoning: str


class RelevanceGrade(BaseModel):
    """LLM-as-judge score for answer relevance."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="0.0 = completely irrelevant, 1.0 = directly answers the question",
    )
    reasoning: str


# ── Retrieval metrics ────────────────────────────────────────────────


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are relevant."""
    if not relevant or k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant docs found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc in relevant if doc in top_k)
    return hits / len(relevant)


def mean_reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


# ── Answer quality (LLM-as-judge) ───────────────────────────────────


_FAITHFULNESS_SYSTEM = """\
You are an evaluation judge for a compliance document retrieval system.
Score how faithfully the ANSWER is grounded in the provided DOCUMENTS.
A faithful answer only makes claims that are directly supported by the documents.
Unsupported claims, hallucinations, or extrapolations lower the score.
Score on a scale of 0.0 to 1.0."""

_FAITHFULNESS_HUMAN = """\
<query>{query}</query>

<documents>
{documents}
</documents>

<answer>
{answer}
</answer>

Score the faithfulness of this answer to the provided documents."""

_RELEVANCE_SYSTEM = """\
You are an evaluation judge for a compliance document retrieval system.
Score how relevant the ANSWER is to the QUERY.
A relevant answer directly addresses what was asked, is complete, and stays on topic.
Off-topic content, missing key information, or vague responses lower the score.
Score on a scale of 0.0 to 1.0."""

_RELEVANCE_HUMAN = """\
<query>{query}</query>

<answer>
{answer}
</answer>

Score the relevance of this answer to the query."""


def _get_judge_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        max_tokens=512,
    )


def judge_faithfulness(
    query: str, documents: str, answer: str
) -> FaithfulnessGrade:
    """Use LLM-as-judge to score answer faithfulness."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _FAITHFULNESS_SYSTEM),
        ("human", _FAITHFULNESS_HUMAN),
    ])
    chain = prompt | _get_judge_llm().with_structured_output(FaithfulnessGrade)
    return chain.invoke({"query": query, "documents": documents, "answer": answer})


def judge_relevance(query: str, answer: str) -> RelevanceGrade:
    """Use LLM-as-judge to score answer relevance."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", _RELEVANCE_SYSTEM),
        ("human", _RELEVANCE_HUMAN),
    ])
    chain = prompt | _get_judge_llm().with_structured_output(RelevanceGrade)
    return chain.invoke({"query": query, "answer": answer})


# ── Pipeline execution with timing ──────────────────────────────────


def run_with_timing(
    query: str, case_id: str
) -> tuple[dict[str, Any], list[NodeTiming], float]:
    """Run the full pipeline and capture per-node wall-clock timing.

    Returns (final_state, node_timings, total_seconds).
    """
    init_db(case_id)
    agent = compile_graph()
    initial_state: AgentState = {
        "query": query,
        "loop_count": 0,
        "case_id": case_id,
    }

    timings: list[NodeTiming] = []
    final_state: dict[str, Any] = {}
    t_start = time.perf_counter()
    t_prev = t_start

    for step in agent.stream(initial_state):
        t_now = time.perf_counter()
        node_name = list(step.keys())[0]
        node_output = step[node_name]

        timings.append(NodeTiming(
            node=node_name,
            duration_seconds=round(t_now - t_prev, 4),
            output_keys=list(node_output.keys()),
        ))
        final_state.update(node_output)
        t_prev = t_now

    total = time.perf_counter() - t_start
    return final_state, timings, round(total, 4)


# ── Full evaluation ─────────────────────────────────────────────────


def evaluate_question(question: TestQuestion, case_id: str) -> QuestionResult:
    """Run full evaluation for a single question."""
    result = QuestionResult(query=question.query, case_id=case_id)

    # Run pipeline with timing
    try:
        state, timings, total_time = run_with_timing(question.query, case_id)
    except Exception as exc:
        logger.error("Pipeline failed for query '%s': %s", question.query[:60], exc)
        result.answer = f"[PIPELINE ERROR: {exc}]"
        return result

    result.node_timings = timings
    result.total_latency_seconds = total_time
    result.loop_count = state.get("loop_count", 0)
    result.grader_score = state.get("grader_score", "")
    result.validation_status = state.get("validation_status", "")
    result.answer = state.get("answer", "")
    result.answer_length = len(result.answer)

    # Extract retrieved sources
    docs: list[Document] = state.get("retrieved_docs", [])
    result.retrieved_sources = [
        d.metadata.get("source", "unknown") for d in docs
    ]

    # Retrieval metrics (only if ground truth provided)
    if question.expected_sources:
        result.has_ground_truth = True
        k = len(result.retrieved_sources)
        result.precision_at_k = precision_at_k(
            result.retrieved_sources, question.expected_sources, k
        )
        result.recall_at_k = recall_at_k(
            result.retrieved_sources, question.expected_sources, k
        )
        result.reciprocal_rank = mean_reciprocal_rank(
            result.retrieved_sources, question.expected_sources
        )

    # Answer quality (LLM-as-judge)
    if result.answer and not result.answer.startswith("[PIPELINE ERROR"):
        doc_text = "\n---\n".join(d.page_content[:500] for d in docs[:6])

        try:
            faith = judge_faithfulness(question.query, doc_text, result.answer)
            result.faithfulness_score = faith.score
            result.faithfulness_reasoning = faith.reasoning
        except Exception as exc:
            logger.error("Faithfulness judge failed: %s", exc)

        try:
            rel = judge_relevance(question.query, result.answer)
            result.relevance_score = rel.score
            result.relevance_reasoning = rel.reasoning
        except Exception as exc:
            logger.error("Relevance judge failed: %s", exc)

    return result


def evaluate_case(
    case_id: str,
    test_questions: list[TestQuestion] | None = None,
) -> list[QuestionResult]:
    """Evaluate all questions for a case.

    If no test_questions provided, uses sample_questions from case config.
    """
    # Clear caches for clean measurements
    from app.cache import cache
    removed = cache.invalidate_all()
    if removed:
        logger.info("[eval] Cleared %d cache entries for clean measurement", removed)

    if test_questions is None:
        from app.cases import get_case
        cfg = get_case(case_id)
        test_questions = [TestQuestion(query=q) for q in cfg.sample_questions]

    if not test_questions:
        logger.warning("No test questions found for case '%s'", case_id)
        return []

    results: list[QuestionResult] = []
    for i, q in enumerate(test_questions, 1):
        logger.info(
            "Evaluating question %d/%d: %s",
            i, len(test_questions), q.query[:80],
        )
        result = evaluate_question(q, case_id)
        results.append(result)
        logger.info(
            "  -> faithfulness=%.2f  relevance=%.2f  latency=%.1fs  grader=%s",
            result.faithfulness_score,
            result.relevance_score,
            result.total_latency_seconds,
            result.grader_score,
        )

    return results


# ── Report generation ────────────────────────────────────────────────


def generate_report(results: list[QuestionResult]) -> dict[str, Any]:
    """Generate an aggregate evaluation report."""
    n = len(results)
    if n == 0:
        return {"error": "No results to report"}

    avg_faithfulness = sum(r.faithfulness_score for r in results) / n
    avg_relevance = sum(r.relevance_score for r in results) / n
    avg_latency = sum(r.total_latency_seconds for r in results) / n
    avg_answer_len = sum(r.answer_length for r in results) / n

    # Retrieval metrics (only for questions with ground truth)
    gt_results = [r for r in results if r.has_ground_truth]
    if gt_results:
        avg_precision = round(sum(r.precision_at_k for r in gt_results) / len(gt_results), 3)
        avg_recall = round(sum(r.recall_at_k for r in gt_results) / len(gt_results), 3)
        avg_mrr = round(sum(r.reciprocal_rank for r in gt_results) / len(gt_results), 3)
    else:
        avg_precision = None
        avg_recall = None
        avg_mrr = None

    # Node-level latency breakdown
    node_latencies: dict[str, list[float]] = {}
    for r in results:
        for t in r.node_timings:
            node_latencies.setdefault(t.node, []).append(t.duration_seconds)

    node_avg_latency = {
        node: round(sum(times) / len(times), 4)
        for node, times in node_latencies.items()
    }

    grader_pass_rate = sum(1 for r in results if r.grader_score == "yes") / n
    rewrite_rate = sum(1 for r in results if r.loop_count > 0) / n

    return {
        "num_questions": n,
        "answer_quality": {
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_relevance": round(avg_relevance, 3),
        },
        "retrieval_metrics": {
            "avg_precision_at_k": avg_precision if avg_precision is not None else "N/A (no ground truth)",
            "avg_recall_at_k": avg_recall if avg_recall is not None else "N/A (no ground truth)",
            "avg_mrr": avg_mrr if avg_mrr is not None else "N/A (no ground truth)",
            "questions_with_ground_truth": len(gt_results),
        },
        "pipeline_metrics": {
            "avg_total_latency_seconds": round(avg_latency, 2),
            "avg_answer_length_chars": round(avg_answer_len),
            "grader_pass_rate": round(grader_pass_rate, 3),
            "rewrite_rate": round(rewrite_rate, 3),
            "node_avg_latency_seconds": node_avg_latency,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("  RAG EVALUATION REPORT")
    print("=" * 60)

    print(f"\n  Questions evaluated: {report['num_questions']}")

    aq = report["answer_quality"]
    print("\n  --- Answer Quality ---")
    print(f"  Avg faithfulness:  {aq['avg_faithfulness']}")
    print(f"  Avg relevance:     {aq['avg_relevance']}")

    rm = report["retrieval_metrics"]
    print("\n  --- Retrieval Metrics ---")
    print(f"  Avg precision@k:   {rm['avg_precision_at_k']}")
    print(f"  Avg recall@k:      {rm['avg_recall_at_k']}")
    print(f"  Avg MRR:           {rm['avg_mrr']}")
    if isinstance(rm.get("questions_with_ground_truth"), int):
        print(f"  Ground truth Qs:   {rm['questions_with_ground_truth']}")

    pm = report["pipeline_metrics"]
    print("\n  --- Pipeline Metrics ---")
    print(f"  Avg latency:       {pm['avg_total_latency_seconds']}s")
    print(f"  Avg answer length: {pm['avg_answer_length_chars']} chars")
    print(f"  Grader pass rate:  {pm['grader_pass_rate']}")
    print(f"  Rewrite rate:      {pm['rewrite_rate']}")
    print("  Node latencies:")
    for node, lat in pm["node_avg_latency_seconds"].items():
        print(f"    {node:15s}  {lat}s")

    print("\n" + "=" * 60)


# ── Test set loading ─────────────────────────────────────────────────


def load_test_set(path: str | Path) -> list[TestQuestion]:
    """Load a JSONL test set file.

    Each line is a JSON object with:
      - query (required)
      - expected_sources (optional list of filenames)
      - expected_answer_keywords (optional list of strings)
      - expected_answer (optional string)
    """
    questions: list[TestQuestion] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line %d: %s", line_num, exc)
                continue
            questions.append(TestQuestion(
                query=data["query"],
                expected_sources=data.get("expected_sources", []),
                expected_answer_keywords=data.get("expected_answer_keywords", []),
                expected_answer=data.get("expected_answer", ""),
            ))
    return questions


# ── CLI entry point ──────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation Framework")
    parser.add_argument("--case", required=True, help="Case ID to evaluate")
    parser.add_argument(
        "--test-set",
        default=None,
        help="Path to JSONL test set (uses case sample_questions if omitted)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write JSON results",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s | %(name)-30s | %(message)s",
    )

    # Load test questions
    if args.test_set:
        questions = load_test_set(args.test_set)
        logger.info("Loaded %d questions from %s", len(questions), args.test_set)
    else:
        questions = None
        logger.info("Using sample questions from case config")

    # Run evaluation
    results = evaluate_case(args.case, questions)
    report = generate_report(results)

    # Output
    print_report(report)

    if args.output:
        output_data = {
            "report": report,
            "questions": [
                {
                    "query": r.query,
                    "faithfulness": r.faithfulness_score,
                    "faithfulness_reasoning": r.faithfulness_reasoning,
                    "relevance": r.relevance_score,
                    "relevance_reasoning": r.relevance_reasoning,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k": r.recall_at_k,
                    "mrr": r.reciprocal_rank,
                    "latency": r.total_latency_seconds,
                    "loop_count": r.loop_count,
                    "grader_score": r.grader_score,
                    "validation_status": r.validation_status,
                    "answer_length": r.answer_length,
                    "retrieved_sources": r.retrieved_sources,
                    "node_timings": [
                        {"node": t.node, "seconds": t.duration_seconds}
                        for t in r.node_timings
                    ],
                }
                for r in results
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results written to {args.output}")


if __name__ == "__main__":
    main()
