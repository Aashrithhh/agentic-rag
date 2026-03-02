#!/usr/bin/env python
"""CI/CD SLO gate — fail the build if load test results breach thresholds.

Usage:
    # After running locust load test that outputs stats JSON:
    python scripts/ci_slo_gate.py --stats locust_stats.json --eval eval_results.json

    # Or check runtime metrics from a live API:
    python scripts/ci_slo_gate.py --api-url http://localhost:8080

Exit codes:
    0  — all SLOs pass
    1  — one or more critical SLO violations
    2  — input/configuration error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_locust_stats(path: str) -> dict:
    """Parse Locust JSON stats into metrics format."""
    with open(path) as f:
        raw = json.load(f)

    counters: dict[str, int] = {}
    histograms: dict[str, dict[str, float]] = {}

    # Locust stats_history or stats array
    stats = raw if isinstance(raw, list) else raw.get("stats", [])

    total_requests = 0
    total_failures = 0

    for entry in stats:
        name = entry.get("name", "unknown")
        method = entry.get("method", "GET")
        key = f"http_request_seconds.{method}.{name}"

        total_requests += entry.get("num_requests", 0)
        total_failures += entry.get("num_failures", 0)

        # Convert ms to seconds
        histograms[key] = {
            "count": entry.get("num_requests", 0),
            "mean": entry.get("avg_response_time", 0) / 1000,
            "p50": entry.get("median_response_time", 0) / 1000,
            "p95": (entry.get("response_times", {}).get("0.95", 0) or
                    entry.get("avg_response_time", 0) * 1.5) / 1000,
            "p99": (entry.get("response_times", {}).get("0.99", 0) or
                    entry.get("avg_response_time", 0) * 2) / 1000,
            "max": entry.get("max_response_time", 0) / 1000,
        }

    counters["node_invocations.retriever"] = total_requests
    counters["llm_call_failure.total"] = total_failures

    return {"counters": counters, "histograms": histograms}


def _load_api_metrics(api_url: str) -> dict:
    """Fetch metrics from running API."""
    import httpx
    resp = httpx.get(f"{api_url}/metrics", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get("pipeline", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="CI/CD SLO Gate")
    parser.add_argument("--stats", help="Path to Locust stats JSON file")
    parser.add_argument("--eval", help="Path to evaluation results JSON file")
    parser.add_argument("--api-url", help="Live API URL to fetch metrics from")
    parser.add_argument("--output", "-o", default="slo_report.json",
                        help="Output path for SLO report")
    parser.add_argument("--cost", type=float, default=0.0,
                        help="Total cost in USD for the test run")
    parser.add_argument("--queries", type=int, default=0,
                        help="Number of queries in the test run")
    args = parser.parse_args()

    # Late import to avoid import errors when running standalone
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.slo import SLO, full_slo_check, save_slo_report

    # Load metrics
    if args.stats:
        metrics_snapshot = _load_locust_stats(args.stats)
    elif args.api_url:
        metrics_snapshot = _load_api_metrics(args.api_url)
    else:
        print("ERROR: Must provide --stats or --api-url", file=sys.stderr)
        sys.exit(2)

    # Load eval results
    eval_report = None
    if args.eval:
        with open(args.eval) as f:
            eval_data = json.load(f)
        eval_report = eval_data.get("report", eval_data)

    # Run SLO checks
    report = full_slo_check(
        metrics_snapshot=metrics_snapshot,
        eval_report=eval_report,
        total_cost_usd=args.cost,
        num_queries=args.queries,
        thresholds=SLO,
    )

    # Output
    print(report.summary())
    save_slo_report(report, args.output)

    if not report.passed:
        print(f"\n❌ CI/CD GATE FAILED — {len(report.violations)} SLO violation(s)")
        sys.exit(1)
    else:
        print(f"\n✅ CI/CD GATE PASSED — all SLOs within thresholds")
        sys.exit(0)


if __name__ == "__main__":
    main()
