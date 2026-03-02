#!/usr/bin/env python3
"""Nightly evaluation runner — benchmark RAG quality against gold queries.

Usage:
    python scripts/nightly_eval.py --case big_thorium --eval-file eval/big_thorium_test.jsonl
    python scripts/nightly_eval.py --all-cases

Outputs a timestamped JSON report to eval/reports/ and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.slo import SLOThresholds, full_slo_check, save_slo_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_eval_files(case_id: str | None = None) -> list[tuple[str, str]]:
    """Return list of (case_id, eval_file_path) pairs."""
    eval_dir = Path("eval")
    pairs: list[tuple[str, str]] = []

    if case_id:
        # Look for <case>.jsonl or <case>_test.jsonl
        candidates = [
            eval_dir / f"{case_id}.jsonl",
            eval_dir / f"{case_id}_test.jsonl",
        ]
        # Also check gold_queries/<case>/*.jsonl
        gold_dir = eval_dir / "gold_queries" / case_id
        if gold_dir.is_dir():
            candidates.extend(gold_dir.glob("*.jsonl"))

        for p in candidates:
            if p.is_file():
                pairs.append((case_id, str(p)))
    else:
        # Discover all eval files
        for f in eval_dir.glob("*_test.jsonl"):
            cid = f.stem.replace("_test", "")
            pairs.append((cid, str(f)))
        gold_root = eval_dir / "gold_queries"
        if gold_root.is_dir():
            for case_dir in gold_root.iterdir():
                if case_dir.is_dir():
                    for f in case_dir.glob("*.jsonl"):
                        pairs.append((case_dir.name, str(f)))

    return pairs


def run_nightly_eval(
    case_id: str,
    eval_file: str,
    *,
    thresholds: SLOThresholds | None = None,
) -> dict:
    """Run a full evaluation cycle for a single case."""
    from app.evaluation import run_batch_evaluation

    thresholds = thresholds or SLOThresholds()
    logger.info("═══ Running nightly eval for case '%s' ═══", case_id)
    logger.info("Eval file: %s", eval_file)

    # 1. Run batch eval
    eval_results = run_batch_evaluation(case_id, eval_file)

    # 2. Check SLOs
    slo_report = full_slo_check(thresholds)

    # 3. Combine into report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "eval_file": eval_file,
        "evaluation": eval_results,
        "slo_report": {
            "pass": slo_report.passed,
            "violations": [
                {"metric": v.metric, "threshold": v.threshold, "actual": v.actual, "severity": v.severity}
                for v in slo_report.violations
            ],
        },
    }

    # 4. Save report
    reports_dir = Path("eval/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"nightly_{case_id}_{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    # 5. Summary
    status = "PASS" if slo_report.passed else "FAIL"
    logger.info(
        "═══ Nightly eval %s for '%s' — %d SLO violations ═══",
        status, case_id, len(slo_report.violations),
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightly RAG quality evaluation")
    parser.add_argument("--case", type=str, help="Case ID to evaluate")
    parser.add_argument("--eval-file", type=str, help="Path to evaluation JSONL file")
    parser.add_argument("--all-cases", action="store_true", help="Evaluate all cases with available eval files")
    parser.add_argument("--fail-on-violation", action="store_true", help="Exit non-zero on SLO violations")
    args = parser.parse_args()

    if args.eval_file and args.case:
        pairs = [(args.case, args.eval_file)]
    elif args.case:
        pairs = _find_eval_files(args.case)
    elif args.all_cases:
        pairs = _find_eval_files()
    else:
        parser.error("Specify --case, --eval-file, or --all-cases")
        return

    if not pairs:
        logger.error("No evaluation files found")
        sys.exit(1)

    all_passed = True
    for case_id, eval_file in pairs:
        try:
            report = run_nightly_eval(case_id, eval_file)
            if not report["slo_report"]["pass"]:
                all_passed = False
        except Exception as exc:
            logger.error("Eval failed for case '%s': %s", case_id, exc)
            all_passed = False

    if args.fail_on_violation and not all_passed:
        logger.error("SLO violations detected — exiting with code 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
