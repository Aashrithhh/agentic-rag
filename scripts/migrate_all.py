#!/usr/bin/env python
"""Run Alembic migrations for ALL registered cases.

Usage::

    python scripts/migrate_all.py              # upgrade all to head
    python scripts/migrate_all.py --revision base  # downgrade all
    python scripts/migrate_all.py --case big-thorium  # single case
"""

from __future__ import annotations

import argparse
import subprocess
import sys

# Ensure project root is on sys.path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from app.cases import list_case_ids


def run_alembic(case_id: str, revision: str = "head") -> bool:
    """Run alembic upgrade/downgrade for one case, return True on success."""
    cmd = ["alembic", "-x", f"case_id={case_id}", "upgrade", revision]
    print(f"  [{case_id}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{case_id}] FAILED:\n{result.stderr}")
        return False
    print(f"  [{case_id}] OK")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Alembic migrations for all cases.")
    parser.add_argument("--revision", default="head", help="Target revision (default: head)")
    parser.add_argument("--case", default=None, help="Migrate only this case_id")
    args = parser.parse_args()

    cases = [args.case] if args.case else list_case_ids()
    if not cases:
        print("No cases found. Register cases in app/cases.py first.")
        sys.exit(1)

    print(f"Migrating {len(cases)} case(s) to '{args.revision}'...\n")

    failed = []
    for cid in cases:
        if not run_alembic(cid, args.revision):
            failed.append(cid)

    print(f"\nDone. {len(cases) - len(failed)}/{len(cases)} succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
