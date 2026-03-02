#!/usr/bin/env python
"""Backup per-case PostgreSQL databases using pg_dump.

Usage::

    python scripts/backup_db.py --case big-thorium     # single case
    python scripts/backup_db.py --all                   # all cases
    python scripts/backup_db.py --all --output-dir /mnt/backups

Outputs gzip-compressed SQL files to ``backups/`` (or ``--output-dir``).
Requires ``pg_dump`` on PATH.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cases import list_case_ids
from app.config import settings


def _parse_db_url(url: str) -> dict[str, str]:
    """Extract host, port, user, password from a SQLAlchemy-style URL."""
    # Strip the +driver suffix: postgresql+psycopg://... → postgresql://...
    clean = url.replace("+psycopg", "").replace("+asyncpg", "")
    parsed = urlparse(clean)
    return {
        "host": parsed.hostname or "localhost",
        "port": str(parsed.port or 5432),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


def backup_case(case_id: str, output_dir: Path) -> bool:
    """Dump one case database to a gzip file. Returns True on success."""
    db_name = f"audit_rag_{case_id.replace('-', '_')}"
    conn = _parse_db_url(settings.database_url)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"{db_name}_{timestamp}.sql.gz"

    cmd = [
        "pg_dump",
        "-h", conn["host"],
        "-p", conn["port"],
        "-U", conn["user"],
        "--no-password",
        "-d", db_name,
    ]

    env = {**__import__("os").environ}
    if conn["password"]:
        env["PGPASSWORD"] = conn["password"]

    print(f"  [{case_id}] Dumping {db_name} → {out_file}")

    try:
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=600)
    except FileNotFoundError:
        print("  ERROR: pg_dump not found on PATH. Install PostgreSQL client tools.")
        return False
    except subprocess.TimeoutExpired:
        print(f"  [{case_id}] TIMEOUT after 600s")
        return False

    if result.returncode != 0:
        print(f"  [{case_id}] FAILED: {result.stderr.decode()[:500]}")
        return False

    with gzip.open(out_file, "wb") as f:
        f.write(result.stdout)

    size_kb = out_file.stat().st_size / 1024
    print(f"  [{case_id}] OK — {size_kb:.1f} KB")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup per-case PostgreSQL databases.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", help="Backup a single case_id")
    group.add_argument("--all", action="store_true", help="Backup all registered cases")
    parser.add_argument("--output-dir", default="backups", help="Output directory (default: backups/)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = list_case_ids() if args.all else [args.case]
    if not cases:
        print("No cases found. Register cases in app/cases.py first.")
        sys.exit(1)

    print(f"Backing up {len(cases)} case(s)...\n")

    failed = []
    for cid in cases:
        if not backup_case(cid, output_dir):
            failed.append(cid)

    print(f"\nDone. {len(cases) - len(failed)}/{len(cases)} succeeded.")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
