"""Drift probe: model-free health check of the upstream context-mode contract.

Checks, in order: DB resolves (SKIP when unindexed), schema matches
EXPECTED_COLUMNS (hard drift), freshness heuristic (stale). Exit codes:

    0  PASS  — schema OK (or SKIP: nothing to check)
    1  DRIFT — schema changed; the fix kit is printed (same text the MCP
               tool error and the server preflight carry)
    2  STALE — schema OK but the resolved DB looks dormant; results would
               still serve, verify before trusting them

Designed for the post-ctx_upgrade check and cron. Never imports
ctx_semantic.embedder — no model load, no GPU, sub-second.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ctx_semantic import dbadapter, projhash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctx_semantic.probe",
        description="Health-check the upstream context-mode DB contract.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project dir (default: $OPENCODE_PROJECT_DIR or cwd)",
    )
    args = parser.parse_args(argv)

    try:
        db = projhash.resolve_db(args.project)
    except projhash.ProjHashError as exc:
        print(f"SKIP: cannot resolve project ({exc})")
        return 0
    if not db.is_file():
        print(f"SKIP: no content DB at {db} (unindexed project — empty KB, not an error)")
        return 0

    stale = projhash.stale_status(db)
    try:
        con = dbadapter.open_db(db)
    except dbadapter.SchemaDrift as exc:
        print(f"DRIFT: {db}\n{exc}")
        return 1
    except FileNotFoundError:
        print(f"SKIP: {db} vanished between resolve and open (queries will report)")
        return 0
    try:
        chunks = con.execute("SELECT count(*) FROM chunks").fetchone()[0]
    finally:
        con.close()

    if stale is not None:
        print(
            f"STALE: {db}\n  {stale}\n"
            f"  schema OK ({chunks} chunks) — verify before trusting results"
        )
        return 2
    print(f"PASS: {db} (schema OK, {chunks} chunks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
