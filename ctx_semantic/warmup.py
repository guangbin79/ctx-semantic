"""Pre-embed missing chunks into the sidecar vector store (T7 warmup CLI).

Usage:
    uv run python -m ctx_semantic.warmup [--db PATH | --project DIR | --all | --prune]

--db embeds one explicit content DB; --project resolves a project directory
through projhash (same chain as the server: $CTX_SEMANTIC_DB >
$OPENCODE_PROJECT_DIR > the given dir); --all scans every content DB under
~/.config/opencode/context-mode/content/; --prune deletes stored vector
rows whose source DB no longer exists on disk (no model load). With no
flag the current project (cwd chain) is warmed. Run BEFORE registering the
server with opencode (T8 gate) so the first real query is model-load-only.

Reads context-mode DBs strictly read-only; writes go only to
~/ctx-semantic/data/vectors.db.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ctx_semantic import dbadapter, projhash, vectors
from ctx_semantic.binding import BoundAdapter
from ctx_semantic.embedder import MODEL_NAME, Embedder


def warm_one(db: Path, embedder: Embedder) -> dict[str, int]:
    """Sync one content DB into the store; prints counts + elapsed."""
    if not db.is_file():
        print(f"skip (no such db): {db}")
        return {"embedded": 0, "removed": 0, "total": 0}
    con = dbadapter.open_db(db)
    try:
        adapter = BoundAdapter(con, db)
        started = time.perf_counter()
        counts = vectors.sync(adapter, embedder.embed_batch, db)
        elapsed = time.perf_counter() - started
    finally:
        con.close()
    print(
        f"{db.name}: embedded={counts['embedded']} removed={counts['removed']}"
        f" total={counts['total']} elapsed={elapsed:.1f}s"
    )
    return counts


def prune(store_path: Path | None = None) -> dict[str, int]:
    """Delete vector rows whose source db_path no longer exists on disk.

    Returns {dead db_path: removed row count}; prints per-path counts.
    """
    con = vectors.connect(store_path)
    try:
        paths = [
            r[0]
            for r in con.execute(
                "SELECT DISTINCT db_path FROM embeddings ORDER BY db_path"
            )
        ]
        removed: dict[str, int] = {}
        for p in paths:
            if not Path(p).is_file():
                removed[p] = con.execute(
                    "DELETE FROM embeddings WHERE db_path=?", (p,)
                ).rowcount
        con.commit()
    finally:
        con.close()
    for p, n in removed.items():
        print(f"pruned {n} rows (dead db_path): {p}")
    print(
        f"prune: removed {sum(removed.values())} rows across {len(removed)} dead paths"
        f" ({len(paths) - len(removed)} live paths kept)"
    )
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctx_semantic.warmup",
        description="Pre-embed context-mode chunks into the vector store.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--db", type=Path, help="explicit content DB file path")
    group.add_argument("--project", type=Path, help="project directory (projhash chain)")
    group.add_argument(
        "--all", action="store_true", help="warm every content DB under the content dir"
    )
    group.add_argument(
        "--prune", action="store_true",
        help="remove vector rows whose source content DB no longer exists",
    )
    args = parser.parse_args(argv)

    if args.prune:
        prune()
        return 0

    if args.db is not None:
        dbs = [args.db.resolve()]  # store keys must be absolute, like the server's
    elif args.all:
        dbs = sorted(projhash.CONTENT_DIR.glob("*.db"))
    else:
        if args.project is not None and not args.project.is_dir():
            parser.error(f"project dir does not exist: {args.project}")
        dbs = [projhash.resolve_db(args.project)]

    if not dbs:
        print(f"no content DBs under {projhash.CONTENT_DIR}")
    embedder = Embedder()
    for db in dbs:
        warm_one(db, embedder)
    print(f"model={MODEL_NAME} device={embedder.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
