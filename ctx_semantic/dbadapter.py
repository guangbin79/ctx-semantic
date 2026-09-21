"""Read-only adapter over context-mode's per-project FTS5 SQLite DBs.

Every connection opened here is read-only twice over: SQLite URI ``mode=ro``
plus ``PRAGMA query_only=ON``. Writers (the context-mode MCP) open/close per
operation and keep ``-wal``/``-shm`` around, so both connection setup and
queries can transiently hit SQLITE_BUSY-class errors; every entry point
retries 3 times with exponential backoff before giving up.

All functions take an explicit DB path or connection — this module has no
projhash dependency (see ctx_semantic.projhash for path resolution).
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypeVar

BUSY_TIMEOUT_MS = 2000
MAX_ATTEMPTS = 3
BACKOFF_S = 0.2

_T = TypeVar("_T")

# Column sets the live context-mode schema carries (verified 2026-09-20
# against 0ef2d5f23b410348.db). Any difference -> SchemaDrift.
EXPECTED_COLUMNS: dict[str, set[str]] = {
    "sources": {
        "id",
        "label",
        "chunk_count",
        "code_chunk_count",
        "indexed_at",
        "file_path",
        "content_hash",
    },
    "chunks": {
        "title",
        "content",
        "source_id",
        "content_type",
        "source_category",
        "session_id",
        "event_id",
        "timestamp",
    },
    "chunks_trigram": {
        "title",
        "content",
        "source_id",
        "content_type",
        "source_category",
        "session_id",
        "event_id",
        "timestamp",
    },
}

ChunkRow = tuple[int, str, str, int, str, str | None]


class SchemaDrift(Exception):
    """The DB schema differs from the context-mode layout this module expects."""


def _retryable(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _with_retry(run: Callable[[], _T]) -> _T:
    """Run a DB step, retrying SQLITE_BUSY-class failures up to MAX_ATTEMPTS."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return run()
        except sqlite3.OperationalError as exc:
            if not _retryable(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_S * 2**attempt)
    raise AssertionError("unreachable")  # for loop always returns or raises


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open a context-mode content DB strictly read-only.

    Retries transient open failures (SQLITE_CANTOPEN / SQLITE_READONLY_
    RECOVERY during WAL/-shm races) up to MAX_ATTEMPTS. Raises
    FileNotFoundError with a clear message when the file is absent, and
    SchemaDrift when tables/columns don't match the expected layout.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such content DB: {p}")
    uri = p.absolute().as_uri() + "?mode=ro"
    for attempt in range(MAX_ATTEMPTS):
        try:
            con = sqlite3.connect(uri, uri=True, isolation_level=None)
            break
        except sqlite3.OperationalError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_S * 2**attempt)
    try:
        con.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        con.execute("PRAGMA query_only=ON")
        # Schema reads can also hit SQLITE_BUSY under a writer's lock.
        _with_retry(lambda: _assert_schema(con))
    except BaseException:
        con.close()
        raise
    return con


def _assert_schema(con: sqlite3.Connection) -> None:
    for table, expected in EXPECTED_COLUMNS.items():
        actual = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        if not actual:
            raise SchemaDrift(f"missing table: {table}")
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise SchemaDrift(
                f"table {table} drifted:"
                f" missing={missing or 'none'} unexpected={extra or 'none'}"
            )


def _append_filters(
    sql: str, params: list[str], source_filter: str | None, content_type: str | None
) -> tuple[str, list[str]]:
    """Extend a ``FROM chunks AS c`` query with the shared filter clauses."""
    if source_filter is not None:
        sql += " JOIN sources AS s ON s.id = c.source_id WHERE s.label = ?"
        params.append(source_filter)
    if content_type is not None:
        sql += " WHERE " if "WHERE" not in sql else " AND "
        sql += "c.content_type = ?"
        params.append(content_type)
    return sql, params


def _select_chunks(
    source_filter: str | None, content_type: str | None
) -> tuple[str, list[str]]:
    sql = (
        "SELECT c.rowid, c.title, c.content, c.source_id, c.content_type,"
        " c.timestamp FROM chunks AS c"
    )
    params: list[str] = []
    sql, params = _append_filters(sql, params, source_filter, content_type)
    sql += " ORDER BY c.rowid"
    return sql, params


def list_chunks(
    con: sqlite3.Connection,
    source_filter: str | None = None,
    content_type: str | None = None,
) -> list[ChunkRow]:
    """All chunks as (rowid, title, content, source_id, content_type, timestamp).

    source_filter matches sources.label exactly; content_type matches the
    chunks content_type column exactly. Both optional.
    """
    sql, params = _select_chunks(source_filter, content_type)
    return _with_retry(lambda: con.execute(sql, params).fetchall())


def snapshot_chunks(
    con: sqlite3.Connection,
    source_filter: str | None = None,
    content_type: str | None = None,
) -> list[ChunkRow]:
    """list_chunks inside ONE explicit read transaction.

    A deferred BEGIN pins the WAL snapshot on the first read, so a concurrent
    writer committing mid-fetch cannot tear the row set. Use this (not
    list_chunks) as the source of truth for embed-sync scans.
    """
    sql, params = _select_chunks(source_filter, content_type)

    def run() -> list[tuple]:
        con.execute("BEGIN")
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.execute("COMMIT")  # read txn; commit just releases the snapshot

    return _with_retry(run)


def _escape_fts(query: str) -> str | None:
    """Neutralize FTS5 query syntax: every token becomes a quoted phrase.

    ``a AND b*`` -> ``"a" "AND" "b*"`` — operators/wildcards degrade to literal
    tokens instead of crashing MATCH. Tokens with no alphanumeric character
    (e.g. "???") are dropped: FTS5 builds exist that reject phrases which
    tokenize to zero tokens. Returns None when nothing survives (whitespace-
    or punctuation-only query) — an empty BM25 leg, not an error.
    """
    tokens = [t for t in query.split() if any(ch.isalnum() for ch in t)]
    if not tokens:
        return None
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def bm25_search(
    con: sqlite3.Connection,
    query: str,
    limit: int,
    source_filter: str | None = None,
    content_type: str | None = None,
) -> list[tuple[int, float]]:
    """Full-text search returning (rowid, bm25 score) — lower score is better.

    source_filter (exact sources.label) and content_type (exact) restrict
    matches WHERE-side, so the BM25 leg of hybrid search honors them natively.
    """
    match_expr = _escape_fts(query)
    if match_expr is None:
        return []
    sql = (
        "SELECT rowid, bm25(chunks) AS score FROM chunks WHERE chunks MATCH ?"
    )
    params: list[str | int] = [match_expr]
    if source_filter is not None:
        sql += " AND source_id IN (SELECT id FROM sources WHERE label = ?)"
        params.append(source_filter)
    if content_type is not None:
        sql += " AND content_type = ?"
        params.append(content_type)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return _with_retry(
        lambda: con.execute(sql, params).fetchall()
    )


def get_many(
    con: sqlite3.Connection, rowids: Iterator[int] | list[int]
) -> dict[int, tuple[str, str, str, str, str | None]]:
    """Fetch chunks for window extraction.

    Returns {rowid: (title, content, source label, content_type, timestamp)}.
    Rowids not present in the DB are silently omitted.
    """
    ids = list(rowids)
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = _with_retry(
        lambda: con.execute(
            "SELECT c.rowid, c.title, c.content, s.label, c.content_type,"
            f" c.timestamp FROM chunks AS c"
            f" LEFT JOIN sources AS s ON s.id = c.source_id"
            f" WHERE c.rowid IN ({placeholders})",
            ids,
        ).fetchall()
    )
    return {r[0]: (r[1], r[2], r[3], r[4], r[5]) for r in rows}


def filtered_rowids(
    con: sqlite3.Connection,
    source_filter: str | None = None,
    content_type: str | None = None,
) -> set[int]:
    """Live chunk rowids surviving the filters (bounds the vector leg)."""
    sql, params = _append_filters(
        "SELECT c.rowid FROM chunks AS c", [], source_filter, content_type
    )
    return {r[0] for r in _with_retry(lambda: con.execute(sql, params).fetchall())}


def live_rowids(con: sqlite3.Connection) -> set[int]:
    """All live chunk rowids (vectors.SourceAdapter.live_rowids)."""
    return {
        r[0] for r in _with_retry(lambda: con.execute("SELECT rowid FROM chunks").fetchall())
    }


def source_hashes(con: sqlite3.Connection) -> dict[int, str]:
    """sources.id -> content_hash ('' when NULL) for embed change detection."""
    return dict(
        _with_retry(
            lambda: con.execute(
                "SELECT id, coalesce(content_hash, '') FROM sources"
            ).fetchall()
        )
    )
