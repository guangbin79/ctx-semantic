"""Sidecar vector store with lazy sync for context-mode content DBs.

The store OWNS exactly one SQLite file — DEFAULT_STORE (~/ctx-semantic/data/
vectors.db), overridable per call via store_path — and never writes a single
byte to the source (context-mode) DBs: the caller's adapter opens those
read-only. Deleting the store file is always a safe full rebuild (sync()
recreates it from scratch).

T3 (adapter) and T4 (embedder) land in parallel; this module codes only
against the narrow SourceAdapter protocol plus a fastembed-shaped embedder
callable (``embed(texts: list[str]) -> iterable of 1-D vectors``), so tests
run on synthetic stubs and T6/T7 do the real wiring.
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "jinaai/jina-embeddings-v2-base-zh"
DEFAULT_STORE = Path.home() / "ctx-semantic" / "data" / "vectors.db"

_DDL = """
CREATE TABLE IF NOT EXISTS embeddings(
    db_path TEXT NOT NULL,
    chunk_rowid INTEGER NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vec BLOB NOT NULL,
    embedded_at REAL NOT NULL,
    PRIMARY KEY(db_path, chunk_rowid, model)
)
"""

# process-local: syncs for one (store, source db_path) never overlap
_sync_locks: dict[tuple[str, str], threading.Lock] = {}
_sync_locks_guard = threading.Lock()


class SourceAdapter(Protocol):
    """Read-only view of ONE context-mode content DB (T3 implements this).

    snapshot_chunks() must return (rowid, title, content, source_id,
    content_type, timestamp) tuples captured inside one transaction, so
    rowids and source hashes stay mutually consistent. Chunks whose
    source_id has no entry in source_hashes() store hash "" and are
    re-embedded on every sync.
    """

    def snapshot_chunks(self) -> list[tuple[int, str, str, int, str, float]]: ...

    def source_hashes(self) -> dict[int, str]: ...

    def live_rowids(self) -> set[int]: ...


Embedder = Callable[[list[str]], Iterable[np.ndarray]]


def _sync_lock(key: tuple[str, str]) -> threading.Lock:
    with _sync_locks_guard:
        return _sync_locks.setdefault(key, threading.Lock())


def connect(store_path: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """Open (creating parent dirs and schema as needed) the store in WAL mode."""
    path = Path(store_path) if store_path else DEFAULT_STORE
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(_DDL)
    con.commit()
    return con


def sync(
    adapter: SourceAdapter,
    embedder: Embedder,
    db_path: str | os.PathLike[str],
    model: str = DEFAULT_MODEL,
    store_path: str | os.PathLike[str] | None = None,
) -> dict[str, int]:
    """Incrementally bring the store in line with the source DB (read-only).

    Every live chunk rowid that is missing from the store, or whose
    source-level content_hash changed, gets embedded as ``title\\ncontent``
    and upserted; stored rows whose rowid no longer exists in the source are
    deleted. Returns {"embedded", "removed", "total"} — total counted back
    from the store, not accumulated arithmetically.

    Known granularity: content_hash lives on the sources table, so one
    changed file re-embeds ALL chunks of that source — coarse but safe.

    Concurrent syncs for the same (store, db_path) serialize on a process-
    local lock: the second caller waits, then finds nothing left to embed,
    so chunks are never embedded twice.

    A corrupted store file is recovered by deleting it and calling sync()
    again (fresh empty store == full rebuild).
    """
    store = Path(store_path) if store_path else DEFAULT_STORE
    src_key = str(db_path)
    with _sync_lock((str(store), src_key)):
        chunks = adapter.snapshot_chunks()
        hashes = adapter.source_hashes()
        live = {row[0] for row in chunks}
        con = connect(store)
        try:
            stored = dict(
                con.execute(
                    "SELECT chunk_rowid, content_hash FROM embeddings"
                    " WHERE db_path=? AND model=?",
                    (src_key, model),
                ).fetchall()
            )
            todo = [
                (rid, title, content, hashes.get(sid, ""))
                for rid, title, content, sid, _ct, _ts in chunks
                if stored.get(rid) != hashes.get(sid, "")
            ]
            stale = [rid for rid in stored if rid not in live]
            embedded = 0
            if todo:
                texts = [f"{title}\n{content}" for _, title, content, _ in todo]
                # generator-friendly: insert as vectors stream out of embed()
                for (rid, _ti, _co, src_hash), vec in zip(
                    todo, embedder(texts), strict=True
                ):
                    v = np.asarray(vec, dtype=np.float32)
                    con.execute(
                        "INSERT OR REPLACE INTO embeddings"
                        "(db_path, chunk_rowid, model, content_hash, dim, vec,"
                        " embedded_at) VALUES(?,?,?,?,?,?,?)",
                        (
                            src_key,
                            rid,
                            model,
                            src_hash,
                            v.size,
                            v.tobytes(),
                            time.time(),
                        ),
                    )
                    embedded += 1
            con.executemany(
                "DELETE FROM embeddings WHERE db_path=? AND model=? AND chunk_rowid=?",
                [(src_key, model, rid) for rid in stale],
            )
            con.commit()
            total = con.execute(
                "SELECT count(*) FROM embeddings WHERE db_path=? AND model=?",
                (src_key, model),
            ).fetchone()[0]
        finally:
            con.close()
    return {"embedded": embedded, "removed": len(stale), "total": total}


def search(
    vectors_con: sqlite3.Connection,
    adapter: SourceAdapter,
    db_path: str | os.PathLike[str],
    qvec: Iterable[float],
    k: int,
    model: str = DEFAULT_MODEL,
) -> list[tuple[int, float]]:
    """Top-k stored (chunk_rowid, cosine) for qvec, best first.

    Loads all vectors for (db_path, model) into memory as float32, L2-
    normalizes, and matmuls against the normalized query. Rowids that no
    longer exist in the source chunks table are filtered out via the
    adapter BEFORE the top-k cut, so k results are live results.
    Raises ValueError when qvec's dim differs from the stored dim.
    """
    # ponytail: brute-force cosine over ≤~100k chunks is sub-ms here; switch to sqlite-vec/HNSW beyond that
    rows = vectors_con.execute(
        "SELECT chunk_rowid, dim, vec FROM embeddings"
        " WHERE db_path=? AND model=? ORDER BY chunk_rowid",
        (str(db_path), model),
    ).fetchall()
    if not rows:
        return []
    dim = rows[0][1]
    q = np.asarray(list(qvec), dtype=np.float32)
    if q.size != dim:
        raise ValueError(f"query dim {q.size} != stored dim {dim}")
    live_rows = adapter.live_rowids()
    kept = [
        (rid, np.frombuffer(vec, dtype=np.float32))
        for rid, _d, vec in rows
        if rid in live_rows
    ]
    if not kept:
        return []
    ids = [rid for rid, _ in kept]
    mat = np.stack([v for _, v in kept])
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = (mat @ qn) / np.maximum(np.linalg.norm(mat, axis=1), 1e-12)
    order = np.argsort(-scores, kind="stable")[:k]
    return [(ids[i], float(scores[i])) for i in order]
