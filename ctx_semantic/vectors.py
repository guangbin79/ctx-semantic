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

import hashlib
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

# process-level matrix cache: (store file, db_path, model) -> (version,
# L2-normalized float32 matrix, rowids array). sync() bumps the key's
# version after every commit, so a stale entry never serves. In-memory
# stores (PRAGMA file == '') bypass the cache: each :memory: connection
# is a distinct store, so a shared entry would leak one store's vectors
# into another. No size cap — one process serves few distinct DBs
# (server: one per project session); add an LRU if that ever changes.
_MatrixKey = tuple[str, str, str]
_matrix_cache: dict[_MatrixKey, tuple[int, np.ndarray, np.ndarray]] = {}
_matrix_versions: dict[_MatrixKey, int] = {}
_matrix_lock = threading.Lock()


class SourceAdapter(Protocol):
    """Read-only view of ONE context-mode content DB (T3 implements this).

    snapshot_chunks() must return (rowid, title, content, source_id,
    content_type, timestamp) tuples captured inside one transaction, so
    rowids and chunk text stay mutually consistent.
    """

    def snapshot_chunks(self) -> list[tuple[int, str, str, int, str, str | None]]: ...

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


def _chunk_hash(title: str, content: str) -> str:
    """Staleness key for one chunk: 16 hex chars of sha256(title\ncontent)."""
    return hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()[:16]


def sync(
    adapter: SourceAdapter,
    embedder: Embedder,
    db_path: str | os.PathLike[str],
    model: str = DEFAULT_MODEL,
    store_path: str | os.PathLike[str] | None = None,
) -> dict[str, int]:
    """Incrementally bring the store in line with the source DB (read-only).

    Staleness is CHUNK-level: each row's content_hash is the sha256 of
    its own ``title\\ncontent`` computed at embed time, so exactly the
    chunks whose text or title changed (plus missing rowids) re-embed —
    hash-less sources included, because source hashes are never consulted.
    Stored rows whose rowid no longer exists in the source are deleted.
    Returns {"embedded", "removed", "total"} — total counted back
    from the store, not accumulated arithmetically.

    Migration: rows written before chunk-level hashing hold SOURCE-level
    hashes, so the first sync after upgrading re-embeds each DB once.

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
        live = {row[0] for row in chunks}
        con = connect(store)
        mkey = _matrix_key(con, src_key, model)
        try:
            stored = dict(
                con.execute(
                    "SELECT chunk_rowid, content_hash FROM embeddings"
                    " WHERE db_path=? AND model=?",
                    (src_key, model),
                ).fetchall()
            )
            todo = [
                (rid, title, content, _chunk_hash(title, content))
                for rid, title, content, _sid, _ct, _ts in chunks
                if stored.get(rid) != _chunk_hash(title, content)
            ]
            stale = [rid for rid in stored if rid not in live]
            embedded = 0
            if todo:
                texts = [f"{title}\n{content}" for _, title, content, _ in todo]
                # generator-friendly: insert as vectors stream out of embed()
                for (rid, _ti, _co, chash), vec in zip(
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
                            chash,
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
            with _matrix_lock:  # invalidate cached matrices for this key
                _matrix_versions[mkey] = _matrix_versions.get(mkey, 0) + 1
            total = con.execute(
                "SELECT count(*) FROM embeddings WHERE db_path=? AND model=?",
                (src_key, model),
            ).fetchone()[0]
        finally:
            con.close()
    return {"embedded": embedded, "removed": len(stale), "total": total}


def _matrix_key(
    con: sqlite3.Connection, db_path: str | os.PathLike[str], model: str
) -> _MatrixKey | None:
    """Cache key for this (store, source, model); None for in-memory stores."""
    file = con.execute("PRAGMA database_list").fetchone()[2]
    return None if not file else (file, str(db_path), model)


def _load_matrix(
    key: _MatrixKey | None,
    vectors_con: sqlite3.Connection,
    db_path: str | os.PathLike[str],
    model: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """(normalized matrix, rowids) for the key, from cache or SQLite.

    The version is read BEFORE the SELECT: an entry is only served while
    its label still matches _matrix_versions, so a sync committing mid-
    load can only cause one wasted reload, never a stale hit.
    """
    if key is not None:
        with _matrix_lock:
            version = _matrix_versions.get(key, 0)
            entry = _matrix_cache.get(key)
        if entry is not None and entry[0] == version:
            return entry[1], entry[2]
    rows = vectors_con.execute(
        "SELECT chunk_rowid, dim, vec FROM embeddings"
        " WHERE db_path=? AND model=? ORDER BY chunk_rowid",
        (str(db_path), model),
    ).fetchall()
    if not rows:
        return None
    rowids = np.array([r[0] for r in rows], dtype=np.int64)
    mat = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    if key is not None:
        with _matrix_lock:
            _matrix_cache[key] = (version, mat, rowids)
    return mat, rowids


def search(
    vectors_con: sqlite3.Connection,
    adapter: SourceAdapter,
    db_path: str | os.PathLike[str],
    qvec: Iterable[float],
    k: int,
    model: str = DEFAULT_MODEL,
) -> list[tuple[int, float]]:
    """Top-k stored (chunk_rowid, cosine) for qvec, best first.

    Vectors for (db_path, model) load from SQLite once per process into
    an L2-normalized float32 matrix cached per (store file, db_path,
    model); sync() invalidates the entry by bumping its version after
    every commit, and in-memory stores bypass the cache entirely. Rowids
    that no longer exist in the source chunks table are filtered out via
    the adapter BEFORE the top-k cut on EVERY call, so k results are live
    results. Raises ValueError when qvec's dim differs from the stored dim.
    """
    # ponytail: brute-force cosine over ≤~100k chunks is sub-ms here; switch to sqlite-vec/HNSW beyond that
    loaded = _load_matrix(
        _matrix_key(vectors_con, db_path, model), vectors_con, db_path, model
    )
    if loaded is None:
        return []
    mat, rowids = loaded
    q = np.asarray(list(qvec), dtype=np.float32)
    if q.size != mat.shape[1]:
        raise ValueError(f"query dim {q.size} != stored dim {mat.shape[1]}")
    mask = np.isin(rowids, list(adapter.live_rowids()))
    if not mask.any():
        return []
    qn = q / max(float(np.linalg.norm(q)), 1e-12)
    scores = mat[mask] @ qn
    kept_ids = rowids[mask]
    order = np.argsort(-scores, kind="stable")[:k]
    return [(int(kept_ids[i]), float(scores[i])) for i in order]
