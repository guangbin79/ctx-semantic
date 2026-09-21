"""Tests for ctx_semantic.vectors — synthetic stubs only, /tmp fixtures.

The real T3 adapter and T4 embedder land in parallel; every test runs a stub
adapter over a fixture source DB in tmp_path plus a deterministic hash
embedder, per the T5 acceptance criteria. Nothing here touches
~/.config/opencode (asserted by test_no_writes_under_opencode_config).
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ctx_semantic.vectors import connect, search, sync

DIM = 8


class StubAdapter:
    """Minimal SourceAdapter over a fixture source DB (T3 stand-in)."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con

    def snapshot_chunks(self):
        return self.con.execute(
            "SELECT rowid, title, content, source_id, content_type, timestamp"
            " FROM chunks ORDER BY rowid"
        ).fetchall()

    def source_hashes(self):
        return dict(self.con.execute("SELECT id, content_hash FROM sources"))

    def live_rowids(self):
        return {r[0] for r in self.con.execute("SELECT rowid FROM chunks")}


class HashEmbedder:
    """Deterministic sha256-based vectors; records batched calls for dedup."""

    def __init__(self, delay: float = 0.0):
        self.calls: list[list[str]] = []
        self.delay = delay

    def __call__(self, texts):
        self.calls.append(list(texts))
        if self.delay:
            time.sleep(self.delay)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            d = hashlib.sha256(t.encode()).digest()
            out[i] = [b / 255.0 - 0.5 for b in d[:DIM]]
        return out


class PresetEmbedder:
    """Maps exact text -> preset vector (independent known-good table)."""

    def __init__(self, table: dict[str, np.ndarray]):
        self.table = table

    def __call__(self, texts):
        return np.stack([self.table[t] for t in texts])


def make_source(tmp_path: Path, n_sources: int = 2, per_source: int = 3):
    """Fixture source DB (context-mode shaped); returns (path, con, texts)."""
    path = tmp_path / "src.db"
    con = sqlite3.connect(path, check_same_thread=False)
    con.execute("CREATE TABLE sources(id INTEGER PRIMARY KEY, content_hash TEXT)")
    con.execute(
        "CREATE TABLE chunks(title TEXT, content TEXT, source_id INTEGER,"
        " content_type TEXT, timestamp REAL)"
    )
    texts: dict[int, list[str]] = {}
    for sid in range(1, n_sources + 1):
        con.execute("INSERT INTO sources VALUES(?,?)", (sid, f"hash-v1-{sid}"))
        for c in range(per_source):
            title, content = f"t{sid}-{c}", f"body {sid} {c} " + "x" * (c + 1)
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?)",
                (title, content, sid, "note", 1.0),
            )
            texts.setdefault(sid, []).append(f"{title}\n{content}")
    con.commit()
    return path, con, texts


def test_fresh_sync_then_search_top1(tmp_path):
    path, con, _ = make_source(tmp_path, 1, 3)
    store, emb = tmp_path / "v.db", HashEmbedder()
    rid, title, content = con.execute(
        "SELECT rowid, title, content FROM chunks WHERE rowid=2"
    ).fetchone()
    assert sync(StubAdapter(con), emb, path, store_path=store) == {
        "embedded": 3,
        "removed": 0,
        "total": 3,
    }
    q = emb([f"{title}\n{content}"])[0]
    got = search(connect(store), StubAdapter(con), path, q, 1)
    assert got[0][0] == rid
    assert got[0][1] == pytest.approx(1.0, abs=1e-5)


def test_hash_bump_reembeds_whole_source(tmp_path):
    path, con, _ = make_source(tmp_path, 2, 3)
    store, emb = tmp_path / "v.db", HashEmbedder()
    assert sync(StubAdapter(con), emb, path, store_path=store)["embedded"] == 6
    con.execute("UPDATE sources SET content_hash='hash-v2-1' WHERE id=1")
    con.commit()
    r = sync(StubAdapter(con), emb, path, store_path=store)
    assert r["embedded"] == 3 and r["removed"] == 0
    vc = connect(store)
    assert vc.execute(
        "SELECT count(*) FROM embeddings WHERE content_hash='hash-v2-1'"
    ).fetchone()[0] == 3
    assert vc.execute(
        "SELECT count(*) FROM embeddings WHERE content_hash='hash-v1-2'"
    ).fetchone()[0] == 3
    # misleading-success guard: counts vs actual store rows, not trusted
    assert vc.execute("SELECT count(*) FROM embeddings").fetchone()[0] == r["total"]


def test_deletion_reclaims_store_rows(tmp_path):
    path, con, _ = make_source(tmp_path, 1, 4)
    store = tmp_path / "v.db"
    assert sync(StubAdapter(con), HashEmbedder(), path, store_path=store)["total"] == 4
    con.execute("DELETE FROM chunks WHERE rowid=2")
    con.commit()
    r = sync(StubAdapter(con), HashEmbedder(), path, store_path=store)
    assert r == {"embedded": 0, "removed": 1, "total": 3}
    vc = connect(store)
    assert vc.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 3
    assert vc.execute(
        "SELECT count(*) FROM embeddings WHERE chunk_rowid=2"
    ).fetchone()[0] == 0


def test_hashless_source_reembeds_every_sync(tmp_path):
    # OCR #6 (branch taken: the live corpus has hash-less sources): a ''
    # hash means change detection is impossible, so those chunks re-embed
    # on EVERY sync; a real hash restores incremental behavior.
    path, con, _ = make_source(tmp_path, 1, 2)
    con.execute("UPDATE sources SET content_hash='' WHERE id=1")
    con.commit()
    store, emb = tmp_path / "v.db", HashEmbedder()
    assert sync(StubAdapter(con), emb, path, store_path=store)["embedded"] == 2
    assert sync(StubAdapter(con), emb, path, store_path=store)["embedded"] == 2
    con.execute("UPDATE sources SET content_hash='h1' WHERE id=1")
    con.commit()
    bumped = sync(StubAdapter(con), emb, path, store_path=store)
    assert bumped["embedded"] == 2  # '' -> 'h1' is a detected change
    assert sync(StubAdapter(con), emb, path, store_path=store) == {
        "embedded": 0, "removed": 0, "total": 2,
    }


def test_topk_matches_reference_bruteforce(tmp_path):
    path, con, texts = make_source(tmp_path, 3, 4)
    rng = np.random.default_rng(42)
    table = {
        t: rng.standard_normal(DIM).astype(np.float32)
        for ts in texts.values()
        for t in ts
    }
    store = tmp_path / "v.db"
    sync(StubAdapter(con), PresetEmbedder(table), path, store_path=store)
    rid2text = {r[0]: f"{r[1]}\n{r[2]}" for r in StubAdapter(con).snapshot_chunks()}
    q = rng.standard_normal(DIM).astype(np.float32)
    ids = list(table.keys())
    mat = np.stack([table[i] for i in ids])
    ref = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q))
    top5 = np.argsort(-ref, kind="stable")[:5]
    got = search(connect(store), StubAdapter(con), path, q, 5)
    assert [rid2text[r] for r, _ in got] == [ids[i] for i in top5]
    for i, j in enumerate(top5):
        assert got[i][1] == pytest.approx(float(ref[j]), abs=1e-5)


def test_search_filters_dead_rowids(tmp_path):
    path, con, _ = make_source(tmp_path, 1, 3)
    store, emb = tmp_path / "v.db", HashEmbedder()
    sync(StubAdapter(con), emb, path, store_path=store)
    rid, title, content = con.execute(
        "SELECT rowid, title, content FROM chunks WHERE rowid=1"
    ).fetchone()
    q = emb([f"{title}\n{content}"])[0]
    assert search(connect(store), StubAdapter(con), path, q, 3)[0][0] == rid
    con.execute("DELETE FROM chunks WHERE rowid=?", (rid,))
    con.commit()
    got = search(connect(store), StubAdapter(con), path, q, 2)
    assert len(got) == 2 and all(r != rid for r, _ in got)


def test_concurrent_sync_embeds_once(tmp_path):
    path, con, _ = make_source(tmp_path, 2, 3)
    store, emb = tmp_path / "v.db", HashEmbedder(delay=0.25)
    out: dict = {}

    def run():
        out["t"] = sync(StubAdapter(con), emb, path, store_path=store)

    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.05)  # let the thread grab the lock first
    main = sync(StubAdapter(con), emb, path, store_path=store)
    t.join()
    assert sum(map(len, emb.calls)) == 6, "each chunk embedded exactly once"
    assert out["t"]["embedded"] == 6 and main == {"embedded": 0, "removed": 0, "total": 6}
    assert connect(store).execute("SELECT count(*) FROM embeddings").fetchone()[0] == 6


def test_corrupt_store_delete_rebuilds(tmp_path):
    path, con, _ = make_source(tmp_path, 1, 4)
    store = tmp_path / "v.db"
    assert sync(StubAdapter(con), HashEmbedder(), path, store_path=store)["total"] == 4
    store.write_bytes(b"garbage")  # simulate corruption
    for suffix in ("", "-wal", "-shm"):
        Path(str(store) + suffix).unlink(missing_ok=True)
    r = sync(StubAdapter(con), HashEmbedder(), path, store_path=store)
    assert r == {"embedded": 4, "removed": 0, "total": 4}
    assert connect(store).execute("SELECT count(*) FROM embeddings").fetchone()[0] == 4


def test_dim_mismatch_raises(tmp_path):
    path, con, _ = make_source(tmp_path, 1, 2)
    store = tmp_path / "v.db"
    sync(StubAdapter(con), HashEmbedder(), path, store_path=store)
    with pytest.raises(ValueError, match="dim"):
        search(connect(store), StubAdapter(con), path, [0.0] * 4, 1)


def test_no_writes_under_opencode_config(tmp_path):
    root = Path.home() / ".config" / "opencode"
    files = [p for p in root.rglob("*") if p.is_file()][:2000]
    before = {str(p): p.stat().st_mtime_ns for p in files}
    db_count = len(list(root.rglob("*.db")))
    path, con, _ = make_source(tmp_path, 1, 3)
    store = tmp_path / "v.db"
    sync(StubAdapter(con), HashEmbedder(), path, store_path=store)
    search(connect(store), StubAdapter(con), path, np.ones(DIM, dtype=np.float32), 3)
    assert {str(p): p.stat().st_mtime_ns for p in files} == before
    assert len(list(root.rglob("*.db"))) == db_count
