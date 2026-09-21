"""Tests for ctx_semantic.server (T7): binding filters, tool wiring, cold start.

The real embedding model is NEVER loaded — the server's lazy Embedder global
is replaced by a deterministic stub (bag-of-chars 4-dim vectors, stable
across runs). The vector leg runs the REAL vectors.search over a per-test
store; the FTS leg runs the real dbadapter over a context-mode-shaped
fixture DB. Schema assertions hit the real MCPServer registration.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from ctx_semantic import dbadapter, hybrid, projhash, server, vectors
from ctx_semantic.binding import BoundAdapter

DIM = 4

SOURCES_DDL = """
CREATE TABLE sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT NOT NULL,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        code_chunk_count INTEGER NOT NULL DEFAULT 0,
        indexed_at TEXT NOT NULL DEFAULT (datetime('now')),
        file_path TEXT,
        content_hash TEXT
      )
"""

FTS_DDL = """
CREATE VIRTUAL TABLE chunks USING fts5(
        title,
        content,
        source_id UNINDEXED,
        content_type UNINDEXED,
        source_category UNINDEXED,
        session_id UNINDEXED,
        event_id UNINDEXED,
        timestamp UNINDEXED,
        tokenize='porter unicode61'
      )
"""

FTS_TRIGRAM_DDL = FTS_DDL.replace("chunks USING", "chunks_trigram USING").replace(
    "tokenize='porter unicode61'", "tokenize='trigram'"
)

# (title, content, source_id, content_type): 1 alpha/session, 2 beta/decision,
# 3 beta/session — "deploy" matches rids 1 and 2 only.
CHUNKS = [
    ("alpha deploy guide", "deploy the service with docker compose on the cluster", 1, "session"),
    ("beta deploy decision", "deploy using blue green rollout strategy", 2, "decision"),
    ("beta garden notes", "roses and tulips seasonal care calendar", 2, "session"),
]


def make_db(path: Path) -> Path:
    con = sqlite3.connect(path)
    try:
        con.execute(SOURCES_DDL)
        con.execute(FTS_DDL)
        con.execute(FTS_TRIGRAM_DDL)
        con.execute("INSERT INTO sources (id, label, content_hash) VALUES (1, 'alpha', 'h1')")
        con.execute("INSERT INTO sources (id, label, content_hash) VALUES (2, 'beta', 'h2')")
        for title, content, sid, ctype in CHUNKS:
            con.execute(
                "INSERT INTO chunks (title, content, source_id, content_type)"
                " VALUES (?, ?, ?, ?)",
                (title, content, sid, ctype),
            )
        con.commit()
    finally:
        con.close()
    return path


class StubEmbedder:
    """Deterministic embedder: bag-of-chars unit vectors, no model load."""

    def __init__(self) -> None:
        self.batch_calls = 0

    @staticmethod
    def _vec(text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for ch in text:
            v[ord(ch) % DIM] += 1.0
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            v[0] = 1.0
            return v
        return v / norm

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self.batch_calls += 1
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@pytest.fixture
def tool_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated store + stubbed lazy embedder; returns the fixture DB path."""
    db = make_db(tmp_path / "src.db")
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))
    monkeypatch.setattr(vectors, "DEFAULT_STORE", tmp_path / "store.db")
    monkeypatch.setattr(server, "_embedder", None)  # teardown restores prior value
    monkeypatch.setattr(server, "Embedder", StubEmbedder)
    monkeypatch.setattr(server, "time", time)  # undo faked clocks from other tests
    server._synced_at.clear()
    server._sync_interval.clear()
    return db


# --- T6 gap closed: filters genuinely restrict BOTH legs --------------------


def test_filters_restrict_both_legs(tool_env: Path):
    con = dbadapter.open_db(tool_env)
    try:
        adapter = BoundAdapter(con, tool_env)
        # BM25 leg, WHERE-side: unfiltered "deploy" hits rids 1+2; each filter
        # drops the non-matching row before ranking; ghost filter drops all.
        assert {r[0] for r in adapter.bm25_search("deploy", 5)} == {1, 2}
        assert {r[0] for r in adapter.bm25_search("deploy", 5, source="beta")} == {2}
        assert {r[0] for r in adapter.bm25_search("deploy", 5, source="alpha")} == {1}
        assert adapter.bm25_search("deploy", 5, source="ghost") == []
        assert {r[0] for r in adapter.bm25_search("deploy", 5, content_type="decision")} == {2}
        # Vector leg bound: filtered_rowids feeds hybrid's _Narrowed view.
        assert adapter.filtered_rowids(source="alpha") == {1}
        assert adapter.filtered_rowids(source="beta") == {2, 3}
        assert adapter.filtered_rowids(content_type="session") == {1, 3}
        assert adapter.filtered_rowids(source="ghost") == set()
    finally:
        con.close()
    # End-to-end through the tool: source=alpha surfaces ONLY alpha's chunk
    # even though beta's chunk matches the query just as well.
    out = server.ctx_hybrid_search(queries=["deploy"], source="alpha", limit=3)
    assert "alpha deploy guide" in out
    assert "beta deploy decision" not in out and "beta garden notes" not in out


def test_ghost_source_filter_is_friendly_empty(tool_env: Path):
    assert server.ctx_hybrid_search(queries=["deploy"], source="ghost") == hybrid.NO_RESULTS


def test_content_type_filter_end_to_end(tool_env: Path):
    out = server.ctx_hybrid_search(queries=["deploy"], content_type="decision", limit=3)
    assert "beta deploy decision" in out
    assert "alpha deploy guide" not in out


# --- registration + conditional project_path --------------------------------


def test_single_tool_registered_with_expected_schema():
    import asyncio

    tools = asyncio.run(server.mcp.list_tools())
    assert [t.name for t in tools] == ["ctx_hybrid_search"]
    schema = tools[0].input_schema
    assert set(schema["properties"]) == {
        "queries", "source", "content_type", "limit", "project_path",
    }
    assert schema["required"] == ["queries"]


def test_project_path_conditional_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_env: Path
):
    monkeypatch.setattr(projhash, "CONTENT_DIR", tmp_path)
    monkeypatch.delenv("CTX_SEMANTIC_DB", raising=False)
    proj_a = tmp_path / "projA"
    proj_b = tmp_path / "projB"
    proj_a.mkdir()
    proj_b.mkdir()
    db_a = projhash.db_path(projhash.resolve(proj_a))
    make_db(db_a)

    # explicit project_path overrides OPENCODE_PROJECT_DIR
    monkeypatch.setenv("OPENCODE_PROJECT_DIR", str(proj_b))
    out = server.ctx_hybrid_search(queries=["deploy"], project_path=str(proj_a))
    assert "alpha deploy guide" in out
    # no explicit param: env dir wins; projB was never indexed -> empty base
    assert server.ctx_hybrid_search(queries=["deploy"]) == hybrid.NO_RESULTS
    # env dir pointing at the indexed project works without project_path
    monkeypatch.setenv("OPENCODE_PROJECT_DIR", str(proj_a))
    assert "alpha deploy guide" in server.ctx_hybrid_search(queries=["deploy"])
    # CTX_SEMANTIC_DB beats even the explicit project_path (projhash contract)
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(tool_env))
    out = server.ctx_hybrid_search(queries=["deploy"], project_path=str(proj_b))
    assert "alpha deploy guide" in out


# --- cold start: model lazy, sync once, progress line -----------------------


def test_cold_start_lazy_construct_and_sync_once(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    assert server._embedder is None  # import did not construct an embedder

    sync_calls: list[int] = []
    real_sync = vectors.sync

    def counting_sync(*args, **kwargs):
        sync_calls.append(1)
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server.vectors, "sync", counting_sync)

    out1 = server.ctx_hybrid_search(queries=["deploy"])
    assert server._embedder is not None  # constructed on first call...
    assert isinstance(server._embedder, StubEmbedder)  # ...from the patched global
    assert sync_calls == [1]
    assert out1.startswith("(embedded 3 chunks in ")  # progress line once
    assert "alpha deploy guide" in out1

    out2 = server.ctx_hybrid_search(queries=["deploy"])
    assert sync_calls == [1]  # no re-sync for the same DB in-process
    assert "(embedded" not in out2  # and no second progress line
    assert "alpha deploy guide" in out2


# --- OCR #3/#4/#5: TTL re-sync, thread-safe cold start, input validation ---


class FakeClock:
    """Stand-in for the time module: time(), monotonic(), perf_counter()."""

    def __init__(self) -> None:
        self.now = 1000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now


def test_resync_after_ttl(tool_env: Path, monkeypatch: pytest.MonkeyPatch):
    clock = FakeClock()
    monkeypatch.setattr(server, "time", clock)
    sync_calls: list[int] = []
    real_sync = vectors.sync

    def counting_sync(*args, **kwargs):
        sync_calls.append(1)
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server.vectors, "sync", counting_sync)

    server.ctx_hybrid_search(queries=["deploy"])
    assert sync_calls == [1]
    # the cold sync embedded everything (3/3) -> hash-less backoff window;
    # pin the standard interval so THIS test exercises the 300s boundary
    with server._state_lock:
        server._sync_interval[str(tool_env)] = server.RESYNC_INTERVAL_S
    clock.now += server.RESYNC_INTERVAL_S - 1  # inside TTL: no re-sync
    server.ctx_hybrid_search(queries=["deploy"])
    assert sync_calls == [1]
    clock.now += 2  # past TTL: vector leg goes stale, sync again
    server.ctx_hybrid_search(queries=["deploy"])
    assert sync_calls == [1, 1]


class SplitClock:
    """time() jumps with the wall clock; monotonic()/perf_counter() do not."""

    def __init__(self) -> None:
        self.wall = 1_000_000.0
        self.mono = 500.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.mono

    def perf_counter(self) -> float:
        return self.mono


def _counting_sync(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Patch vectors.sync with a call counter; returns the counter list."""
    sync_calls: list[int] = []
    real_sync = vectors.sync

    def counting_sync(*args, **kwargs):
        sync_calls.append(1)
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server.vectors, "sync", counting_sync)
    return sync_calls


def test_ttl_expiry_sync_runs_once_across_threads(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    server.ctx_hybrid_search(queries=["deploy"])  # initial sync claims the key
    key = str(tool_env)
    with server._state_lock:
        server._synced_at[key] -= (  # force expiry under whatever interval applies
            server._sync_interval.get(key, server.RESYNC_INTERVAL_S) + 1
        )

    sync_calls: list[int] = []
    real_sync = vectors.sync
    real_sleep = time.sleep

    def counting_slow_sync(*args, **kwargs):  # widen the race window
        sync_calls.append(1)
        real_sleep(0.02)
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server.vectors, "sync", counting_slow_sync)
    barrier = threading.Barrier(2)

    def query():
        barrier.wait()  # both threads hit the stale check together
        server.ctx_hybrid_search(queries=["deploy"])

    threads = [threading.Thread(target=query) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sync_calls) == 1  # the claim dedups: exactly one sync ran


def test_sync_ttl_ignores_wall_clock_jumps(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    clock = SplitClock()
    monkeypatch.setattr(server, "time", clock)
    sync_calls = _counting_sync(monkeypatch)

    server.ctx_hybrid_search(queries=["deploy"])  # claimed at mono=500
    clock.wall -= 3600  # NTP step backwards — must not extend freshness
    server.ctx_hybrid_search(queries=["deploy"])
    clock.wall += 7200  # NTP step far forwards — must not fake expiry
    server.ctx_hybrid_search(queries=["deploy"])
    assert len(sync_calls) == 1  # monotonic age is still ~0


def test_hashless_backoff_after_full_reembed(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    clock = FakeClock()
    monkeypatch.setattr(server, "time", clock)
    sync_calls = _counting_sync(monkeypatch)
    key = str(tool_env)

    server.ctx_hybrid_search(queries=["deploy"])  # cold sync: embedded=3=total
    assert server._sync_interval[key] == server.HASHLESS_RESYNC_INTERVAL_S

    clock.now += server.RESYNC_INTERVAL_S + 1  # past the STANDARD interval
    server.ctx_hybrid_search(queries=["deploy"])  # backoff holds: no sync
    assert len(sync_calls) == 1

    clock.now += server.HASHLESS_RESYNC_INTERVAL_S  # past the long interval
    con = sqlite3.connect(tool_env)
    con.execute("UPDATE sources SET content_hash='h2b' WHERE id=2")  # 2 of 3
    con.commit()
    con.close()
    server.ctx_hybrid_search(queries=["deploy"])  # partial sync (2 < 3)
    assert len(sync_calls) == 2
    assert server._sync_interval[key] == server.RESYNC_INTERVAL_S

    clock.now += server.RESYNC_INTERVAL_S + 1  # standard window applies again
    server.ctx_hybrid_search(queries=["deploy"])
    assert len(sync_calls) == 3


def test_failed_sync_resets_claim_for_retry(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[int] = []
    real_sync = vectors.sync

    def flaky_sync(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("embedder exploded")
        return real_sync(*args, **kwargs)

    monkeypatch.setattr(server.vectors, "sync", flaky_sync)

    with pytest.raises(RuntimeError, match="embedder exploded"):
        server.ctx_hybrid_search(queries=["deploy"])
    assert str(tool_env) not in server._synced_at  # claim reset, not false-fresh

    out = server.ctx_hybrid_search(queries=["deploy"])  # retries the sync
    assert len(calls) == 2
    assert "alpha deploy guide" in out


def test_embedder_constructed_once_across_threads(monkeypatch: pytest.MonkeyPatch):
    constructions: list[int] = []

    class SlowStubEmbedder:
        def __init__(self) -> None:
            constructions.append(1)
            time.sleep(0.02)  # widen the cold-start race window

    monkeypatch.setattr(server, "Embedder", SlowStubEmbedder)
    monkeypatch.setattr(server, "_embedder", None)
    got: list = []

    def construct():
        got.append(server._get_embedder())

    threads = [threading.Thread(target=construct) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(constructions) == 1
    assert all(e is got[0] for e in got)


def test_tool_rejects_wrong_query_count(tool_env: Path):
    for queries in ([], ["a"] * 4, ["a"] * 5):
        with pytest.raises(ValueError, match="1-3"):
            server.ctx_hybrid_search(queries=queries)


def test_tool_limit_clamped_to_range(
    tool_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = make_db(tmp_path / "big.db")
    con = sqlite3.connect(db)
    con.executemany(
        "INSERT INTO chunks (title, content, source_id, content_type)"
        " VALUES (?, ?, 1, 'session')",
        [(f"deploy extra {i}", f"deploy filler content {i}") for i in range(12)],
    )
    con.commit()
    con.close()
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))
    server._synced_at.clear()
    server._sync_interval.clear()

    def block_count(resp: str) -> int:
        return sum(1 for line in resp.splitlines() if line.startswith("## "))

    assert block_count(server.ctx_hybrid_search(queries=["deploy"], limit=99)) == 10
    assert block_count(server.ctx_hybrid_search(queries=["deploy"], limit=-3)) == 1

def test_resync_boundary_age_equal_interval_is_fresh(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    # age == interval is still fresh (<= in _claim_sync): exactly at the TTL
    # there is no re-sync; one tick later there is.
    clock = FakeClock()
    monkeypatch.setattr(server, "time", clock)
    sync_calls = _counting_sync(monkeypatch)
    key = str(tool_env)

    server.ctx_hybrid_search(queries=["deploy"])
    with server._state_lock:  # pin standard interval (cold sync was hashless)
        server._sync_interval[key] = server.RESYNC_INTERVAL_S

    clock.now += server.RESYNC_INTERVAL_S  # exactly at the boundary
    server.ctx_hybrid_search(queries=["deploy"])
    assert len(sync_calls) == 1  # fresh — claim holds

    clock.now += 1  # strictly past it
    server.ctx_hybrid_search(queries=["deploy"])
    assert len(sync_calls) == 2


def test_resync_with_zero_embedded_has_no_progress_line(
    tool_env: Path, monkeypatch: pytest.MonkeyPatch
):
    # the progress line is a signal about REAL work: a TTL re-sync that
    # embeds nothing must stay silent (>= 0 would prepend it everywhere).
    clock = FakeClock()
    monkeypatch.setattr(server, "time", clock)
    server.ctx_hybrid_search(queries=["deploy"])  # cold sync: embedded 3
    clock.now += server.HASHLESS_RESYNC_INTERVAL_S + 1
    out = server.ctx_hybrid_search(queries=["deploy"])  # re-sync, embedded=0
    assert "(embedded" not in out
    assert "alpha deploy guide" in out  # results still served
