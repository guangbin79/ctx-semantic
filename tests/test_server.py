"""Tests for ctx_semantic.server (T7): binding filters, tool wiring, cold start.

The real embedding model is NEVER loaded — the server's lazy Embedder global
is replaced by a deterministic stub (bag-of-chars 4-dim vectors, stable
across runs). The vector leg runs the REAL vectors.search over a per-test
store; the FTS leg runs the real dbadapter over a context-mode-shaped
fixture DB. Schema assertions hit the real MCPServer registration.
"""

from __future__ import annotations

import sqlite3
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
    server._synced_dbs.clear()
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
