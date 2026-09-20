"""Tests for ctx_semantic.hybrid — synthetic stubs, real RRF math, exact output.

The adapter is stubbed (the real T7 binding over dbadapter does not exist
yet); the vector leg runs the REAL vectors.search over an in-memory store so
fusion is hand-computable: stored vectors are unit basis vectors, queries are
basis vectors too, so cosines are exactly 0.0 / 1.0. Nothing here loads the
real embedding model.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from ctx_semantic import dbadapter, vectors
from ctx_semantic.hybrid import NO_RESULTS, _excerpt, rrf, search

DIM = 4
DB_PATH = "/proj.db"

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


def make_store(rows: dict[int, list[float]]) -> sqlite3.Connection:
    """In-memory vector store; unit-basis vectors keep cosines exact."""
    con = vectors.connect(":memory:")
    for rid, vec in rows.items():
        con.execute(
            "INSERT INTO embeddings(db_path, chunk_rowid, model, content_hash,"
            " dim, vec, embedded_at) VALUES(?,?,?,?,?,?,?)",
            (DB_PATH, rid, vectors.DEFAULT_MODEL, "", DIM,
             np.asarray(vec, dtype=np.float32).tobytes(), 0.0),
        )
    return con


class StubAdapter:
    """Canned adapter: records filter reach, serves preset results."""

    def __init__(
        self,
        bm25: dict[str, list[tuple[int, float]]] | None = None,
        allowed: set[int] | None = None,
        chunks: dict[int, tuple[str, str, str]] | None = None,
    ):
        self.db_path: str = DB_PATH
        self._bm25 = bm25 or {}
        self._allowed = allowed
        self._chunks = chunks or {}
        self.bm25_calls: list[tuple[str, int, str | None, str | None]] = []
        self.rowids_calls: list[tuple[str | None, str | None]] = []

    def bm25_search(self, query, limit, source=None, content_type=None):
        self.bm25_calls.append((query, limit, source, content_type))
        return self._bm25.get(query, [])[:limit]

    def filtered_rowids(self, source=None, content_type=None):
        self.rowids_calls.append((source, content_type))
        if self._allowed is not None:
            return set(self._allowed)
        ids = {rid for hits in self._bm25.values() for rid, _ in hits}
        return ids | set(self._chunks)

    def get_many(self, rowids):
        return {
            rid: (t, c, label, "note", 1.0)
            for rid, (t, c, label) in self._chunks.items()
            if rid in set(rowids)
        }


class RealAdapter:
    """Delegates legs to the real dbadapter functions over a fixture FTS DB."""

    def __init__(self, con: sqlite3.Connection, path: Path):
        self.con = con
        self.db_path = str(path)
        self.bm25_queries: list[str] = []

    def bm25_search(self, query, limit, source=None, content_type=None):
        self.bm25_queries.append(query)  # raw pass-through, escaping upstream
        return dbadapter.bm25_search(self.con, query, limit)

    def filtered_rowids(self, source=None, content_type=None):
        return {r[0] for r in self.con.execute("SELECT rowid FROM chunks")}

    def get_many(self, rowids):
        return dbadapter.get_many(self.con, rowids)


class StubEmbedder:
    """Preset query -> vector table (basis vectors keep cosines exact)."""

    def __init__(self, table: dict[str, list[float]]):
        self.table = table
        self.queries: list[str] = []

    def embed_query(self, text: str) -> np.ndarray:
        self.queries.append(text)
        return np.asarray(self.table[text], dtype=np.float32)


E1 = [1.0, 0.0, 0.0, 0.0]
E2 = [0.0, 1.0, 0.0, 0.0]
E3 = [0.0, 0.0, 1.0, 0.0]

CHUNKS = {
    1: ("部署手册", "run deploy.sh to ship the service", "note"),
    2: ("缓存策略", "cache TTL is 60 seconds for hot keys", "decision"),
    3: ("错误处理", "retry three times then surface the error", "error-resolution"),
}
BASIS_STORE = {1: E1, 2: E2, 3: E3}


# --- rrf: pure fusion math -------------------------------------------------


def test_rrf_hand_computed_fusion():
    bm25 = [(10, 0.1), (20, 0.5), (30, 0.9)]  # best first: lower bm25 wins
    vec = [(20, 0.99), (40, 0.90), (10, 0.80)]  # best first: higher cosine
    got = rrf(bm25, vec, k=60)
    # Hand-computed (k=60): 20 -> 1/62+1/61, 10 -> 1/61+1/63, 40 -> 1/62,
    # 30 -> 1/63. Fused order differs from either single leg
    # (bm25-only: 10,20,30; vec-only: 20,40,10).
    assert got == [
        (20, pytest.approx(1 / 62 + 1 / 61)),
        (10, pytest.approx(1 / 61 + 1 / 63)),
        (40, pytest.approx(1 / 62)),
        (30, pytest.approx(1 / 63)),
    ]


def test_rrf_ignores_raw_scores_uses_positions_only():
    # Same rank positions, wildly different raw scores -> identical fusion.
    a = rrf([(1, 0.0), (2, 0.0)], [(3, 0.0)], k=60)
    b = rrf([(1, 99.0), (2, -5.0)], [(3, 42.0)], k=60)
    assert a == b


def test_rrf_bm25_empty_keeps_vector_ranking():
    vec = [(7, 0.9), (3, 0.5)]
    assert rrf([], vec, k=60) == [
        (7, pytest.approx(1 / 61)),
        (3, pytest.approx(1 / 62)),
    ]


# --- search: leg composition -----------------------------------------------


def test_search_bm25_empty_preserves_vector_ranking():
    store = make_store(BASIS_STORE)
    adapter = StubAdapter(chunks=CHUNKS)  # no bm25 hits: query not in table
    emb = StubEmbedder({"err": E3})
    out = search(adapter, store, emb, ["err"], limit=3)
    # vec: 3 (cos 1.0), then 1 and 2 (cos 0.0, stable rowid order)
    titles = [line[3:] for line in out.splitlines() if line.startswith("## ")]
    assert titles == ["错误处理", "部署手册", "缓存策略"]


def test_search_both_empty_friendly_no_results():
    store = make_store({})
    adapter = StubAdapter(chunks=CHUNKS)
    emb = StubEmbedder({"q": E1})
    assert search(adapter, store, emb, ["q"], limit=3) == NO_RESULTS


def test_search_filters_reach_both_legs():
    store = make_store(BASIS_STORE)
    adapter = StubAdapter(allowed={2}, chunks=CHUNKS)
    emb = StubEmbedder({"deploy": E1})
    # BM25 leg unfiltered by the stub, but records what it received; the
    # vector leg narrows candidates to {2}, so rid 1 (perfect cosine) must
    # NOT surface even though it would win an unfiltered vector pass.
    out = search(
        adapter, store, emb, ["deploy"],
        source="src-one", content_type="code", limit=1,
    )
    assert adapter.bm25_calls == [("deploy", 1, "src-one", "code")]
    assert adapter.rowids_calls == [("src-one", "code")]
    assert emb.queries == ["deploy"]
    assert "部署手册" not in out and "缓存策略" in out


# --- search: output format (golden, exact full string) ---------------------


def test_search_golden_output_exact():
    store = make_store(BASIS_STORE)
    adapter = StubAdapter(
        bm25={"deploy": [(1, 0.5)]},
        chunks=CHUNKS,
    )
    emb = StubEmbedder({"deploy": E1})
    out = search(adapter, store, emb, ["deploy"], limit=2)
    # fused: 1 (1/61+1/61), 2 (1/62), 3 (1/63) -> limit 2 keeps 1, 2
    assert out == (
        "### 查询 1：deploy\n"
        "\n"
        "## 部署手册\n"
        "note\n"
        "run deploy.sh to ship the service\n"
        "\n"
        "## 缓存策略\n"
        "decision\n"
        "cache TTL is 60 seconds for hot keys"
    )


def test_search_multi_query_sections_and_cross_query_dedup():
    store = make_store(BASIS_STORE)
    adapter = StubAdapter(
        bm25={
            "deploy": [(1, 0.5)],
            "缓存": [(2, 0.3), (1, 0.9)],
        },
        chunks=CHUNKS,
    )
    emb = StubEmbedder({"deploy": E1, "缓存": E2})
    out = search(adapter, store, emb, ["deploy", "缓存"], limit=2)
    # fused(q1): 1 -> 2/61, 2 -> 1/62, 3 -> 1/63; fused(q2): 2 -> 2/61,
    # 1 -> 2/62, 3 -> 1/63. Chunk 1 peaks in q1, chunk 2 in q2 -> each
    # appears exactly once, in its best-scoring section.
    assert out == (
        "### 查询 1：deploy\n"
        "\n"
        "## 部署手册\n"
        "note\n"
        "run deploy.sh to ship the service\n"
        "\n"
        "### 查询 2：缓存\n"
        "\n"
        "## 缓存策略\n"
        "decision\n"
        "cache TTL is 60 seconds for hot keys"
    )
    assert out.count("部署手册") == 1 and out.count("缓存策略") == 1


def test_search_dedup_tie_keeps_first_query():
    # Chunk 5 tops both queries with the identical fused score 2/61.
    store = make_store({5: [0.5, 0.5, 0.5, 0.5]})
    chunk = {5: ("t5", "shared content", "note")}
    adapter = StubAdapter(
        bm25={"q1": [(5, 0.4)], "q2": [(5, 0.4)]}, chunks=chunk
    )
    emb = StubEmbedder({"q1": [0.5, 0.5, 0.5, 0.5], "q2": [0.5, 0.5, 0.5, 0.5]})
    out = search(adapter, store, emb, ["q1", "q2"], limit=1)
    assert out == (
        "### 查询 1：q1\n"
        "\n"
        "## t5\n"
        "note\n"
        "shared content"
    )


def test_search_synced_info_prepended_once():
    store = make_store(BASIS_STORE)
    adapter = StubAdapter(bm25={"deploy": [(1, 0.5)]}, chunks=CHUNKS)
    emb = StubEmbedder({"deploy": E1})
    out = search(
        adapter, store, emb, ["deploy"], limit=1,
        synced_info="(embedded 3 new chunks in 1.3s)",
    )
    assert out.startswith("(embedded 3 new chunks in 1.3s)\n")
    assert out.count("(embedded") == 1


# --- excerpt windows --------------------------------------------------------


def test_excerpt_window_around_first_hit():
    content = "a" * 300 + "NEEDLE" + "b" * 300
    # hit at 300; start = 300-60 = 240; window = content[240:480]
    assert _excerpt(content, "miss needle", width=240) == (
        "a" * 60 + "NEEDLE" + "b" * 174
    )


def test_excerpt_head_fallback_and_case_insensitive():
    assert _excerpt("short body", "nothing matches here", width=240) == "short body"
    assert _excerpt("x" * 300, "UPPER", width=240) == "x" * 240
    assert _excerpt("has Needle inside", "needle", width=240) == "has Needle inside"


# --- FTS5 syntax adversarial (real dbadapter legs) --------------------------


def test_fts5_syntax_query_no_crash(tmp_path):
    path = tmp_path / "src.db"
    con = sqlite3.connect(path)
    con.execute(SOURCES_DDL)
    con.execute(FTS_DDL)
    con.execute("INSERT INTO sources (id, label) VALUES (1, 'src')")
    con.execute(
        "INSERT INTO chunks (title, content, source_id, content_type)"
        " VALUES ('deploy doc', 'deploy the service now', 1, 'note')"
    )
    con.commit()
    adapter = RealAdapter(con, path)
    store = make_store({1: E1})
    emb = StubEmbedder({
        'weird " AND (x*': E1,
        "deploy": E1,
    })
    nasty = 'weird " AND (x*'
    out = search(adapter, store, emb, [nasty, "deploy"], limit=2)
    # escaping is dbadapter's job: the raw string reached the BM25 leg intact
    assert adapter.bm25_queries[0] == nasty
    assert isinstance(out, str) and "deploy doc" in out
