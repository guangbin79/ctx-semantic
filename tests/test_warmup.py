"""Tests for ctx_semantic.warmup (T7 warmup CLI).

vectors.sync and Embedder are stubbed (no model load, no store writes); the
DB open + BoundAdapter wiring run against a real context-mode-shaped fixture
DB. Covers: call args, printed lines, --db relative .resolve(), --all scan,
--project chain, missing-DB skip.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ctx_semantic import projhash, warmup
from ctx_semantic.binding import BoundAdapter

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

FTS_DDL_PORTER = """
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

FTS_DDL_TRIGRAM = FTS_DDL_PORTER.replace("chunks USING", "chunks_trigram USING")


def make_db(path: Path) -> Path:
    """Context-mode-shaped DB that survives open_db's schema assertion."""
    con = sqlite3.connect(path)
    try:
        con.execute(SOURCES_DDL)
        con.execute(FTS_DDL_PORTER)
        con.execute(FTS_DDL_TRIGRAM)
        con.execute(
            "INSERT INTO sources (id, label, content_hash) VALUES (1, 'src', 'h1')"
        )
        con.execute(
            "INSERT INTO chunks (title, content, source_id, content_type)"
            " VALUES ('t', 'c', 1, 'note')"
        )
        con.commit()
    finally:
        con.close()
    return path


class StubEmbedder:
    """Construction-counting stand-in; embed_batch is never called (sync stubbed)."""

    built: int = 0

    def __init__(self) -> None:
        StubEmbedder.built += 1
        self.device = "stub-device"

    def embed_batch(self, texts):
        raise AssertionError("stubbed sync must not embed")


@pytest.fixture
def sync_stub(monkeypatch: pytest.MonkeyPatch):
    """Patch vectors.sync; returns (calls, counts-to-report)."""
    calls: list[tuple[object, object, object]] = []
    counts = {"embedded": 2, "removed": 1, "total": 7}

    def fake_sync(adapter, embed_fn, db, **kwargs):
        calls.append((adapter, embed_fn, db))
        return dict(counts)

    monkeypatch.setattr(warmup.vectors, "sync", fake_sync)
    monkeypatch.setattr(warmup, "Embedder", StubEmbedder)
    StubEmbedder.built = 0
    return calls, counts


def test_bound_adapter_serves_source_adapter_protocol(tmp_path: Path):
    # warmup hands ONE BoundAdapter to vectors.sync: pin its SourceAdapter
    # surface over the real read-only connection.
    from ctx_semantic import dbadapter

    db = make_db(tmp_path / "src.db")
    con = dbadapter.open_db(db)
    try:
        adapter = BoundAdapter(con, db)
        assert adapter.live_rowids() == {1}
        assert adapter.source_hashes() == {1: "h1"}
        assert adapter.snapshot_chunks()[0][:4] == (1, "t", "c", 1)
    finally:
        con.close()


# --- warm_one ---------------------------------------------------------------


def test_warm_one_missing_db_skips_without_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    def explode(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("must not open a missing DB")

    monkeypatch.setattr(warmup.dbadapter, "open_db", explode)
    counts = warmup.warm_one(tmp_path / "missing.db", StubEmbedder())
    assert counts == {"embedded": 0, "removed": 0, "total": 0}
    assert "skip (no such db)" in capsys.readouterr().out


def test_warm_one_wires_bound_adapter_and_prints_counts(
    tmp_path: Path, sync_stub, capsys
):
    calls, counts = sync_stub
    db = make_db(tmp_path / "src.db")
    emb = StubEmbedder()

    got = warmup.warm_one(db, emb)

    assert got == counts
    assert len(calls) == 1
    adapter, embed_fn, db_arg = calls[0]
    assert isinstance(adapter, BoundAdapter)
    assert adapter.db_path == str(db)
    assert adapter.con is not None
    assert embed_fn == emb.embed_batch
    assert db_arg == db
    out = capsys.readouterr().out
    assert "src.db: embedded=2 removed=1 total=7" in out
    assert "elapsed=" in out
    assert "model=" not in out  # model line belongs to main(), not warm_one


# --- main: flag matrix --------------------------------------------------------


def test_main_db_relative_path_resolved_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_stub, capsys
):
    calls, _ = sync_stub
    monkeypatch.chdir(tmp_path)
    make_db(Path("rel.db"))

    assert warmup.main(["--db", "rel.db"]) == 0

    assert len(calls) == 1
    resolved = (tmp_path / "rel.db").resolve()
    assert calls[0][2] == resolved
    assert calls[0][2].is_absolute()
    out = capsys.readouterr().out
    assert "model=jinaai/jina-embeddings-v2-base-zh device=stub-device" in out
    assert StubEmbedder.built == 1


def test_main_db_missing_file_still_exits_zero(
    tmp_path: Path, sync_stub, capsys
):
    calls, _ = sync_stub
    assert warmup.main(["--db", str(tmp_path / "nope.db")]) == 0
    assert calls == []  # warm_one skipped; embedder still constructed for report
    out = capsys.readouterr().out
    assert "skip (no such db)" in out
    assert "device=stub-device" in out


def test_main_all_scans_content_dir_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_stub
):
    calls, _ = sync_stub
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(projhash, "CONTENT_DIR", content)
    make_db(content / "b.db")
    make_db(content / "a.db")

    assert warmup.main(["--all"]) == 0
    assert [Path(c[2]).name for c in calls] == ["a.db", "b.db"]


def test_main_all_empty_dir_reports_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_stub, capsys
):
    calls, _ = sync_stub
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(projhash, "CONTENT_DIR", content)

    assert warmup.main(["--all"]) == 0
    assert calls == []
    assert f"no content DBs under {content}" in capsys.readouterr().out


def test_main_default_uses_env_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_stub
):
    calls, _ = sync_stub
    db = make_db(tmp_path / "env.db")
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))

    assert warmup.main([]) == 0
    assert calls[0][2] == db


def test_main_project_resolves_hashed_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sync_stub
):
    calls, _ = sync_stub
    content = tmp_path / "content"
    content.mkdir()
    monkeypatch.setattr(projhash, "CONTENT_DIR", content)
    monkeypatch.delenv("CTX_SEMANTIC_DB", raising=False)
    monkeypatch.delenv("OPENCODE_PROJECT_DIR", raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    make_db(projhash.db_path(projhash.resolve(proj)))

    assert warmup.main(["--project", str(proj)]) == 0
    assert calls[0][2] == projhash.db_path(projhash.resolve(proj))


def test_main_nonexistent_project_dir_is_parser_error(tmp_path: Path):
    with pytest.raises(SystemExit) as exc:
        warmup.main(["--project", str(tmp_path / "never")])
    assert exc.value.code == 2
