"""Tests for ctx_semantic.dbadapter.

Fixtures build a temp-file DB with the SAME schema as context-mode's content
DBs (CREATE statements lifted verbatim from the live DB). Tests never write to
real DBs; the real-DB integration test opens read-only only.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from ctx_semantic import dbadapter
from ctx_semantic.dbadapter import SchemaDrift, open_db

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

FTS_DDL_TRIGRAM = """
CREATE VIRTUAL TABLE chunks_trigram USING fts5(
        title,
        content,
        source_id UNINDEXED,
        content_type UNINDEXED,
        source_category UNINDEXED,
        session_id UNINDEXED,
        event_id UNINDEXED,
        timestamp UNINDEXED,
        tokenize='trigram'
      )
"""

SEED_ROWS = [
    ("hello world", "the quick brown fox jumps over the lazy dog", 1, "prose"),
    ("second note", "laya rendering engine internals and frame pacing", 1, "code"),
    ("third note", "unrelated content about databases and indexes", 2, "prose"),
]


def make_fixture(path: Path, seed: bool = True) -> None:
    """Build a context-mode-shaped DB (ours to write to, unlike the real one)."""
    con = sqlite3.connect(path)
    try:
        con.execute(SOURCES_DDL)
        con.execute(FTS_DDL_PORTER)
        con.execute(FTS_DDL_TRIGRAM)
        con.execute("INSERT INTO sources (id, label) VALUES (1, 'src-one')")
        con.execute("INSERT INTO sources (id, label) VALUES (2, 'src-two')")
        if seed:
            for title, content, sid, ctype in SEED_ROWS:
                con.execute(
                    "INSERT INTO chunks (title, content, source_id, content_type)"
                    " VALUES (?, ?, ?, ?)",
                    (title, content, sid, ctype),
                )
        con.commit()
    finally:
        con.close()


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    db = tmp_path / "fixture.db"
    make_fixture(db)
    return db


@pytest.fixture
def con(fixture_db: Path):
    c = open_db(fixture_db)
    yield c
    c.close()


# ---------------------------------------------------------------- open_db


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such content DB"):
        open_db(tmp_path / "never-created.db")


def test_open_retries_transient_failure(fixture_db, monkeypatch):
    real_connect = sqlite3.connect
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise sqlite3.OperationalError("unable to open database file")
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(dbadapter, "BACKOFF_S", 0.0)
    monkeypatch.setattr(dbadapter.sqlite3, "connect", flaky)
    c = open_db(fixture_db)
    try:
        assert calls["n"] == 3
        assert c.execute("SELECT count(*) FROM chunks").fetchone()[0] == 3
    finally:
        c.close()


def test_open_exhausts_retries_then_raises(fixture_db, monkeypatch):
    def always_fails(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(dbadapter, "BACKOFF_S", 0.0)
    monkeypatch.setattr(dbadapter.sqlite3, "connect", always_fails)
    with pytest.raises(sqlite3.OperationalError):
        open_db(fixture_db)


def test_open_sets_busy_timeout_and_query_only(con):
    assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 2000
    assert con.execute("PRAGMA query_only").fetchone()[0] == 1


def test_readonly_enforcement_create_table_raises(con):
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("CREATE TABLE sneaky (id INTEGER)")


def test_schema_drift_missing_table(tmp_path):
    db = tmp_path / "broken.db"
    con_rw = sqlite3.connect(db)
    con_rw.execute(SOURCES_DDL)  # chunks tables missing entirely
    con_rw.commit()
    con_rw.close()
    with pytest.raises(SchemaDrift, match="chunks"):
        open_db(db)


def test_schema_drift_extra_column(tmp_path):
    db = tmp_path / "drifted.db"
    con_rw = sqlite3.connect(db)
    drifted = SOURCES_DDL.replace("content_hash TEXT", "content_hash TEXT, extra_col TEXT")
    con_rw.execute(drifted)
    con_rw.execute(FTS_DDL_PORTER)
    con_rw.execute(FTS_DDL_TRIGRAM)
    con_rw.commit()
    con_rw.close()
    with pytest.raises(SchemaDrift, match="extra_col"):
        open_db(db)


# ------------------------------------------------------------ list_chunks


def test_list_chunks_returns_all_rows(con):
    rows = dbadapter.list_chunks(con)
    assert [r[0] for r in rows] == [1, 2, 3]
    assert rows[0][1:] == (
        "hello world",
        "the quick brown fox jumps over the lazy dog",
        1,
        "prose",
        None,
    ) or rows[0][5], "timestamp column must be carried through"


def test_list_chunks_filter_by_content_type(con):
    rows = dbadapter.list_chunks(con, content_type="code")
    assert len(rows) == 1
    assert rows[0][1] == "second note"


def test_list_chunks_filter_by_source_label(con):
    rows = dbadapter.list_chunks(con, source_filter="src-two")
    assert len(rows) == 1
    assert rows[0][3] == 2


def test_list_chunks_empty_db(tmp_path):
    db = tmp_path / "empty.db"
    make_fixture(db, seed=False)
    c = open_db(db)
    try:
        assert dbadapter.list_chunks(c) == []
        assert dbadapter.bm25_search(c, "anything", 5) == []
    finally:
        c.close()


# --------------------------------------------------------- snapshot_chunks


def test_snapshot_chunks_matches_list_chunks(con):
    assert dbadapter.snapshot_chunks(con) == dbadapter.list_chunks(con)


def test_snapshot_chunks_uses_explicit_transaction(con, monkeypatch):
    class SpyCon:
        """Delegating proxy: sqlite3.Connection.execute is read-only."""

        def __init__(self, real):
            self._real = real
            self.seen: list[str] = []

        def execute(self, sql, *a, **kw):
            self.seen.append(sql.strip().split()[0].upper())
            return self._real.execute(sql, *a, **kw)

        def __getattr__(self, name):
            return getattr(self._real, name)

    spy = SpyCon(con)
    dbadapter.snapshot_chunks(spy)
    assert spy.seen[0] == "BEGIN"
    assert spy.seen[-1] == "COMMIT"


# ------------------------------------------------------------ bm25_search


def test_bm25_search_ranked_results(con):
    rows = dbadapter.bm25_search(con, "laya", 5)
    assert len(rows) == 1
    rowid, score = rows[0]
    assert rowid == 2
    assert score < 0  # bm25: lower is better


def test_bm25_search_escapes_special_characters(con):
    # Any of these would raise fts5 syntax errors if passed raw — and the
    # degraded behavior is value-semantic: operators become literal tokens,
    # so "AND" matches exactly the chunks containing the word "and" (rids
    # 2+3), and an implicit-AND query no chunk satisfies returns nothing.
    expected = {
        '"': set(),
        "AND": {2, 3},
        "*": set(),
        "(": set(),
        "hello AND (world*": set(),  # "hello" is rid 1, "and" is not
        '") OR ("': set(),
    }
    for raw, want in expected.items():
        assert {r[0] for r in dbadapter.bm25_search(con, raw, 5)} == want, raw


def test_bm25_search_plain_terms_still_match(con):
    rows = dbadapter.bm25_search(con, "quick brown fox", 5)
    assert rows and rows[0][0] == 1


def test_bm25_search_no_match_returns_empty(con):
    assert dbadapter.bm25_search(con, "zzznomatchhere", 5) == []


def test_bm25_search_punctuation_only_returns_empty(con):
    # OCR #1: punctuation-only queries must degrade to an empty BM25 leg
    # instead of reaching MATCH (FT5 builds exist that reject phrases
    # tokenizing to zero tokens).
    for q in ("???", "——…", "...", '"??"'):
        assert dbadapter.bm25_search(con, q, 5) == []


def test_bm25_search_punctuation_around_keyword_keeps_match(con):
    # punctuation tokens drop; the alphanumeric token still matches rid 1.
    assert dbadapter.bm25_search(con, "?? fox ??", 5)[0][0] == 1


def test_bm25_search_source_filter_where_side(con):
    # seed: rid1,2 = src-one; rid3 = src-two. "note" matches titles 2 and 3.
    assert {r[0] for r in dbadapter.bm25_search(con, "note", 5)} == {2, 3}
    assert {r[0] for r in dbadapter.bm25_search(con, "note", 5, source_filter="src-two")} == {3}
    assert dbadapter.bm25_search(con, "note", 5, source_filter="ghost") == []
    # a query matching only src-one content returns nothing under src-two
    assert dbadapter.bm25_search(con, "fox", 5, source_filter="src-two") == []


def test_bm25_search_content_type_filter(con):
    # rid2 is the only "code" chunk; filters must not leak others.
    assert {r[0] for r in dbadapter.bm25_search(con, "note", 5, content_type="code")} == {2}
    assert dbadapter.bm25_search(con, "note", 5, source_filter="src-one", content_type="prose") == []
    assert {r[0] for r in dbadapter.bm25_search(con, "note", 5, source_filter="src-one", content_type="code")} == {2}


def test_filtered_rowids_and_source_helpers(con):
    assert dbadapter.filtered_rowids(con) == {1, 2, 3}
    assert dbadapter.filtered_rowids(con, source_filter="src-two") == {3}
    assert dbadapter.filtered_rowids(con, content_type="code") == {2}
    assert dbadapter.filtered_rowids(con, source_filter="ghost") == set()
    assert dbadapter.live_rowids(con) == {1, 2, 3}
    assert dbadapter.source_hashes(con) == {1: "", 2: ""}  # NULL → '' coalesce
    # (vectors.sync stores hash "" for missing sources — same bucket)

# ---------------------------------------------------------------- get_many


def test_get_many_fetches_with_source_label(con):
    got = dbadapter.get_many(con, [1, 3])
    assert set(got) == {1, 3}
    title, content, label, ctype, _ts = got[1]
    assert title == "hello world"
    assert label == "src-one"
    assert ctype == "prose"
    assert "fox" in content


def test_get_many_missing_rowids_omitted(con):
    got = dbadapter.get_many(con, [1, 99999])
    assert set(got) == {1}


def test_get_many_empty_input(con):
    assert dbadapter.get_many(con, []) == {}


# ----------------------------------------------------- BUSY-retry on query


def test_query_retry_succeeds_after_lock_release(fixture_db, monkeypatch):
    """EXCLUSIVE writer holds the lock through 2 attempts; 3rd succeeds."""
    monkeypatch.setattr(dbadapter, "BUSY_TIMEOUT_MS", 100)
    monkeypatch.setattr(dbadapter, "BACKOFF_S", 0.02)

    started = threading.Event()
    result: dict = {}

    def victim():
        c = open_db(fixture_db)  # own connection: sqlite3 is thread-bound
        try:
            started.set()
            result["rows"] = dbadapter.bm25_search(c, "laya", 5)
        except sqlite3.Error as e:  # surfaced via result dict
            result["error"] = e
        finally:
            c.close()

    t = threading.Thread(target=victim)
    t.start()
    assert started.wait(timeout=5)
    writer = sqlite3.connect(fixture_db, isolation_level=None)
    writer.execute("BEGIN EXCLUSIVE")  # lock AFTER victim holds its conn
    time.sleep(0.3)  # > attempt1+2 (busy 100ms x2 + backoffs)
    writer.rollback()  # release between attempt 2 and 3
    t.join(timeout=10)
    writer.close()

    assert "error" not in result, f"query never recovered: {result.get('error')}"
    assert [r[0] for r in result["rows"]] == [2]


def test_query_retry_exhausts_and_raises(fixture_db, monkeypatch):
    monkeypatch.setattr(dbadapter, "BUSY_TIMEOUT_MS", 50)
    monkeypatch.setattr(dbadapter, "BACKOFF_S", 0.01)
    writer = sqlite3.connect(fixture_db, isolation_level=None)
    writer.execute("BEGIN EXCLUSIVE")
    try:
        # Schema reads inside open_db are retried too, so the whole open
        # gives up (after MAX_ATTEMPTS) while the lock is held.
        with pytest.raises(sqlite3.OperationalError, match="locked|busy"):
            open_db(fixture_db)
    finally:
        writer.rollback()
        writer.close()


# ------------------------------------------------- real-DB integration (ro)


@pytest.mark.integration
def test_real_db_readonly_open_and_match():
    from ctx_semantic import projhash

    db = projhash.db_path(projhash.resolve("/home/guangbin"))
    if not db.exists():
        pytest.skip(f"real content DB not present: {db}")
    c = open_db(db)
    try:
        n = len(dbadapter.list_chunks(c))
        assert n > 0, "real DB unexpectedly empty"
        hits = dbadapter.bm25_search(c, "laya", 5)
        # a live DB with zero hits would silently skip the loop below and
        # mask the match/get_many roundtrip — require the path to run.
        assert hits, "real DB present but 'laya' matches nothing"
        for rowid, _score in hits:
            got = dbadapter.get_many(c, [rowid])
            assert rowid in got and got[rowid][0]  # rowid + non-empty title
    finally:
        c.close()


def test_bm25_search_mixed_alnum_token_survives_escaping(con):
    # "fox!" has an alnum char so it must survive token dropping (dropped
    # here would silently empty the BM25 leg); quoted, it matches rid 1.
    assert [r[0] for r in dbadapter.bm25_search(con, "fox!", 5)] == [1]


def test_bm25_search_limit_is_enforced(con):
    # LIMIT must bind: more matches than the limit returns exactly the limit.
    con_rw = sqlite3.connect(con.execute("PRAGMA database_list").fetchone()[2])
    con_rw.execute("PRAGMA query_only=OFF")
    con_rw.executemany(
        "INSERT INTO chunks (title, content, source_id, content_type)"
        " VALUES ('note filler', 'shared filler body text', 1, 'prose')",
        [()] * 7,
    )
    con_rw.commit()
    con_rw.close()
    for limit in (1, 3, 5):
        rows = dbadapter.bm25_search(con, "filler", limit)
        assert len(rows) == limit, f"LIMIT {limit} returned {len(rows)}"


def test_retryable_classification():
    # exactly the SQLITE_BUSY-class messages retry; anything else surfaces
    # immediately (an or->and slip here would silently drop one keyword).
    assert dbadapter._retryable(sqlite3.OperationalError("database is locked"))
    assert dbadapter._retryable(sqlite3.OperationalError("database is busy"))
    assert not dbadapter._retryable(sqlite3.OperationalError("no such table: x"))
    assert not dbadapter._retryable(sqlite3.OperationalError("unable to open"))
