"""Tests for ctx_semantic.harness gate arithmetic (T8 acceptance gate).

evaluate() is pure: no corpus, no DB, no embedder — every threshold rule is
table-driven here so the gate can be trusted before the live-DB harness runs.
Row shape is (query, lang, title, bm25_rank, hybrid_rank); rank None = miss
(rank infinity), a hit is rank <= TOP_K (=5).
"""

from __future__ import annotations

import pytest

from ctx_semantic import harness
from ctx_semantic.harness import TOP_K, Gates, _rank, evaluate


def r(b: int | None, h: int | None) -> tuple[str, str, str, int | None, int | None]:
    return ("query", "en", "gold title", b, h)


# --- _rank: gate input -------------------------------------------------------


@pytest.mark.parametrize(
    ("rowid", "ranked", "expected"),
    [
        (10, [(10, 0.1), (20, 0.2)], 1),  # 1-based best-first position
        (20, [(10, 0.1), (20, 0.2)], 2),
        (99, [(10, 0.1), (20, 0.2)], None),  # absent -> miss
        (10, [], None),  # empty leg
    ],
)
def test_rank_position_or_none(rowid, ranked, expected):
    assert _rank(rowid, ranked) == expected


# --- evaluate: table over every gate rule ------------------------------------


CASES = [
    # name, rows, expected Gates(n, bm25, hyb, not_worse, binding, rescues, t1, t2, t3)
    (
        "normal corpus, binding met, one rescue",
        [r(1, 1), r(2, 2), r(None, 3), r(None, None)],
        Gates(4, 2, 3, 4, 2, 1, True, True, True),
    ),
    (
        "vacuous t2: bm25 hits nothing, t1/t3 still gate hybrid",
        [r(None, 1), r(None, 4), r(None, None)],
        Gates(3, 0, 2, 3, 0, 2, True, True, True),
    ),
    (
        "all-miss corpus: t1/t2 vacuous, t3 fails",
        [r(None, None), r(None, None)],
        Gates(2, 0, 0, 2, 0, 0, True, True, False),
    ),
    (
        "both-miss inf<=inf is not-worse but never satisfies the binding gate",
        [r(1, None), r(2, None), r(None, None)],
        Gates(3, 2, 0, 1, 0, 0, False, False, False),
    ),
    (
        "binding boundary ceil(0.8*5)=4 exactly met",
        [r(1, 1), r(2, 2), r(3, 3), r(4, 4), r(5, None), r(None, 1)],
        Gates(6, 5, 5, 5, 4, 1, True, True, True),
    ),
    (
        "binding boundary ceil(0.8*5)=4 missed by one",
        [r(1, 1), r(2, 2), r(3, 3), r(4, None), r(5, None), r(None, 1)],
        Gates(6, 5, 4, 4, 3, 1, False, False, True),
    ),
    (
        "ceil(0.8*2)=2: BOTH of two winnable rows must be won (hybrid rank worse)",
        [r(1, 1), r(2, 3), r(None, None)],
        Gates(3, 2, 2, 2, 1, 0, True, False, False),
    ),
    (
        "ceil(0.8*2)=2 met: hybrid ties both",
        [r(1, 1), r(2, 2), r(None, None)],
        Gates(3, 2, 2, 3, 2, 0, True, True, False),
    ),
    (
        "single winnable row won: max(1, ceil(0.8*1))=1",
        [r(1, 1), r(None, None)],
        Gates(2, 1, 1, 2, 1, 0, True, True, False),
    ),
    (
        "single winnable row lost: rescue cannot heal the binding gate",
        [r(1, None), r(None, 2)],
        Gates(2, 1, 1, 1, 0, 1, True, False, True),
    ),
    (
        "rescue beyond TOP_K (rank 6) is neither hit nor rescue",
        [r(None, 6)],
        Gates(1, 0, 0, 1, 0, 0, True, True, False),
    ),
    (
        "rescue at exactly TOP_K (rank 5) counts",
        [r(None, 5)],
        Gates(1, 0, 1, 1, 0, 1, True, True, True),
    ),
    (
        "hit window boundary: bm25 rank 5 hits, rank 6 does not",
        [r(5, None), r(6, None)],
        Gates(2, 1, 0, 0, 0, 0, False, False, False),
    ),
    (
        "hybrid worse than bm25 on a hit row (rank 4 -> 7) loses not-worse",
        # hybrid rank 7 > TOP_K: not a hit, not not-worse, binding lost
        [r(4, 7), r(None, 1)],
        Gates(2, 1, 1, 1, 0, 1, True, False, True),
    ),
    (
        "t1 equality passes: hybrid hits exactly as many as bm25",
        [r(1, 1)],
        Gates(1, 1, 1, 1, 1, 0, True, True, False),
    ),
]


@pytest.mark.parametrize(("name", "rows", "expected"), CASES, ids=[c[0] for c in CASES])
def test_evaluate_table(name, rows, expected):
    assert evaluate(rows) == expected


def test_evaluate_empty_corpus():
    # no rows: t1/t2 hold vacuously; t3 (>=1 rescue) is unsatisfiable
    assert evaluate([]) == Gates(0, 0, 0, 0, 0, 0, True, True, False)


@pytest.mark.parametrize(
    ("t1", "t2", "t3", "ok"), [(True, True, True, True)] + [
        (a, b, c, False)
        for a in (True, False)
        for b in (True, False)
        for c in (True, False)
        if (a, b, c) != (True, True, True)
    ],
)
def test_ok_requires_all_three_thresholds(t1, t2, t3, ok):
    g = Gates(1, 1, 1, 1, 1, 1, t1, t2, t3)
    assert g.ok is ok


def test_top_k_is_five():
    # the table above is calibrated to the server tool's limit=5
    assert TOP_K == 5


# --- main(): full wiring on stubs — no corpus, real rrf + report rendering ----


class FakeCon:
    """sqlite3.Connection stand-in: serves the chunk count, close() is a no-op."""

    def __init__(self, chunk_total: int = 659):
        self.chunk_total = chunk_total
        self.closed = False

    def execute(self, sql, *args):
        class _R:
            def __init__(self, value):
                self.value = value

            def fetchone(self):
                return (self.value,)

        return _R(self.chunk_total)

    def close(self):
        self.closed = True


class HarnessAdapter:
    """BoundAdapter stand-in: golds table + per-lang BM25 behavior."""

    def __init__(self, golds, bm25_by_query, vec_by_query):
        self.golds = golds
        self.bm25_by_query = bm25_by_query
        self.vec_by_query = vec_by_query
        self.bm25_calls: list[tuple[str, int]] = []

    def get_many(self, rowids):
        return {rid: self.golds[rid] for rid in rowids if rid in self.golds}

    def bm25_search(self, query, limit):
        self.bm25_calls.append((query, limit))
        return self.bm25_by_query.get(query, [])[:limit]


class HarnessEmbedder:
    device = "stub-cpu"

    def __init__(self):
        self.batch_texts: list[list[str]] = []

    def embed_batch(self, texts):
        self.batch_texts.append(list(texts))
        return []

    def embed_query(self, text):  # query text rides through to the search stub
        return text


def _wire_main(monkeypatch, tmp_path, adapter, sync_counts):
    """Patch every live-DB edge main() touches; return (report, stdout extras)."""
    con = FakeCon()
    report = tmp_path / "report.md"

    def fake_sync(adapter_arg, embed_fn, db):
        return dict(sync_counts)

    def fake_search(vcon, adapter_arg, db, qvec, k):
        assert k == harness.VEC_K  # must stay the server's vector-leg width
        return adapter.vec_by_query.get(qvec, [])

    monkeypatch.setattr(harness.dbadapter, "open_db", lambda db: con)
    monkeypatch.setattr(harness, "BoundAdapter", lambda con, db: adapter)
    monkeypatch.setattr(harness, "Embedder", HarnessEmbedder)
    monkeypatch.setattr(harness.vectors, "sync", fake_sync)
    monkeypatch.setattr(harness.vectors, "connect", lambda: FakeCon())
    monkeypatch.setattr(harness.vectors, "search", fake_search)
    monkeypatch.setattr(harness, "REPORT_PATH", report)
    return con, report


def _corpus_scenarios(cases):
    """golds table + legs: en cases hit BM25 at rank 1, every case hits vectors."""
    golds = {rid: (title, "content", "label", "note", None) for _q, _l, rid, title in cases}
    bm25 = {q: [(rid, 0.1)] for q, lang, rid, _t in cases if lang == "en"}
    vec = {q: [(rid, 0.9)] for q, _lang, rid, _t in cases}
    return golds, bm25, vec


def test_main_all_pass_writes_report_and_exits_zero(
    monkeypatch, tmp_path, capsys
):
    golds, bm25, vec = _corpus_scenarios(harness.CASES)
    adapter = HarnessAdapter(golds, bm25, vec)
    con, report = _wire_main(
        monkeypatch, tmp_path, adapter, {"embedded": 3, "removed": 1, "total": 659}
    )

    code = harness.main(["--db", str(tmp_path / "corpus.db")])

    assert code == 0
    out = capsys.readouterr().out
    assert "thresholds: ALL PASS" in out
    assert out.count("[zh]") == 8 and out.count("[en]") == 4  # per-row prints
    assert len(adapter.bm25_calls) == len(harness.CASES)
    assert all(limit == harness.TOP_K for _q, limit in adapter.bm25_calls)
    assert con.closed
    text = report.read_text(encoding="utf-8")
    assert "sync before run: embedded=3 removed=1 total=659" in text
    assert "device=stub-cpu" in text
    assert text.count("**PASS**") == 3  # t1, t2, t3 lines
    assert "**FAIL**" not in text


def test_main_gold_drift_aborts_before_sync(monkeypatch, tmp_path, capsys):
    golds, bm25, vec = _corpus_scenarios(harness.CASES)
    drifted_rid = harness.CASES[0][2]
    golds[drifted_rid] = ("WRONG TITLE", "content", "label", "note", None)
    adapter = HarnessAdapter(golds, bm25, vec)
    _wire_main(monkeypatch, tmp_path, adapter, {"embedded": 0, "removed": 0, "total": 0})

    def explode(*a, **k):  # pragma: no cover - must not run
        raise AssertionError("gold drift must abort before any embedding")

    monkeypatch.setattr(harness.vectors, "sync", explode)
    code = harness.main(["--db", str(tmp_path / "corpus.db")])

    assert code == 2
    drift_out = capsys.readouterr().out
    assert f"gold drift at rowid={drifted_rid}" in drift_out
    assert "expected" in drift_out and "WRONG TITLE" in drift_out


def test_main_no_rescues_fails_t3_exits_one(monkeypatch, tmp_path, capsys):
    # BM25 hits every gold at rank 1 and the vector leg is empty: each fused
    # list inherits the bm25 hit (so t1/t2 hold), but no BM25-miss was ever
    # rescued — t3 fails and the harness reports FAIL honestly.
    golds, _bm25, _vec = _corpus_scenarios(harness.CASES)
    bm25 = {q: [(rid, 0.1)] for q, _l, rid, _t in harness.CASES}
    adapter = HarnessAdapter(golds, bm25, {})  # no vector hits anywhere
    _wire_main(monkeypatch, tmp_path, adapter, {"embedded": 0, "removed": 0, "total": 659})

    assert harness.main(["--db", str(tmp_path / "corpus.db")]) == 1
    out = capsys.readouterr().out
    assert "thresholds: FAIL" in out
    assert "bm25_hits=14 hybrid_hits=14 not_worse=14/14 rescues=0" in out
