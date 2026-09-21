"""One-shot BM25-vs-hybrid recall harness over a REAL project DB (T8).

Ground truth is programmatic: gold chunks were sampled read-only from a
live content DB, and each query below is a hand-written paraphrase of one
sampled chunk (Chinese paraphrases of English chunks carry zero lexical
overlap — the case pure BM25 cannot serve; English and mixed queries keep
the BM25 leg non-empty so RRF fusion is genuinely exercised). A hit means
the gold chunk's ROWID appears in the top-5 — chunk identity, never title
substrings.

Corpus: resampled 2026-09-21 from the OSMDataCompiler content DB
(53883986ad0936d4, 659 chunks) via ``--db`` — the original home-project
corpus was purged by the context-mode 1.0.169 upgrade.

Per query the harness runs the exact T7 server ranking path at limit=5:
dbadapter.bm25_search(query, 5) vs hybrid.rrf(bm25, vectors.search(qvec,
k=20)) — the same two legs and fusion hybrid.search applies before its
markdown rendering. Read-only on the context-mode DB; the sidecar store gets
one idempotent vectors.sync (steady state: embedded=0; the OSMDataCompiler
corpus is hash-less, so every sync re-embeds — see vectors.sync).

Writes the report to ~/.omo/evidence/ctx-semantic/recall-report.md and exits
0 only if all three thresholds hold: (1) hybrid top-5 hit-rate >= BM25's,
(2) binding not-worse: on >= max(1, ceil(0.8 * bm25_hits)) of the rows BM25
actually hits, hybrid also hits and is not worse (hybrid rank <= BM25 rank —
the winnable rows; a miss counts as rank infinity, so both-miss ∞≤∞ is
"not-worse" but can never fail and is reported, not gated; when
bm25_hits == 0 there are no winnable rows and gate 2 passes vacuously —
t1/t3 still gate hybrid quality, though zero BM25 hits on a corpus whose
CASES include en/mixed queries signals a broken precondition), (3) >=1 query
where BM25 misses top-5 and hybrid hits. A FAIL is reported honestly, not
tuned away.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import NamedTuple

from ctx_semantic import dbadapter, hybrid, projhash, vectors
from ctx_semantic.binding import BoundAdapter
from ctx_semantic.embedder import Embedder

PROJECT = Path.home()  # default corpus anchor (projhash resolve_db)

TOP_K = 5  # hit window — mirrors the server tool's limit
VEC_K = 20  # max(4*TOP_K, 20) — hybrid.search's vector-leg width
REPORT_PATH = Path.home() / ".omo" / "evidence" / "ctx-semantic" / "recall-report.md"

# (query, lang, gold_rowid, gold_title) — paraphrases authored 2026-09-21
# from a read-only sample of the OSMDataCompiler DB (53883986ad0936d4).
# gold_title guards rowid drift: the harness aborts (exit 2) if the rowid
# no longer holds the recorded chunk.
CASES: list[tuple[str, str, int, str]] = [
    # --- zh: pure-CJK paraphrases of English chunks (zero lexical overlap) ---
    ("一个目录里最多允许存放多少个瓦片文件", "zh", 867, "Lines 73-92"),
    ("整数值超出三十二位范围时字段值会改用哪个更宽的类型", "zh", 890,
     "sValue.eType = (nVal >= INT_MIN && nVal <= INT_MAX)"),
    ("解码线串几何时先跳过移动命令的变元再读坐标增量", "zh", 916,
     "poMultiPoint->addGeometryDirectly(poPoint);"),
    ("解析协议缓冲出错时打印调试日志然后直接返回失败", "zh", 982,
     'CPLDebug("MVT", "Protobuf error: line %d",'),
    ("往临时表写瓦片时把行列号序号和二进制块绑定到插入语句", "zh", 1102,
     "sqlite3_bind_int(m_hInsertStmt, 2, nTileX);"),
    ("瓦片压缩后仍超出大小上限就逐级把范围值减半", "zh", 1141,
     "size_t nSizeBefore = oTileBuffer.size();"),
    ("从临时数据库按层级和行列顺序读出全部瓦片组装输出", "zh", 1152,
     "std::map<CPLString, MVTLayerProperties> oMapLayerProps;"),
    ("驱动元数据里声明支持哪几种查询方言", "zh", 1212, '"Boolean Float32");'),
    # --- en: lexical overlap present — BM25 leg non-empty, RRF fused ---
    # (FTS5 MATCH is an implicit AND of all tokens, so these are authored
    #  with tokens that co-occur in the gold chunk.)
    ("ferry 1.1px step", "en", 809,
     "Map Styles > Strategies > Ferry 缩放线宽 (2026-05-27)"),
    ("tunnel transit railway", "en", 828,
     "Map Styles > Bug Experience > tunnel-transit 被道路遮挡 (2026-06-02)"),
    ("131 road classes day json", "en", 779,
     "Map Styles > Decisions > 完整道路层级 50→131 层 (2026-04-16)"),
    ("nproc CPU_CORES DEFAULT_THREADS", "en", 1311,
     "自动检测 CPU 核心数，默认使用一半核心防止系统过载"),
    # --- mixed zh+en: single Latin token keeps BM25 matching, vector carries semantics ---
    ("ogr2ogr 不修改任何输入", "mixed", 838,
     "total 83796 chars"),
    ("business anchor 方案", "mixed", 797,
     "Map Styles > Decisions > business anchor v2 实施 (2026-08-07)"),
]


def _rank(rowid: int, ranked: list[tuple[int, float]]) -> int | None:
    """1-based position of rowid in a best-first list, None when absent."""
    for pos, (rid, _score) in enumerate(ranked, start=1):
        if rid == rowid:
            return pos
    return None


def _fmt(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


class Gates(NamedTuple):
    """Threshold arithmetic result over harness rows (see evaluate)."""

    n: int
    bm25_hits: int
    hybrid_hits: int
    not_worse: int  # incl. both-miss ∞≤∞ rows (reported, never gated)
    binding_not_worse: int  # winnable rows only: BM25 hit AND hybrid not-worse
    rescues: int
    t1: bool
    t2: bool
    t3: bool

    @property
    def ok(self) -> bool:
        return self.t1 and self.t2 and self.t3


Row = tuple[str, str, str, int | None, int | None]


def evaluate(rows: Sequence[Row]) -> Gates:
    """Pure gate arithmetic over (query, lang, title, bm25_rank, hybrid_rank).

    A hit means rank is not None and rank <= TOP_K; a miss counts as rank
    infinity (both-miss ∞≤∞ is "not-worse" in the reported count but can
    never fail the binding gate). Gate 2 passes vacuously when bm25_hits
    is 0 — there are no winnable rows.
    """
    bm25_hits = sum(1 for r in rows if r[3] is not None and r[3] <= TOP_K)
    hybrid_hits = sum(1 for r in rows if r[4] is not None and r[4] <= TOP_K)
    not_worse = sum(
        1 for r in rows if r[3] is None or (r[4] is not None and r[4] <= r[3])
    )
    binding_not_worse = sum(
        1 for r in rows if r[3] is not None and r[4] is not None and r[4] <= r[3]
    )
    rescues = sum(
        1 for r in rows if r[3] is None and r[4] is not None and r[4] <= TOP_K)
    n = len(rows)
    return Gates(
        n=n,
        bm25_hits=bm25_hits,
        hybrid_hits=hybrid_hits,
        not_worse=not_worse,
        binding_not_worse=binding_not_worse,
        rescues=rescues,
        t1=hybrid_hits >= bm25_hits,
        t2=bm25_hits == 0 or binding_not_worse >= max(1, ceil(0.8 * bm25_hits)),
        t3=rescues >= 1,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctx_semantic.harness",
        description="BM25-vs-hybrid recall harness over a real content DB.",
    )
    parser.add_argument(
        "--db", type=Path, help="explicit content DB (default: projhash resolve of PROJECT)"
    )
    args = parser.parse_args(argv)
    # resolve() so the store key matches the server's absolute-path key (#5)
    db = (args.db if args.db is not None else projhash.resolve_db(PROJECT)).resolve()
    con = dbadapter.open_db(db)
    try:
        adapter = BoundAdapter(con, db)

        golds = adapter.get_many([rid for _, _, rid, _ in CASES])
        for _q, _lang, rid, title in CASES:
            if rid not in golds or golds[rid][0] != title:
                got = golds.get(rid, ("?",))[0]
                print(f"gold drift at rowid={rid}: expected {title!r}, got {got!r}")
                return 2

        embedder = Embedder()
        counts = vectors.sync(adapter, embedder.embed_batch, db)
        chunk_total = con.execute("SELECT count(*) FROM chunks").fetchone()[0]

        rows: list[tuple[str, str, str, int | None, int | None]] = []
        vcon = vectors.connect()
        try:
            for query, lang, rid, title in CASES:
                bm25 = adapter.bm25_search(query, TOP_K)
                vec = vectors.search(
                    vcon, adapter, db, embedder.embed_query(query), VEC_K
                )
                fused = hybrid.rrf(bm25, vec)
                rows.append((query, lang, title, _rank(rid, bm25), _rank(rid, fused)))
                print(f"[{lang}] {_rank(rid, bm25)=} {_rank(rid, fused)=} {query[:36]}")
        finally:
            vcon.close()
    finally:
        con.close()

    g = evaluate(rows)
    t2_label = (
        "PASS (vacuous: bm25_hits=0)"
        if g.bm25_hits == 0
        else "PASS" if g.t2 else "FAIL"
    )

    lines = [
        "# ctx-semantic recall report — BM25 vs hybrid (top-5 hit)",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M %Z')}  db: `{db}`",
        (
            f"- corpus: `{db.name}` ({chunk_total} chunks) — **real DB only,"
            f" no fixture rows**"
        ),
        (
            f"- sync before run: embedded={counts['embedded']} removed={counts['removed']}"
            f" total={counts['total']}  model=jina-v2-base-zh device={embedder.device}"
        ),
        (
            "- ranking path = T7 server pipeline at limit=5:"
            " `bm25_search(q,5)` vs `rrf(bm25, vectors.search(qvec,k=20))`"
            " (the legs + fusion `hybrid.search` applies before rendering)."
        ),
        "- hit = gold chunk **rowid** in top-5 (chunk identity, not title text).",
        "  A miss counts as rank ∞; both-miss therefore counts as not-worse.",
        "- BM25 rank comes from the top-5 BM25 leg itself; hybrid rank from the",
        "  full fused list (so rank 6+ is a miss but informative).",
        "",
        "| # | lang | BM25 | hybrid | query → gold |",
        "|---|------|------|--------|-------------|",
    ]
    for i, (query, lang, title, b, h) in enumerate(rows, start=1):
        b_hit = "✓" if b is not None and b <= TOP_K else "✗"
        h_hit = "✓" if h is not None and h <= TOP_K else "✗"
        lines.append(
            f"| {i} | {lang} | {b_hit} {_fmt(b)} | {h_hit} {_fmt(h)}"
            f" | {query} → r{CASES[i - 1][2]} «{title[:48]}» |"
        )
    by_lang: dict[str, list[int]] = {}
    for _q, lang, _t, b, h in rows:
        by_lang.setdefault(lang, []).append(
            int(h is not None and h <= TOP_K) - int(b is not None and b <= TOP_K)
        )
    lines += [
        "",
        (
            f"**Totals (n={g.n}):** BM25 top-5 hits **{g.bm25_hits}** · hybrid top-5 hits"
            f" **{g.hybrid_hits}** · not-worse **{g.not_worse}/{g.n}** · BM25-zero→hybrid-hit"
            f" **{g.rescues}**"
        ),
        "",
        (
            f"Per-lang hybrid−BM25 hit delta (zh={sum(by_lang.get('zh', []))},"
            f" en={sum(by_lang.get('en', []))}, mixed={sum(by_lang.get('mixed', []))}) —"
            " all rows real-DB."
        ),
        "",
        "## Thresholds",
        "",
        (
            f"1. hybrid hit-rate ({g.hybrid_hits}/{g.n}) >= BM25 ({g.bm25_hits}/{g.n}):"
            f" **{'PASS' if g.t1 else 'FAIL'}**"
        ),
        (
            f"2. binding not-worse >= max(1, ceil(0.8*{g.bm25_hits}))"
            f" = {max(1, ceil(0.8 * g.bm25_hits))}: {g.binding_not_worse}/{g.bm25_hits}:"
            f" **{t2_label}**"
            f" (overall not-worse incl. both-miss ∞≤∞ rows: {g.not_worse}/{g.n})"
        ),
        (
            f"3. >=1 BM25-zero-hit rescued by hybrid: {g.rescues}:"
            f" **{'PASS' if g.t3 else 'FAIL'}**"
        ),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {REPORT_PATH}")
    print(
        f"bm25_hits={g.bm25_hits} hybrid_hits={g.hybrid_hits}"
        f" not_worse={g.not_worse}/{g.n} rescues={g.rescues}"
    )
    ok = g.ok
    print(f"thresholds: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
