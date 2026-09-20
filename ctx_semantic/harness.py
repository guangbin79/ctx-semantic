"""One-shot BM25-vs-hybrid recall harness over the REAL project DB (T8).

Ground truth is programmatic: gold chunks were sampled read-only from the
/home/guangbin content DB (projhash-resolved), and each query below is a
hand-written paraphrase of one sampled chunk (Chinese paraphrases of English
chunks carry zero lexical overlap — the case pure BM25 cannot serve; English
and mixed queries keep the BM25 leg non-empty so RRF fusion is genuinely
exercised). A hit means the gold chunk's ROWID appears in the top-5 — chunk
identity, never title substrings.

Per query the harness runs the exact T7 server ranking path at limit=5:
dbadapter.bm25_search(query, 5) vs hybrid.rrf(bm25, vectors.search(qvec,
k=20)) — the same two legs and fusion hybrid.search applies before its
markdown rendering. Read-only on the context-mode DB; the sidecar store gets
one idempotent vectors.sync (steady state: embedded=0).

Writes the report to ~/.omo/evidence/ctx-semantic/recall-report.md and exits
0 only if all three thresholds hold: (1) hybrid top-5 hit-rate >= BM25's,
(2) >=7 of the queries not worse (hybrid rank <= BM25 rank; a miss counts as
rank infinity, so both-miss is "not worse"), (3) >=1 query where BM25 misses
top-5 and hybrid hits. A FAIL is reported honestly, not tuned away.
"""

from __future__ import annotations

import time
from pathlib import Path

from ctx_semantic import dbadapter, hybrid, projhash, vectors
from ctx_semantic.binding import BoundAdapter
from ctx_semantic.embedder import Embedder

PROJECT = "/home/guangbin"
TOP_K = 5  # hit window — mirrors the server tool's limit
VEC_K = 20  # max(4*TOP_K, 20) — hybrid.search's vector-leg width
REPORT_PATH = Path.home() / ".omo" / "evidence" / "ctx-semantic" / "recall-report.md"

# (query, lang, gold_rowid, gold_title) — paraphrases authored 2026-09-20 from
# a read-only sample of the real DB. gold_title guards rowid drift: the harness
# aborts (exit 2) if the rowid no longer holds the recorded chunk.
CASES: list[tuple[str, str, int, str]] = [
    # --- zh: pure-CJK paraphrases of English chunks (zero lexical overlap) ---
    ("记忆分层里哪一层只活在当前会话里、会话一结束就被硬性丢弃", "zh", 53,
     "ai-memory - Architecture > Storage architecture (3)"),
    ("那个能装进手机、手表、机器人、智能家居和单片机里运行的迷你基础模型体积有多大",
     "zh", 1628, "A foundation model for mobiles, wearables, robots, smart home, automotive and mi"),
    ("鉴权用的活跃凭证分成几类、浏览器兼容的过渡路径又是哪条", "zh", 60,
     "ai-memory - Architecture > HTTP authentication classes"),
    ("外壳脚本钩子把生命周期事件数据用什么方式投递给服务端", "zh", 48,
     "ai-memory - Architecture > Data flow (2)"),
    ("故障发生后逐级恢复的分层策略一共有几层、分别是什么", "zh", 92,
     "Squad > Watch Mode — Ralph's Automated Polling > Error Recovery (4-Tier Escalation)"),
    ("这台机器设的是什么时区、有没有夏令时", "zh", 573, "timezone and cron env"),
    # --- en: lexical overlap present — BM25 leg non-empty, RRF fused ---
    ("human-directed development team specialists frontend backend tester Copilot",
     "en", 76, "Squad > What is Squad?"),
    ("Ralph continuously polls for work and dispatches agents watch mode",
     "en", 87, "Squad > Watch Mode — Ralph's Automated Polling"),
    ("bundle local embedding model API-key-free homelab image bloat",
     "en", 67, "ai-memory - Architecture > Future work"),
    ("Needle Laddered Simple Attention Network Monarch Hadamard MLP",
     "en", 1631, "Needle 3 is a Laddered Simple Attention Network: a Monarch Hadamard MLP in place"),
    # --- mixed zh+en: single Latin token keeps BM25 matching, vector carries semantics ---
    ("Squad 一条命令拉起一支帮你推进代码的团队", "mixed", 75, "Squad"),
    ("watch 模式怎么开启自动执行、轮询间隔怎么设", "mixed", 88,
     "Squad > Watch Mode — Ralph's Automated Polling > Quick Start"),
]


def _rank(rowid: int, ranked: list[tuple[int, float]]) -> int | None:
    """1-based position of rowid in a best-first list, None when absent."""
    for pos, (rid, _score) in enumerate(ranked, start=1):
        if rid == rowid:
            return pos
    return None


def _fmt(rank: int | None) -> str:
    return "—" if rank is None else str(rank)


def main() -> int:
    db = projhash.resolve_db(PROJECT)
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

    bm25_hits = sum(1 for r in rows if r[3] is not None and r[3] <= TOP_K)
    hybrid_hits = sum(1 for r in rows if r[4] is not None and r[4] <= TOP_K)
    not_worse = sum(
        1 for r in rows if r[4] is not None and (r[3] is None or r[4] <= r[3])
    )
    rescues = sum(1 for r in rows if r[3] is None and r[4] is not None)
    n = len(rows)
    t1 = hybrid_hits >= bm25_hits
    t2 = not_worse >= 7
    t3 = rescues >= 1

    lines = [
        "# ctx-semantic recall report — BM25 vs hybrid (top-5 hit)",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M %Z')}  project: `{PROJECT}`",
        f"- db: `{db}` ({chunk_total} chunks)  — **real DB only, no fixture rows**",
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
            f"**Totals (n={n}):** BM25 top-5 hits **{bm25_hits}** · hybrid top-5 hits"
            f" **{hybrid_hits}** · not-worse **{not_worse}/{n}** · BM25-zero→hybrid-hit"
            f" **{rescues}**"
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
            f"1. hybrid hit-rate ({hybrid_hits}/{n}) >= BM25 ({bm25_hits}/{n}):"
            f" **{'PASS' if t1 else 'FAIL'}**"
        ),
        f"2. not-worse >= 7: {not_worse}/{n}: **{'PASS' if t2 else 'FAIL'}**",
        (
            f"3. >=1 BM25-zero-hit rescued by hybrid: {rescues}:"
            f" **{'PASS' if t3 else 'FAIL'}**"
        ),
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"report: {REPORT_PATH}")
    print(f"bm25_hits={bm25_hits} hybrid_hits={hybrid_hits}"
          f" not_worse={not_worse}/{n} rescues={rescues}")
    ok = t1 and t2 and t3
    print(f"thresholds: {'ALL PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
