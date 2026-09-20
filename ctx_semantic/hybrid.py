"""Hybrid recall core: BM25 + vector retrieval fused with RRF (T6).

Per query, the BM25 leg (top ``limit``) and the vector leg (top
``max(4*limit, 20)``) are retrieved independently and fused with reciprocal
rank fusion; the top ``limit`` fused chunks are rendered as ctx_search-style
blocks ``## <title>\\n<source label>\\n<excerpt>``.

Multi-query merge (pinned by the plan): each query is retrieved and fused
independently and emitted as its own ``### 查询 N：<query>`` section in query
order; a chunk reached by several queries appears exactly once — in the
section where its fused score was highest (exact tie: earliest query wins).

Injected dependencies, both duck-typed via the Protocols below:

- ``adapter`` — the per-project view over one content DB. The T7 server
  binds ctx_semantic.dbadapter's functions to an open connection behind it;
  tests stub it. It owns FTS5 query escaping (hybrid passes query text
  through untouched) and owns the source/content_type filters, which must
  reach BOTH legs: the BM25 WHERE-side and the vector candidate set.
- ``embedder`` — ``embed_query(text) -> 1-D vector``; the real jina loader
  (ctx_semantic.embedder.Embedder) fits, tests use synthetic stubs and never
  load the model.

Empty handling: a BM25 miss degrades to the pure vector ranking; if no query
produces any chunk the result is a friendly ``NO_RESULTS`` string, never an
exception.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from typing import Protocol

import numpy as np

from .vectors import search as _vector_search

NO_RESULTS = "(no results)"
_EXCERPT_WIDTH = 240


class SearchAdapter(Protocol):
    """Narrow adapter contract hybrid.search codes against (see module doc).

    bm25_search returns best-first (rowid, bm25) pairs — lower score is
    better. filtered_rowids returns the live rowids that survive the
    source/content_type filters; they bound the vector leg's candidates.
    get_many mirrors dbadapter.get_many:
    {rowid: (title, content, source label, content_type, timestamp)}.
    """

    db_path: str | os.PathLike[str]

    def bm25_search(
        self,
        query: str,
        limit: int,
        source: str | None = None,
        content_type: str | None = None,
    ) -> list[tuple[int, float]]: ...

    def filtered_rowids(
        self, source: str | None = None, content_type: str | None = None
    ) -> set[int]: ...

    def get_many(
        self, rowids: Iterable[int]
    ) -> dict[int, tuple[str, str, str, str, str | None]]: ...


class QueryEmbedder(Protocol):
    """Query-side embedding: raw text in (no instruction prefix), vector out."""

    def embed_query(self, text: str) -> np.ndarray: ...


def rrf(
    bm25_ranked: list[tuple[int, float]],
    vec_ranked: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal-rank fusion of two best-first ranked lists.

    score(d) = sum over lists containing d of 1/(k + rank_in_list), ranks
    starting at 1; raw per-leg scores are ignored, only positions count.
    Returns (rowid, fused score) ordered by score descending (ties broken by
    rowid ascending for determinism).
    """
    fused: dict[int, float] = {}
    for ranked in (bm25_ranked, vec_ranked):
        for rank, (rid, _score) in enumerate(ranked, start=1):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def _excerpt(content: str, query: str, width: int = _EXCERPT_WIDTH) -> str:
    """~`width`-char window around the first query term found in content.

    Terms split on whitespace (the same rule as dbadapter's FTS escaping);
    a query with no findable term (e.g. space-less CJK) falls back to the
    first `width` chars of content.
    """
    lowered = content.lower()
    hit = -1
    for term in query.split():
        hit = lowered.find(term.lower())
        if hit != -1:
            break
    if hit == -1:
        return content[:width]
    start = max(0, hit - width // 4)
    end = min(len(content), start + width)
    return content[max(0, end - width) : end]


class _Narrowed:
    """vectors.SourceAdapter stand-in: live_rowids() pinned to filter matches."""

    __slots__ = ("_rowids",)

    def __init__(self, rowids: set[int]) -> None:
        self._rowids = rowids

    def live_rowids(self) -> set[int]:
        return self._rowids


def search(
    adapter: SearchAdapter,
    vectors: sqlite3.Connection,
    embedder: QueryEmbedder,
    queries: list[str],
    source: str | None = None,
    content_type: str | None = None,
    limit: int = 3,
    k: int = 60,
    synced_info: str | None = None,
) -> str:
    """Hybrid recall over `queries`; see the module docstring for the shape.

    `synced_info`, when the caller (T7) just lazy-synced, is prepended as a
    single progress line before the sections.
    """
    view = _Narrowed(adapter.filtered_rowids(source=source, content_type=content_type))
    vec_k = max(4 * limit, 20)
    fused_per_query: list[list[tuple[int, float]]] = []
    for query in queries:
        bm25 = adapter.bm25_search(
            query, limit, source=source, content_type=content_type
        )
        qvec = embedder.embed_query(query)
        vec = _vector_search(vectors, view, adapter.db_path, qvec, vec_k)
        fused_per_query.append(rrf(bm25, vec, k))

    owner: dict[int, tuple[int, float]] = {}
    for qi, fused in enumerate(fused_per_query):
        for rid, score in fused[:limit]:
            if rid not in owner or score > owner[rid][1]:
                owner[rid] = (qi, score)  # strict >: ties keep the earlier query

    sections: list[str] = []
    for qi, query in enumerate(queries):
        rowids = [r for r, _ in fused_per_query[qi][:limit] if owner[r][0] == qi]
        if not rowids:
            continue
        rows = adapter.get_many(rowids)
        blocks = [
            f"## {rows[r][0]}\n{rows[r][2] or ''}\n{_excerpt(rows[r][1], query)}"
            for r in rowids
        ]
        sections.append(f"### 查询 {qi + 1}：{query}\n\n" + "\n\n".join(blocks))

    if not sections:
        return NO_RESULTS
    body = "\n\n".join(sections)
    return f"{synced_info}\n{body}" if synced_info else body
