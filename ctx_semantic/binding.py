"""Bind dbadapter's functions to an open connection (the T7 binding layer).

hybrid.search codes against SearchAdapter (bm25_search / filtered_rowids /
get_many + db_path); vectors.sync/search code against SourceAdapter
(snapshot_chunks / source_hashes / live_rowids). BoundAdapter satisfies BOTH
duck-typed protocols over one read-only connection, so the server and the
warmup CLI wire the same object into the whole pipeline.

Filters genuinely restrict BOTH legs: dbadapter.bm25_search applies them
WHERE-side, and filtered_rowids bounds the vector leg's candidate set.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable

from ctx_semantic import dbadapter

ChunkRow = tuple[int, str, str, int, str, str | None]


class BoundAdapter:
    """SearchAdapter + SourceAdapter over one ro connection to a content DB."""

    __slots__ = ("con", "db_path")

    def __init__(self, con: sqlite3.Connection, db_path: str | os.PathLike[str]) -> None:
        self.con = con
        self.db_path = str(db_path)

    # --- hybrid.SearchAdapter -------------------------------------------------

    def bm25_search(
        self,
        query: str,
        limit: int,
        source: str | None = None,
        content_type: str | None = None,
    ) -> list[tuple[int, float]]:
        return dbadapter.bm25_search(
            self.con, query, limit, source_filter=source, content_type=content_type
        )

    def filtered_rowids(
        self, source: str | None = None, content_type: str | None = None
    ) -> set[int]:
        return dbadapter.filtered_rowids(self.con, source, content_type)

    def get_many(
        self, rowids: Iterable[int]
    ) -> dict[int, tuple[str, str, str, str, str | None]]:
        return dbadapter.get_many(self.con, rowids)

    # --- vectors.SourceAdapter ------------------------------------------------

    def snapshot_chunks(self) -> list[ChunkRow]:
        return dbadapter.snapshot_chunks(self.con)

    def source_hashes(self) -> dict[int, str]:
        return dbadapter.source_hashes(self.con)

    def live_rowids(self) -> set[int]:
        return dbadapter.live_rowids(self.con)
