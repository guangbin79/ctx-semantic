"""ctx-semantic MCP sidecar server (T7) — one stdio tool over the hybrid core.

``ctx_hybrid_search`` pipeline per call: projhash.resolve_db → dbadapter
open (ro) → vectors.connect (sidecar store) → lazy per-DB vectors.sync →
hybrid.search. The embedder model loads on the FIRST query, never at
startup — process start stays seconds-fast so opencode's MCP registration
does not stall. T8 registers the server only after `warmup` has pre-embedded,
so the steady-state first query is model load + a zero-work incremental sync.

stdio purity: stdout carries ONLY MCP protocol bytes. The model stack's
chatter (tqdm, warnings, onnxruntime) writes to stderr by default; this
module never prints. Hub env hygiene (mirror + xet off + proxies off) is
applied at ctx_semantic.embedder import time, before fastembed loads.

project_path is the documented override of the cwd/$OPENCODE_PROJECT_DIR
default; $CTX_SEMANTIC_DB still wins over everything (projhash contract).
Run via ../run.sh (LD_LIBRARY_PATH for the CUDA EP) or:
    uv run python -m ctx_semantic.server
"""

from __future__ import annotations

import threading
import time

from mcp.server.mcpserver import MCPServer

from ctx_semantic import dbadapter, hybrid, projhash, vectors
from ctx_semantic.binding import BoundAdapter
from ctx_semantic.embedder import Embedder

mcp = MCPServer("ctx-semantic")

# Re-sync a DB at most this often: keeps the vector leg live over long
# server sessions while BM25 stays live on every query (syncs are cheap
# in steady state — embedded=0 when nothing changed).
RESYNC_INTERVAL_S = 300.0
# A sync that re-embedded EVERY stored chunk (cold start, model swap,
# hash-format migration) is the expensive shape; widen the next interval.
FULL_REEMBED_RESYNC_INTERVAL_S = 3600.0

_embedder: Embedder | None = None
_synced_at: dict[str, float] = {}  # key -> monotonic claim time (set at claim)
_sync_interval: dict[str, float] = {}  # key -> seconds until next sync is due
_state_lock = threading.Lock()


def _get_embedder() -> Embedder:
    """Construct the Embedder lazily and exactly once across threads."""
    global _embedder
    with _state_lock:
        if _embedder is None:
            _embedder = Embedder()
        return _embedder


def _claim_sync(key: str) -> bool:
    """Claim the sync slot for key; True only for the one caller that wins.

    The claim timestamp is written under the lock BEFORE the sync runs, so
    cold-start/TTL-expiry dedup is exact: concurrent callers of a stale key
    see a fresh claim and skip instead of queueing their own full re-embed.
    A sync that raises must release the claim so the next call retries.
    """
    now = time.monotonic()
    with _state_lock:
        if key in _synced_at and now - _synced_at[key] <= _sync_interval.get(
            key, RESYNC_INTERVAL_S
        ):
            return False
        _synced_at[key] = now
        return True



@mcp.tool()
def ctx_hybrid_search(
    queries: list[str],
    source: str | None = None,
    content_type: str | None = None,
    limit: int = 3,
    project_path: str | None = None,
) -> str:
    """Hybrid BM25 + vector search over this project's context-mode knowledge base.

    Searches the indexed session/decision markdown of the CURRENT project
    (mixed Chinese/English) with reciprocal-rank fusion of an FTS5 BM25 leg
    and a jina-v2-base-zh vector leg. Each query gets its own
    '### 查询 N：<query>' section of ctx_search-style blocks; a chunk hit by
    several queries appears exactly once.

    Args:
        queries: 1-3 search queries (different phrasings improve recall).
        source: optional exact source label filter (sources.label).
        content_type: optional exact content_type filter (e.g. "decision",
            "session", "note").
        limit: chunks per query section (default 3).
        project_path: optional project directory override. Default resolution
            is $CTX_SEMANTIC_DB > $OPENCODE_PROJECT_DIR > the server's cwd
            (opencode spawns MCP children with the project root as cwd).

    Returns:
        Markdown sections, or "(no results)". The first call in a server
        lifetime may embed newly indexed chunks and prepends one
        "(embedded N chunks in Xs)" progress line.
    """
    if not 1 <= len(queries) <= 3:
        raise ValueError("queries must contain 1-3 items")
    limit = max(1, min(limit, 10))
    db = projhash.resolve_db(project_path)
    if not db.is_file():
        # projhash contract: an unindexed project resolves to a nonexistent
        # DB — that is an empty knowledge base, not an error.
        return hybrid.NO_RESULTS
    con = dbadapter.open_db(db)
    try:
        adapter = BoundAdapter(con, db)
        synced_info: str | None = None
        key = str(db)
        if _claim_sync(key):
            started = time.perf_counter()
            try:
                counts = vectors.sync(adapter, _get_embedder().embed_batch, db)
            except BaseException:
                with _state_lock:
                    _synced_at.pop(key, None)  # a false-fresh claim blocks retries
                raise
            elapsed = time.perf_counter() - started
            if counts["embedded"] > 0:
                synced_info = f"(embedded {counts['embedded']} chunks in {elapsed:.1f}s)"
            # embedded == total marks a full re-embed (every stored chunk
            # changed at once); widen the next interval.
            interval = (
                FULL_REEMBED_RESYNC_INTERVAL_S
                if counts["total"] > 0 and counts["embedded"] >= counts["total"]
                else RESYNC_INTERVAL_S
            )
            with _state_lock:
                _sync_interval[key] = interval
        vcon = vectors.connect()
        try:
            return hybrid.search(
                adapter,
                vcon,
                _get_embedder(),
                queries,
                source=source,
                content_type=content_type,
                limit=limit,
                synced_info=synced_info,
            )
        finally:
            vcon.close()
    finally:
        con.close()


if __name__ == "__main__":
    mcp.run("stdio")
