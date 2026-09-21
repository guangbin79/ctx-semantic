"""E2E stdio client driving the REAL server process (T7 acceptance).

Spawns run.sh through ``tee`` so the child's RAW stdout is captured to a file
while still feeding the MCP SDK's pipe — after the session, every captured
line must parse as a JSON-RPC frame (stdio purity: stdout carries ONLY MCP
protocol bytes; anything else corrupts the protocol stream).

Prerequisite: `uv run python -m ctx_semantic.warmup --project $HOME` (or warm
the DB given here via ``--db`` — the spawned server receives it as
CTX_SEMANTIC_DB) so first-query latency measures steady state (model load +
incremental sync, not a full embed). Records startup / first-query /
steady-state latencies and verifies returned titles really exist in the real
content DB.

``--db PATH`` is the escape hatch for corpora whose project dir no longer
resolves (context-mode purges): with it the FULL happy path runs against
that DB; without it the default project resolution applies, falling back
to the missing-DB smoke when the project has no content DB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ctx_semantic import projhash

RUN_SH = Path(__file__).resolve().parent.parent / "run.sh"
RAW_STDOUT = Path("/tmp/ctx-semantic-e2e-stdout.log")
RAW_STDERR = Path("/tmp/ctx-semantic-e2e-stderr.log")
PROJECT = Path.home()

QUERIES = [
    "向量检索与混合搜索",  # zh — must yield a Chinese-labeled section
    "embedding model load GPU fallback",
    "projhash content db hash resolution",
]
GHOST_SOURCE = "no-such-source-xyz-12345"
FIRST_CALL_TIMEOUT_S = 300  # cold model load + full sync headroom


def _text(result) -> str:
    assert not getattr(result, "isError", False), f"tool returned isError: {result}"
    return "".join(getattr(block, "text", "") for block in result.content)


def _titles(resp: str) -> list[str]:
    return [line[3:] for line in resp.splitlines() if line.startswith("## ")]


def _assert_titles_are_real(resp: str, db: Path) -> None:
    titles = [t for t in _titles(resp) if t.strip()]
    titles = [t for t in _titles(resp) if t.strip()]
    assert titles, "no title lines in response"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for title in titles:
            hits = con.execute(
                "SELECT count(*) FROM chunks WHERE title = ?", (title,)
            ).fetchone()[0]
            assert hits >= 1, f"title not present in real DB: {title!r}"
            return  # one verified real title is sufficient
    finally:
        con.close()


def _assert_stdout_purity() -> None:
    bad: list[tuple[int, str]] = []
    lines = RAW_STDOUT.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            frame = json.loads(line)
        except json.JSONDecodeError:
            bad.append((i, line[:100]))
            continue
        if not isinstance(frame, dict) or "jsonrpc" not in frame:
            bad.append((i, line[:100]))
    assert not bad, f"stdout carries non-MCP bytes: {bad[:3]}"
    print(f"stdout purity: {sum(1 for l in lines if l.strip())} frames, all JSON-RPC")


async def _run_session(db: Path | None) -> tuple[dict[str, float | str], dict[str, str]]:
    params = StdioServerParameters(
        command="sh",
        args=["-c", f"'{RUN_SH}' 2>>'{RAW_STDERR}' | tee '{RAW_STDOUT}'"],
        cwd=PROJECT,
        env={"CTX_SEMANTIC_DB": str(db)} if db is not None else None,
    )
    timings: dict[str, float | str] = {}
    responses: dict[str, str] = {}
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            t0 = time.perf_counter()
            await session.initialize()
            timings["startup_s"] = time.perf_counter() - t0

            tools = (await session.list_tools()).tools
            assert [t.name for t in tools] == ["ctx_hybrid_search"], (
                f"expected exactly one tool, got {[t.name for t in tools]}"
            )

            t0 = time.perf_counter()
            responses["first"] = _text(
                await session.call_tool(
                    "ctx_hybrid_search",
                    {"queries": [QUERIES[0]]},
                    read_timeout_seconds=FIRST_CALL_TIMEOUT_S,
                )
            )
            timings["first_query_s"] = time.perf_counter() - t0
            timings["first_embedded_line"] = (
                "yes" if "(embedded" in responses["first"] else "none (steady state)"
            )

            t0 = time.perf_counter()
            responses["steady"] = _text(
                await session.call_tool("ctx_hybrid_search", {"queries": QUERIES})
            )
            timings["steady_multi_query_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            responses["again"] = _text(
                await session.call_tool("ctx_hybrid_search", {"queries": [QUERIES[1]]})
            )
            timings["steady_single_query_s"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            responses["ghost"] = _text(
                await session.call_tool(
                    "ctx_hybrid_search",
                    {"queries": ["deploy"], "source": GHOST_SOURCE},
                )
            )
            timings["ghost_source_s"] = time.perf_counter() - t0

    return timings, responses


async def _run_missing_db_session() -> str:
    """Degradation probe for an unindexed project: one tool call, no model."""
    params = StdioServerParameters(
        command="sh",
        args=["-c", f"'{RUN_SH}' 2>>'{RAW_STDERR}' | tee '{RAW_STDOUT}'"],
        cwd=PROJECT,
    )
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        assert [t.name for t in tools] == ["ctx_hybrid_search"]
        return _text(
            await session.call_tool(
                "ctx_hybrid_search", {"queries": ["任何查询"]}
            )
        )


def _missing_db_smoke() -> int:
    """Missing content DB must degrade to friendly '(no results)', exit 0.

    projhash contract: an unindexed project resolves to a nonexistent DB;
    the server treats that as an empty knowledge base, never an error.
    """
    resp = asyncio.run(_run_missing_db_session())
    time.sleep(0.2)  # let tee flush the last frames
    assert resp == "(no results)", f"missing-DB response: {resp[:200]}"
    _assert_stdout_purity()
    print(f"project DB missing ({projhash.resolve_db(PROJECT)})")
    print("server degraded to '(no results)' — friendly empty, exit 0")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ctx_semantic.e2e",
        description="E2E stdio client driving the REAL ctx-semantic server.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="explicit content DB; the spawned server gets CTX_SEMANTIC_DB=<abs path>",
    )
    args = parser.parse_args(argv)
    RAW_STDOUT.write_text("")
    RAW_STDERR.write_text("")
    explicit = args.db.resolve() if args.db is not None else None
    if explicit is not None and not explicit.is_file():
        parser.error(f"--db does not exist: {explicit}")
    default_db = projhash.resolve_db(PROJECT)
    if explicit is None and not default_db.is_file():
        # context-mode may purge stale project DBs — exercise the friendly
        # degradation path instead of the full-result assertions.
        return _missing_db_smoke()
    timings, responses = asyncio.run(_run_session(explicit))
    time.sleep(0.2)  # let tee flush the last frames

    # --- content assertions (never trust exit codes alone) ---
    assert responses["first"] != "(no results)" and "## " in responses["first"]
    assert responses["steady"].count("### 查询") == 3, responses["steady"][:300]
    assert QUERIES[0] in responses["steady"], "Chinese query section missing"
    assert responses["ghost"] == "(no results)", responses["ghost"][:200]
    _assert_titles_are_real(responses["steady"], explicit or default_db)

    # --- stdio purity ---
    _assert_stdout_purity()

    print("timings:")
    for key, value in timings.items():
        print(f"  {key}: {value}")
    print(f"stderr capture: {RAW_STDERR} ({RAW_STDERR.stat().st_size} bytes)")
    for name in ("first", "steady", "ghost"):
        print(f"{name} response head: {responses[name].splitlines()[0][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
