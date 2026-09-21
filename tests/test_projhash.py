"""Tests for ctx_semantic.projhash.

Vector tests harvest LIVE from ~/.omo/codegraph/projects/ (re-listed at test
time, never hardcoded) and verify each against the on-disk project path, per
the stale-state adversarial requirement.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path

import pytest

from ctx_semantic.projhash import (
    CONTENT_DIR,
    ProjHashError,
    db_path,
    resolve,
    resolve_db,
)

PROJECTS_DIR = Path.home() / ".omo" / "codegraph" / "projects"
SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    ".cache",
    "target",
    "build",
    "dist",
}


def _hash16(p: str) -> str:
    e = p.replace("\\", "/")
    e = "/" if re.match(r"^/+$", e) else re.sub(r"/+$", "", e)
    return hashlib.sha256(e.encode()).hexdigest()[:16]


def harvest_vectors(min_count: int = 5) -> list[tuple[str, str]]:
    """Re-derive (path, hash) vectors live: codegraph dir names + disk walk."""
    vectors: list[tuple[str, str]] = []
    if not PROJECTS_DIR.is_dir():
        return vectors
    wanted: dict[str, str] = {}
    for d in PROJECTS_DIR.iterdir():
        m = re.match(r"^(.*)-([0-9a-f]{16})$", d.name)
        if d.is_dir() and m:
            wanted.setdefault(m.group(1), m.group(2))
    home = Path.home()
    for dirpath, dirnames, _ in os.walk(home):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = os.path.basename(dirpath)
        if base in wanted and _hash16(dirpath) == wanted[base]:
            vectors.append((dirpath, wanted[base]))
        if len(vectors) >= 12:  # ponytail: cap walk, 12 verified is plenty
            break
    return vectors


def test_known_ground_truth_vectors():
    assert resolve("/home/guangbin") == "0ef2d5f23b410348"
    assert resolve("/home/guangbin/TrendRadar") == "a0531afb75e2ba0a"


def test_harvested_vectors():
    vectors = harvest_vectors()
    assert len(vectors) >= 5, f"expected >=5 live vectors, got {len(vectors)}"
    for path, expected in vectors:
        assert resolve(path) == expected, f"vector mismatch for {path}"


def test_trailing_slash_and_redundant_separators():
    assert resolve("/home/guangbin/") == resolve("/home/guangbin")
    assert resolve("/home/guangbin///") == resolve("/home/guangbin")


def test_relative_path_and_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real-project"
    real.mkdir()
    link = tmp_path / "link-project"
    link.symlink_to(real)
    monkeypatch.chdir(tmp_path)
    assert resolve("real-project") == resolve(str(real))
    assert resolve(str(link)) == resolve(str(real))


def test_nonexistent_dir_raises_not_none():
    with pytest.raises(ProjHashError):
        resolve("/nonexistent/project/dir/that/never/exists")


def test_ctx_semantic_db_override_wins(monkeypatch, tmp_path):
    # Override is authoritative even when the file does not exist.
    override = tmp_path / "override.db"
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(override))
    assert resolve_db("/home/guangbin") == override
    assert not override.exists()  # returned unvalidated, by contract


def test_opencode_project_dir_env_used_as_default(monkeypatch, tmp_path):
    monkeypatch.delenv("CTX_SEMANTIC_DB", raising=False)
    monkeypatch.setenv("OPENCODE_PROJECT_DIR", "/home/guangbin")
    monkeypatch.chdir(tmp_path)  # cwd would give a different hash
    assert resolve() == "0ef2d5f23b410348"
    assert resolve_db() == CONTENT_DIR / "0ef2d5f23b410348.db"


def test_cwd_default(monkeypatch):
    monkeypatch.delenv("CTX_SEMANTIC_DB", raising=False)
    monkeypatch.delenv("OPENCODE_PROJECT_DIR", raising=False)
    assert resolve() == resolve(os.getcwd())


def test_db_path_layout():
    p = db_path("0ef2d5f23b410348")
    assert (
        p == Path.home() / ".config/opencode/context-mode/content/0ef2d5f23b410348.db"
    )


def test_self_check_this_project():
    """Acceptance self-check: DB exists, chunks>0, codegraph dir-name match.

    2026-09-21: the /home/guangbin content DB was purged by the
    context-mode 1.0.169 upgrade, so the check anchors on THIS repo's own
    project dir — hash resolution + codegraph dir-name + a live content DB
    verified in one chain.
    """
    repo = Path(__file__).resolve().parents[1]
    h = resolve(str(repo))
    name = repo.name
    assert (PROJECTS_DIR / f"{name}-{h}").is_dir(), "codegraph dir-name mismatch"
    db = db_path(h)
    assert db.exists(), f"content DB missing: {db}"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        chunks = con.execute("select count(*) from chunks").fetchone()[0]
    finally:
        con.close()
    assert chunks > 0, "content DB has no chunks"
