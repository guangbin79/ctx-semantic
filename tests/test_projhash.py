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


def test_yr_root_and_windows_root_normalization():
    # "/" and "C://" are the two roots whose trailing slash MUST survive
    # stripping — hashing "/x" instead of "/" would break root-project DBs.
    from ctx_semantic.projhash import _yr

    assert _yr("///") == "/"
    assert _yr("//") == "/"
    assert _yr("C://") == "C:/"
    assert _yr("/a/b///") == "/a/b"


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


def test_self_check_live_project():
    """Acceptance self-check: hash + codegraph dir-name + live DB in one chain.

    2026-09-21: BOTH the /home/guangbin and the repo's own content DBs were
    purged by context-mode upgrades, so no single project is stable to anchor
    on. Instead verify the chain against any codegraph-indexed project whose
    content DB currently exists; skip gracefully when none survive.
    """
    for _path, h in harvest_vectors():
        db = db_path(h)
        if not db.is_file():
            continue
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            chunks = con.execute("select count(*) from chunks").fetchone()[0]
        finally:
            con.close()
        assert chunks > 0, f"content DB has no chunks: {db}"
        return  # one fully verified live project is sufficient
    pytest.skip("no codegraph-indexed project with a live content DB")


# ----------------------------------------------------------- stale_status


def test_stale_status_fresh_db_returns_none(tmp_path):
    from ctx_semantic.projhash import stale_status

    db = tmp_path / "a16hash16abcdef1.db"
    db.write_bytes(b"x")
    mtime = db.stat().st_mtime
    assert stale_status(db, now=mtime + 86400) is None  # 1 day old: live


def test_stale_status_old_db_reports_days_and_probe_hint(tmp_path):
    from ctx_semantic.projhash import stale_status

    db = tmp_path / "a16hash16abcdef1.db"
    db.write_bytes(b"x")
    mtime = db.stat().st_mtime
    warn = stale_status(db, now=mtime + 20 * 86400)
    assert warn is not None
    assert "20" in warn  # age in days is the evidence
    assert "probe" in warn  # points at the drift probe for the verdict


def test_stale_status_old_db_with_newer_sibling_names_it(tmp_path):
    from ctx_semantic.projhash import stale_status

    db = tmp_path / "a16hash16abcdef1.db"
    db.write_bytes(b"x")
    sibling = tmp_path / "b16hash16abcdef2.db"
    sibling.write_bytes(b"x")
    mtime = db.stat().st_mtime
    newer = mtime + 10 * 86400
    os.utime(sibling, (newer, newer))
    warn = stale_status(db, now=mtime + 20 * 86400)
    assert warn is not None and "sibling" in warn


def test_stale_status_missing_file_is_none_not_crash(tmp_path):
    from ctx_semantic.projhash import stale_status

    assert stale_status(tmp_path / "nope.db") is None


def test_stale_status_threshold_is_knobbed(tmp_path, monkeypatch):
    # 14d default: a 5d-old DB is fresh; dropping the knob to 1d flips it.
    from ctx_semantic import projhash

    monkeypatch.setattr(projhash, "STALE_AFTER_DAYS", 1.0)
    db = tmp_path / "a16hash16abcdef1.db"
    db.write_bytes(b"x")
    mtime = db.stat().st_mtime
    assert projhash.stale_status(db, now=mtime + 5 * 86400) is not None
