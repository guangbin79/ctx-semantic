"""Project-hash resolution compatible with context-mode's content DB naming.

Algorithm (reverse-engineered from context-mode's cli.bundle.mjs — see
~/.omo/evidence/ctx-semantic/task-2-projhash.md for the verbatim snippets):

    yr(p):  backslashes -> "/", trailing slashes stripped (root "/" kept)
    rt(p):  sha256( yr(p), lowercased on darwin/win32 only ).hexdigest()[:16]
    db:     <content_dir>/<rt(project_dir)>.db

This module additionally applies os.path.realpath() before hashing so symlinked
and relative inputs resolve to the same hash as the real directory. For inputs
that are already absolute real paths (e.g. the cwd opencode gives an MCP child
process) this is byte-identical to context-mode's own yr().

Resolution priority for db_path()/resolve():
    1. CTX_SEMANTIC_DB env var — explicit DB file path, wins over everything.
       It is returned as-is even when the file does not exist yet; callers
       that open it must handle FileNotFoundError themselves (the override is
       authoritative, not validated).
    2. OPENCODE_PROJECT_DIR env var — used as the project dir when none is
       passed explicitly.
    3. Current working directory.

Failures raise ProjHashError; this module never returns None silently.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

CONTENT_DIR = Path.home() / ".config" / "opencode" / "context-mode" / "content"

_ROOT_ONLY = re.compile(r"^/+$")
_WIN_ROOT = re.compile(r"^[A-Za-z]:/+$")


class ProjHashError(Exception):
    """Project dir could not be resolved to a context-mode content DB."""


def _yr(path: str) -> str:
    """Port of context-mode's yr(): slash normalization, no realpath."""
    e = path.replace("\\", "/")
    if _ROOT_ONLY.match(e):
        return "/"
    if _WIN_ROOT.match(e):
        return e[:2] + "/"
    return re.sub(r"/+$", "", e)


def _hash16(path: str) -> str:
    """Port of context-mode's rt(): sha256 of the normalized path, 16 hex chars."""
    e = _yr(path)
    if os.name == "nt" or os.uname().sysname == "Darwin":
        e = e.lower()
    return hashlib.sha256(e.encode()).hexdigest()[:16]


def resolve(project_dir: str | os.PathLike[str] | None = None) -> str:
    """Return context-mode's 16-hex-char project hash for project_dir.

    project_dir defaults to $OPENCODE_PROJECT_DIR, then the cwd. Raises
    ProjHashError if the directory does not exist.
    """
    if project_dir is None:
        project_dir = os.environ.get("OPENCODE_PROJECT_DIR") or os.getcwd()
    real = os.path.realpath(project_dir)
    if not os.path.isdir(real):
        raise ProjHashError(f"project dir does not exist: {project_dir}")
    return _hash16(real)


def db_path(hash_: str) -> Path:
    """Content DB path for a 16-hex project hash (no existence check)."""
    return CONTENT_DIR / f"{hash_}.db"


def resolve_db(project_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the content DB file to use, applying the env priority chain.

    CTX_SEMANTIC_DB wins over everything and is returned unvalidated (see
    module docstring). Otherwise the path is
    CONTENT_DIR/<resolve(project_dir)>.db — which may or may not exist yet
    for rarely-used projects; callers should treat a missing file as empty.
    """
    env = os.environ.get("CTX_SEMANTIC_DB")
    if env:
        return Path(env)
    return db_path(resolve(project_dir))
