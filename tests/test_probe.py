"""Tests for ctx_semantic.probe: exit codes for PASS / DRIFT / STALE / SKIP.

The probe is the post-ctx_upgrade health check; it must stay model-free
(imports dbadapter/projhash only — never ctx_semantic.embedder).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from ctx_semantic import probe
from test_dbadapter import make_fixture


def test_probe_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    db = tmp_path / "ok.db"
    make_fixture(db)
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))
    assert probe.main([]) == 0
    out = capsys.readouterr().out
    assert out.startswith("PASS:") and "chunks" in out


def test_probe_drift_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    db = tmp_path / "drifted.db"
    make_fixture(db)
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE sources ADD COLUMN upstream_extra TEXT")
    con.commit()
    con.close()
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))
    assert probe.main([]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DRIFT:") and "Fix:" in out


def test_probe_stale_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    db = tmp_path / "old.db"
    make_fixture(db)
    old = db.stat().st_mtime - 20 * 86400
    os.utime(db, (old, old))
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(db))
    assert probe.main([]) == 2
    assert capsys.readouterr().out.startswith("STALE:")


def test_probe_skip_when_unindexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setenv("CTX_SEMANTIC_DB", str(tmp_path / "never.db"))
    assert probe.main([]) == 0
    assert capsys.readouterr().out.startswith("SKIP:")
