"""Tests for ctx_semantic.embedder (T2: llama.cpp backend, fakes level).

The jina-era real-model tests were removed with the backend swap and return
in T4 (they need llama_cpp in the lock, which lands in T5); nothing in this
file may trigger a real model load. GPU usage inside Embedder is verified
via nvidia-smi PID lookup, not device claims (the silent-fallback trap).

Device-fallback paths (VRAM check falsy, OOM rebuild, CPU re-raise,
nvidia-smi parsing) run on fakes so they are deterministic on any host.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ctx_semantic import embedder as embedder_mod
from ctx_semantic.embedder import (
    DEFAULT_CACHE_DIR,
    DIM,
    Embedder,
)

CACHE_DIR = Path(DEFAULT_CACHE_DIR)


def test_format_query_exact_string():
    # Qwen3 official query prefix; embed_batch (doc side) adds nothing.
    q = "怎么排查内存泄漏"
    assert embedder_mod._format_query(q) == (
        "Instruct: Given a web search query, retrieve relevant passages "
        "that answer the query\nQuery: 怎么排查内存泄漏"
    )


def test_embed_batch_empty():
    mat = Embedder().embed_batch([])
    assert mat.shape == (0, DIM)
    assert mat.dtype == np.float32


# --- deterministic device-fallback paths (FakeModel, no real load) -----------


class FakeModel:
    """Backend-neutral stand-in: yields (n, DIM) float64 rows per embed().

    fail_on_call makes the Nth embed() call raise — call 1 is _ensure_loaded's
    device probe, call 2+ is a real embed_batch.
    """

    def __init__(self, fail_on_call: int | None = None, message: str = "CUDA out of memory"):
        self.fail_on_call = fail_on_call
        self.message = message
        self.calls = 0

    def embed(self, texts, **kwargs):
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError(self.message)
        return [np.full(DIM, 0.5, dtype=np.float64) for _ in texts]


def _fake_factory(made: list[bool], models: list[FakeModel]):
    """_new_model stand-in recording use_gpu flags, serving preset models."""

    def new_model(*args):  # bound call: (self, use_gpu)
        made.append(args[-1])
        return models[len(made) - 1]

    return new_model


def test_ensure_loaded_cpu_fallback_when_vram_check_falsy(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    # Silent-fallback trap: CUDA session created but nvidia-smi shows no VRAM
    # for this pid — the Embedder must refuse to claim GPU and retry on CPU.
    made: list[bool] = []
    monkeypatch.setattr(
        Embedder, "_new_model", _fake_factory(made, [FakeModel(), FakeModel()])
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: None)

    e = Embedder(cache_dir=CACHE_DIR)
    with caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"):
        v = e.embed_query("fallback probe")

    assert e.device == "cpu"
    assert made == [True, False]  # CUDA attempted, then CPU retry
    assert v.shape == (DIM,)
    assert "falling back to CPU" in caplog.text


def test_ensure_loaded_cuda_success_records_vram(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    made: list[bool] = []
    monkeypatch.setattr(
        Embedder, "_new_model", _fake_factory(made, [FakeModel()])
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: "512 MiB")

    e = Embedder(cache_dir=CACHE_DIR)
    with caplog.at_level(logging.INFO, logger="ctx_semantic.embedder"):
        e.embed_query("gpu probe")

    assert e.device == "cuda"
    assert made == [True]  # no CPU retry on a healthy GPU path
    assert "embedder on GPU" in caplog.text and "512 MiB" in caplog.text

    # CUDA-time OOM (probe ok, first batch raises) rebuilds the model on CPU
    # once and retries; `device` flips to "cpu".
    made: list[bool] = []
    monkeypatch.setattr(
        Embedder,
        "_new_model",
        _fake_factory(made, [FakeModel(fail_on_call=2), FakeModel()]),
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: "512 MiB")

    e = Embedder(cache_dir=CACHE_DIR)
    with caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"):
        mat = e.embed_batch(["a", "b"])  # probe loads cuda; batch OOMs

    assert mat.shape == (2, DIM)  # the CPU retry produced normalized output
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
    assert made == [True, False]  # rebuilt on CPU, retried
    assert e.device == "cpu"
    assert "rebuilding embedder on CPU" in caplog.text


def test_embed_batch_cpu_failure_reraises_without_rebuild(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    # A CPU-side embed failure must NOT trigger the GPU fallback rebuild —
    # the OOM retry is cuda-only; the error surfaces unchanged.
    made: list[bool] = []
    monkeypatch.setattr(
        Embedder,
        "_new_model",
        _fake_factory(
            made, [FakeModel(), FakeModel(fail_on_call=2, message="cpu backend exploded")]
        ),
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: None)

    e = Embedder(cache_dir=CACHE_DIR)
    e._ensure_loaded()  # vram falsy -> cuda attempt, cpu fallback at load
    assert e.device == "cpu"

    with (
        caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"),
        pytest.raises(RuntimeError, match="cpu backend exploded"),
    ):
        e.embed_batch(["x"])  # cpu model's 2nd call: re-raise, no rebuild

    assert made == [True, False]  # no third model was built
    assert "rebuilding embedder on CPU" not in caplog.text


def test_ensure_loaded_probe_shape_mismatch_raises(monkeypatch):
    # DIM is pinned at 1024: a model emitting another dim is a wrong artifact
    # and must fail loudly, not silently degrade.
    class WrongDimModel:
        def embed(self, texts, **kwargs):
            return [np.full(2048, 0.5, dtype=np.float64) for _ in texts]

    monkeypatch.setattr(
        Embedder, "_new_model", lambda *args: WrongDimModel()
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: "1 MiB")
    e = Embedder(cache_dir=CACHE_DIR)
    with pytest.raises(AssertionError, match="1024"):
        e.embed_query("shape check")


# --- _pid_vram_on_gpu: nvidia-smi parsing table ------------------------------


@pytest.mark.parametrize(
    ("which", "run_raises", "stdout", "expected"),
    [
        (None, False, None, None),  # no nvidia-smi on PATH
        ("nvidia-smi", OSError("spawn failed"), None, None),  # query failure
        ("nvidia-smi", False, "99999, 1 GiB\n", None),  # other pids only
        ("nvidia-smi", False, f"{os.getpid()}, 512 MiB\n", "512 MiB"),
        ("nvidia-smi", False, f"{os.getpid()}\n", "unknown"),  # no memory column
    ],
)
def test_pid_vram_on_gpu_table(monkeypatch, which, run_raises, stdout, expected):
    monkeypatch.setattr(embedder_mod.shutil, "which", lambda name: which)

    if run_raises:

        def fake_run(*a, **k):
            raise run_raises

    else:

        def fake_run(*a, **k):
            return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(embedder_mod.subprocess, "run", fake_run)
    assert embedder_mod._pid_vram_on_gpu(os.getpid()) == expected
