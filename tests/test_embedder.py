"""Tests for ctx_semantic.embedder (T2 fakes + T4 real-model contract).

Real-model tests (module-scoped fixture, one lazy load) pin the qwen3 GGUF
contract: DIM=1024, device=cuda backed by nvidia-smi PID VRAM proof,
bilingual cosine sanity, query-prefix wiring, and n_ctx truncation. Device-
fallback paths (VRAM check falsy, OOM rebuild, CPU re-raise, nvidia-smi
parsing) run on fakes so they are deterministic on any host. GPU usage is
verified via nvidia-smi PID lookup, not device claims (the silent-fallback
trap).
"""

from __future__ import annotations

import glob
import logging
import os
import sys
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


# --- real-model contract tests (T4; llama_cpp pinned in the lock since T5) -----


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder(cache_dir=CACHE_DIR)
    e.embed_query("warmup")  # lazy load once for the whole module
    return e


def test_dim_is_1024(embedder: Embedder):
    assert DIM == 1024
    v = embedder.embed_query("维度检查")
    assert v.shape == (1024,)


def test_device_cuda_pid(embedder: Embedder):
    # "cuda" is only reported after _ensure_loaded's nvidia-smi PID check
    # passed; this re-verifies VRAM is still held by THIS pid. Hermeticity
    # (jina-era sys.prefix convention, adapted): llama-cpp dlopens libcudart/
    # libcublas from nvidia wheel dirs under sys.prefix — without them CUDA
    # is genuinely unloadable and the Embedder legitimately lands on cpu
    # (that degradation is covered by the fakes fallback tests above).
    if embedder_mod.shutil.which("nvidia-smi") is None:
        pytest.skip("no nvidia-smi on PATH — GPU presence cannot be verified")
    pattern = str(Path(sys.prefix) / "lib" / "**" / "nvidia" / "*" / "lib")
    if not glob.glob(pattern, recursive=True):
        pytest.skip(
            "CUDA wheel libs unavailable under sys.prefix — degradation is "
            "caught by embedder fallback tests"
        )
    assert embedder.device == "cuda", f"device reported: {embedder.device}"
    vram = embedder_mod._pid_vram_on_gpu(os.getpid())
    assert vram is not None, "device=cuda but nvidia-smi shows no VRAM for this pid"
    if vram != "unknown":  # row exists but no memory column
        assert float(vram.split()[0]) > 0, f"VRAM reported as {vram}"


def test_bilingual_cosine_sanity(embedder: Embedder):
    # qwen3-embedding-0.6b is zh/en bilingual: each query must match its
    # related doc far better than an unrelated concept — zh, en, and the
    # zh-query→en-doc cross-lingual pair. Margin 0.2: measured gaps ~0.54
    # (2026-09-28), robust to determinism noise, not a knife-edge.
    related_doc = embedder.embed_batch(
        ["model routing\nDistribute requests across models based on load"]
    )[0]
    zh_doc = embedder.embed_batch(["模型路由与分流策略\n根据负载将请求分发到不同模型"])[0]
    unrelated = embedder.embed_batch(["晚餐食谱\n今日晚餐菜单与采购清单"])[0]
    pairs = {
        "zh→zh": (embedder.embed_query("模型路由"), zh_doc),
        "zh→en": (embedder.embed_query("模型路由"), related_doc),
        "en→en": (embedder.embed_query("model routing"), related_doc),
    }
    for name, (q, pos) in pairs.items():
        cos_pos = float(np.dot(q, pos))
        cos_neg = float(np.dot(q, unrelated))
        assert cos_pos > cos_neg, f"{name}: {cos_pos=} !> {cos_neg=}"
        assert cos_pos - cos_neg > 0.2, f"{name}: gap {cos_pos - cos_neg:.4f} too small"


def test_query_prefix_applied_and_effective(embedder: Embedder):
    # embed_query(t) must be the IDENTICAL computation to embedding the
    # preformatted string (cos >= 0.99999 after float32 normalization), and
    # the prefix must actually change the vector vs the raw text
    # (cos < 0.999; measured 0.840) — proves _format_query is wired AND
    # load-bearing, not a decorative call.
    for t in ("怎么排查内存泄漏", "how to debug a memory leak"):
        q = embedder.embed_query(t)
        pre = embedder.embed_batch([embedder_mod._format_query(t)])[0]
        raw = embedder.embed_batch([t])[0]
        cos_same = float(np.dot(q, pre))
        cos_raw = float(np.dot(q, raw))
        assert cos_same >= 0.99999, f"{t!r}: {cos_same=}"
        assert cos_raw < 0.999, f"{t!r}: prefix changed nothing ({cos_raw=})"


def test_long_text_beyond_n_ctx_truncates_not_raises(embedder: Embedder):
    # ~9.6k tokens > N_CTX (8192): llama.cpp truncates the input at n_ctx
    # instead of raising; the output stays a normalized 1024-dim vector.
    # Corpus p99 is 2146 tokens — only extreme outliers ever hit this path.
    para = (
        "The quick brown fox jumps over the lazy dog. "
        "这是一段用于测试长文本截断行为的混合语言段落，包含中英文内容。"
    ) * 300
    v = embedder.embed_batch([para])[0]
    assert v.shape == (DIM,)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
