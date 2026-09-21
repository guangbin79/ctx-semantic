"""Tests for ctx_semantic.embedder (T4).

The model load is module-scoped (~1.5 s warm cache; models/ is the pinned
622 MB artifact from task-1-scaffold.md — never deleted by tests). GPU usage
inside Embedder is verified via nvidia-smi PID lookup, not provider names
(the task-1 silent-fallback trap).

Device-fallback paths (VRAM check falsy, OOM rebuild, CPU re-raise,
lib preload fixpoint, nvidia-smi parsing) run on fakes so they are
deterministic on any host; the real-model tests keep contract checks.
"""

from __future__ import annotations

import ctypes
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
    _apply_hf_env,
)

CACHE_DIR = Path(DEFAULT_CACHE_DIR)



@pytest.fixture(scope="module")
def embedder() -> Embedder:
    e = Embedder(cache_dir=CACHE_DIR)
    e.embed_query("warmup")  # lazy load once for the whole module
    return e


def test_dim_is_768(embedder: Embedder):
    v = embedder.embed_query("维度检查")
    assert DIM == 768
    assert v.shape == (768,)


def test_device_report_line(embedder: Embedder):
    # "cuda" is only set after the nvidia-smi PID check passed inside
    # _ensure_loaded; a silent CPU fallback would have logged + flipped this.
    # Hermeticity: uv --with overlays put an ephemeral venv at sys.prefix
    # while the nvidia wheels stay in the project venv — there the CUDA EP
    # cannot load at all, so a cpu report is an env artifact, not a bug.
    # Probe with the embedder's own discovery rule (nvidia lib dirs under
    # sys.prefix); when libs ARE present the cuda assert stays intentional.
    pattern = str(Path(sys.prefix) / "lib" / "**" / "nvidia" / "*" / "lib")
    if not glob.glob(pattern, recursive=True):
        pytest.skip(
            "CUDA EP unavailable — degradation is caught by embedder fallback tests"
        )
    assert embedder.device == "cuda", f"device reported: {embedder.device}"


def test_bilingual_cosine_sanity(embedder: Embedder):
    # jina-v2-base-zh is zh/en bilingual: the zh query must match its English
    # paraphrase far better than an unrelated concept. Strict inequality with
    # a 0.2 margin — measured gap is ~0.68, so this is robust to model
    # determinism noise, not a knife-edge threshold.
    q = embedder.embed_query("模型路由")
    pos = embedder.embed_query("model routing 分流")
    neg = embedder.embed_query("晚餐食谱")
    cos_pos = float(np.dot(q, pos))
    cos_neg = float(np.dot(q, neg))
    assert cos_pos > cos_neg, f"{cos_pos=} !> {cos_neg=}"
    assert cos_pos - cos_neg > 0.2, f"gap {cos_pos - cos_neg:.4f} too small"


def test_embed_batch_contract(embedder: Embedder):
    texts = ["标题\n\n内容第一段", "model routing", "晚餐食谱 今日菜单"]
    mat = embedder.embed_batch(texts)
    assert mat.shape == (3, DIM)
    assert mat.dtype == np.float32
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_embed_batch_empty():
    mat = Embedder().embed_batch([])
    assert mat.shape == (0, DIM)
    assert mat.dtype == np.float32


def test_query_unit_norm_float32(embedder: Embedder):
    v = embedder.embed_query("单位范数检查")
    assert v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_apply_hf_env_proxy_opt_out(monkeypatch: pytest.MonkeyPatch):
    # OCR #11: CTX_SEMANTIC_KEEP_PROXY=1 keeps proxy vars (e.g. when the
    # mirror itself must be reached through one); default still strips them.
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:7890")
    monkeypatch.setenv("CTX_SEMANTIC_KEEP_PROXY", "1")
    _apply_hf_env()
    assert os.environ["https_proxy"] == "http://127.0.0.1:7890"
    monkeypatch.delenv("CTX_SEMANTIC_KEEP_PROXY")
    _apply_hf_env()
    assert "https_proxy" not in os.environ


# --- deterministic device-fallback paths (FakeModel, no real load) -----------

CUDA_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
CPU_PROVIDERS = ["CPUExecutionProvider"]


class FakeModel:
    """fastembed-shaped stand-in: yields (n, DIM) float64 rows.

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


def _fake_factory(made: list[list[str]], models: list[FakeModel]):
    """_new_model stand-in recording provider lists, serving preset models."""

    def new_model(*args):  # bound call: (self, providers)
        made.append(list(args[-1]))
        return models[len(made) - 1]

    return new_model


def test_ensure_loaded_cpu_fallback_when_vram_check_falsy(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    # T1 trap: CUDA session created but nvidia-smi shows no VRAM for this pid
    # — the Embedder must refuse to claim GPU and retry once on CPU.
    made: list[list[str]] = []
    monkeypatch.setattr(
        Embedder, "_new_model", _fake_factory(made, [FakeModel(), FakeModel()])
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: None)

    e = Embedder(cache_dir=CACHE_DIR)
    with caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"):
        v = e.embed_query("fallback probe")

    assert e.device == "cpu"
    assert made == [CUDA_PROVIDERS, CPU_PROVIDERS]  # CUDA attempted, then CPU retry
    assert v.shape == (DIM,)
    assert "falling back to CPU" in caplog.text


def test_ensure_loaded_cuda_success_records_vram(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    made: list[list[str]] = []
    monkeypatch.setattr(
        Embedder, "_new_model", _fake_factory(made, [FakeModel()])
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: "512 MiB")

    e = Embedder(cache_dir=CACHE_DIR)
    with caplog.at_level(logging.INFO, logger="ctx_semantic.embedder"):
        e.embed_query("gpu probe")

    assert e.device == "cuda"
    assert made == [CUDA_PROVIDERS]  # no CPU retry on a healthy GPU path
    assert "embedder on GPU" in caplog.text and "512 MiB" in caplog.text


    # CUDA-time OOM (probe ok, first batch raises) rebuilds the model on CPU
    # once and retries; `device` flips to "cpu".
    made: list[list[str]] = []
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
    assert made == [CUDA_PROVIDERS, CPU_PROVIDERS]  # rebuilt on CPU, retried
    assert e.device == "cpu"
    assert "rebuilding embedder on CPU" in caplog.text


def test_embed_batch_cpu_failure_reraises_without_rebuild(
    monkeypatch: pytest.MonkeyPatch, caplog
):
    # A CPU-side embed failure must NOT trigger the GPU fallback rebuild —
    # the OOM retry is cuda-only; the error surfaces unchanged.
    made: list[list[str]] = []
    monkeypatch.setattr(
        Embedder,
        "_new_model",
        _fake_factory(
            made, [FakeModel(), FakeModel(fail_on_call=2, message="onnx exploded")]
        ),
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: None)

    e = Embedder(cache_dir=CACHE_DIR)
    e._ensure_loaded()  # vram falsy -> cuda attempt, cpu fallback at load
    assert e.device == "cpu"

    with (
        caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"),
        pytest.raises(RuntimeError, match="onnx exploded"),
    ):
        e.embed_batch(["x"])  # cpu model's 2nd call: re-raise, no rebuild

    assert made == [CUDA_PROVIDERS, CPU_PROVIDERS]  # no third model was built
    assert "rebuilding embedder on CPU" not in caplog.text


def test_ensure_loaded_probe_shape_mismatch_raises(monkeypatch):
    # DIM is pinned at 768 (task-1-scaffold.md): a model emitting another dim
    # is a wrong artifact and must fail loudly, not silently degrade.
    class WrongDimModel:
        def embed(self, texts, **kwargs):
            return [np.full(1024, 0.5, dtype=np.float64) for _ in texts]

    monkeypatch.setattr(
        Embedder, "_new_model", lambda *args: WrongDimModel()
    )
    monkeypatch.setattr(embedder_mod, "_pid_vram_on_gpu", lambda pid: "1 MiB")
    e = Embedder(cache_dir=CACHE_DIR)
    with pytest.raises(AssertionError, match="768"):
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


# --- _preload_cuda_libs: fixpoint + env + total failure -----------------------


def _fake_glob(monkeypatch: pytest.MonkeyPatch, lib_dirs, files_by_dir):
    def fake_glob(pattern, recursive=False):
        if pattern.endswith("lib"):  # the sys.prefix nvidia dir discovery
            return list(lib_dirs)
        return list(files_by_dir.get(Path(pattern).parent, []))

    monkeypatch.setattr(embedder_mod.glob, "glob", fake_glob)


def test_preload_cuda_libs_fixpoint_and_ld_library_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    lib_dir = tmp_path / "lib" / "py3.10" / "nvidia" / "cublas" / "lib"
    lib_dir.mkdir(parents=True)
    so = lib_dir / "libcublas.so.12"
    so.touch()
    _fake_glob(monkeypatch, [str(lib_dir)], {lib_dir: [str(so)]})

    loaded: list[tuple[str, int]] = []
    monkeypatch.setattr(
        embedder_mod.ctypes, "CDLL", lambda path, mode: loaded.append((path, mode))
    )
    monkeypatch.setattr(embedder_mod, "_cuda_libs_preloaded", False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    embedder_mod._preload_cuda_libs()

    assert loaded == [(str(so), ctypes.RTLD_GLOBAL)]
    assert os.environ["LD_LIBRARY_PATH"] == str(lib_dir)  # child processes only

    # idempotent per process: the second call returns before any globbing

    glob_calls: list[str] = []

    def counting_glob(pattern, recursive=False):
        glob_calls.append(pattern)
        return []

    monkeypatch.setattr(embedder_mod.glob, "glob", counting_glob)
    embedder_mod._preload_cuda_libs()
    assert glob_calls == []


def test_preload_cuda_libs_none_loadable_warns_and_skips_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path, caplog
):
    lib_dir = tmp_path / "lib" / "py3.10" / "nvidia" / "cublas" / "lib"
    lib_dir.mkdir(parents=True)
    so = lib_dir / "libcublas.so.12"
    so.touch()
    _fake_glob(monkeypatch, [str(lib_dir)], {lib_dir: [str(so)]})

    def fail_cdll(path, mode):
        raise OSError("cannot open shared object file")

    monkeypatch.setattr(embedder_mod.ctypes, "CDLL", fail_cdll)
    monkeypatch.setattr(embedder_mod, "_cuda_libs_preloaded", False)
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)

    with caplog.at_level(logging.WARNING, logger="ctx_semantic.embedder"):
        embedder_mod._preload_cuda_libs()

    assert "no nvidia wheel libs" in caplog.text
    assert "LD_LIBRARY_PATH" not in os.environ  # early return wrote no env
