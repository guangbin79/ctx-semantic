"""ctx_semantic.embedder — Qwen3-Embedding-0.6B GGUF embedding layer (T2).

Model: qwen/qwen3-embedding-0.6b-gguf-q8 (MODEL_NAME is the vectors-store
key), the Qwen3-Embedding-0.6B Q8_0 GGUF served by llama-cpp-python, dim
1024, pooling LAST (pinned by the GGUF's qwen3.pooling_type metadata —
verified in ~/.omo/evidence/ctx-semantic/spike-qwen3-llama-cpp.out).

Embedding input contract:
- document side: title + "\\n" + content, assembled by the caller; this
  layer embeds whatever raw string it receives. Qwen3 takes NO instruction
  prefix on the document side.
- query side: embed_query applies the Instruct prefix internally via
  _format_query — callers always pass the raw query text.

Truncation: llama.cpp truncates each input at n_ctx = 8192 tokens; the
corpus p99 is 2146 tokens (spike phase 8), so only extreme outliers are
clipped.

GPU: llama-cpp-python ships its own CUDA backend (n_gpu_layers=-1) — no
provider libraries to preload. Actual GPU use is verified via nvidia-smi
PID lookup, never device claims — a CUDA session that silently fell back
to CPU is treated as a load failure and retried on CPU (n_gpu_layers=0).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

MODEL_NAME: Final = "qwen/qwen3-embedding-0.6b-gguf-q8"  # vectors-store key
DIM: Final = 1024  # re-asserted from live output at load
DEFAULT_CACHE_DIR: Final = Path.home() / "ctx-semantic" / "models"
GGUF_PATH: Final = DEFAULT_CACHE_DIR / "Qwen3-Embedding-0.6B-Q8_0.gguf"
QUERY_INSTRUCT: Final = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
# T1 spike phase 9: peak pid VRAM 1966 MiB at one embed() of 32 docs on the
# 6 GB card (budget 5120 MiB); llama.cpp's constraint is KV capacity per
# batch, not a MatMul workspace. Knob for embed_batch sub-batching.
EMBED_BATCH_SIZE: Final = 32
N_CTX: Final = 8192


def _format_query(text: str) -> str:
    """Apply Qwen3's query-side Instruct prefix (documents stay raw)."""
    return f"Instruct: {QUERY_INSTRUCT}\nQuery: {text}"


def _pid_vram_on_gpu(pid: int) -> str | None:
    """VRAM usage nvidia-smi reports for pid, or None if pid is not on the GPU."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("nvidia-smi query failed: %s", exc)
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0] == str(pid):
            return parts[1] if len(parts) > 1 else "unknown"
    return None


def _l2_normalized(mat: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Cast to float32 and L2-normalize rows (raw embeddings arrive float64
    and near-unit norm; the cast can perturb them by ~1e-7)."""
    m = mat.astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class Embedder:
    """Lazy-loading qwen3-embedding-0.6b embedder with deterministic device
    fallback.

    Load flow: CUDA attempt (n_gpu_layers=-1) -> probe embed -> nvidia-smi
    PID check. Any failure (backend unavailable, lib load error, OOM, silent
    CPU fallback) retries once on CPU and is logged; `device` then reports
    "cpu". Both devices failing raises the underlying error.
    """

    def __init__(self, cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self._device: str | None = None
        self._model = None

    @property
    def device(self) -> str | None:
        """Device actually in use: "cuda", "cpu", or None before first load."""
        return self._device

    def _new_model(self, use_gpu: bool):
        # lazy: absent from the lock until T5 — a top-level import would
        # ImportError every uv run before that todo lands.
        import llama_cpp

        return llama_cpp.Llama(
            model_path=str(GGUF_PATH),
            embedding=True,
            n_gpu_layers=-1 if use_gpu else 0,
            pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_ctx=N_CTX,
            verbose=False,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            model = self._new_model(True)
            probe = np.stack(model.embed(["device probe"]))
            vram = _pid_vram_on_gpu(os.getpid())
            if vram is None:
                raise RuntimeError(
                    "CUDA session created but nvidia-smi shows no VRAM for this "
                    "pid — silent CPU fallback, refusing to claim GPU"
                )
            self._device = "cuda"
            logger.info("embedder on GPU: pid=%s vram=%s", os.getpid(), vram)
        except Exception as exc:  # noqa: BLE001 — spec: ANY CUDA failure (OOM, backend, lib load) falls back
            logger.warning(
                "CUDA init failed (%s: %s) — falling back to CPU embedder",
                type(exc).__name__,
                exc,
            )
            model = self._new_model(False)
            probe = np.stack(model.embed(["device probe"]))
            self._device = "cpu"
        assert probe.shape == (1, DIM), (
            f"{MODEL_NAME} embedding shape {probe.shape}, want (1, {DIM})"
        )
        self._model = model

    def _rebuild_on_cpu(self, exc: Exception) -> None:
        logger.warning(
            "GPU embed failed (%s: %s) — rebuilding embedder on CPU",
            type(exc).__name__,
            exc,
        )
        self._model = self._new_model(False)
        self._device = "cpu"

    def _embed_all(self, texts: list[str]) -> npt.NDArray[np.floating]:
        mats = [
            np.stack(self._model.embed(texts[i : i + EMBED_BATCH_SIZE]))
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]
        return np.concatenate(mats)

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed raw documents -> (n, 1024) float32, rows L2-normalized.

        Callers pass title + "\\n" + content per document; no prefix is
        applied on the doc side. A CUDA-time failure (e.g. OOM) rebuilds on
        CPU once and retries; the fallback is logged and `device` flips to
        "cpu".
        """
        if not texts:
            return np.empty((0, DIM), dtype=np.float32)
        self._ensure_loaded()
        try:
            mat = self._embed_all(texts)
        except Exception as exc:
            if self._device != "cuda":
                raise
            self._rebuild_on_cpu(exc)
            mat = self._embed_all(texts)
        return _l2_normalized(mat)

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed one query -> (1024,) float32, unit norm.

        The Qwen3 Instruct prefix is applied internally via _format_query;
        callers pass the raw query text.
        """
        return self.embed_batch([_format_query(text)])[0]
