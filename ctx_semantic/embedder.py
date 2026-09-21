"""ctx_semantic.embedder — jina-v2-base-zh embedding layer (T4).

Model: jinaai/jina-embeddings-v2-base-zh, dim 768 — pinned in
~/.omo/evidence/ctx-semantic/task-1-scaffold.md (the plan's original "1024"
is obsolete). zh/en bilingual, which is why the bilingual cosine sanity test
must hold.

Embedding input spec:
- document side: title + "\\n\\n" + content, assembled by the caller; this
  layer embeds whatever string it receives.
- query side: raw query text. jina v2 takes NO instruction/passage prefix
  (unlike bge/e5 families); adding one would corrupt the embedding.

Truncation: fastembed enables tokenizer truncation at tokenizer_config.json's
model_max_length = 512 tokens for this artifact — NOT the model's 8192-position
capacity (config.json max_position_embeddings). Measured in
~/.omo/evidence/ctx-semantic/task-4-embedder.md: the tokenizer clamps a
1792-char text to exactly 512 tokens (len(ids)==512), and prefixes that cover
the first 512 tokens embed identically (cos = 1.000000 at >= 1200 chars; < 1.0
at 800). Content beyond the first 512 tokens is silently dropped.

GPU (the T1 trap): onnxruntime-gpu dlopens libonnxruntime_providers_cuda.so,
whose CUDA deps (libcublasLt.so.12, ...) live under
.venv/lib/**/nvidia/*/lib — outside the system loader path. glibc reads
LD_LIBRARY_PATH at process startup ONLY: mutating os.environ in-process is a
no-op for dlopen (verified in T4 exploration). Fix: ctypes RTLD_GLOBAL preload
of those libs before session creation; the loader then resolves DT_NEEDED
SONAMEs against already-loaded objects. Actual GPU use is verified via
nvidia-smi PID lookup, never provider names — a CUDA session that silently
fell back to CPU is treated as a load failure and retried on CPU.
"""

from __future__ import annotations

import ctypes
import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

MODEL_NAME: Final = "jinaai/jina-embeddings-v2-base-zh"
DIM: Final = 768  # task-1-scaffold.md; re-asserted from live output at load
DEFAULT_CACHE_DIR: Final = Path.home() / "ctx-semantic" / "models"
# fastembed default batch 256 needs a ~3 GiB fused-MatMul workspace -> OOM
# on a 6 GB card; 32 fits. Knob for embed(), see embed_batch.
EMBED_BATCH_SIZE: Final = 32

_cuda_libs_preloaded = False


def _apply_hf_env() -> None:
    """Pin task-1's verified hub combo before fastembed touches the network.

    The model ships from the local cache; when fastembed still resolves hub
    metadata remotely, only HF_ENDPOINT=hf-mirror + xet off + proxies off was
    verified working on this host (direct hf-mirror ~2 MB/s; proxied
    huggingface.co ~110 KB/s; xet 401s against the mirror). Idempotent —
    explicit HF_HUB_DISABLE_XET/HF_ENDPOINT values win via setdefault, proxy
    vars are dropped because the working combo is direct.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    for var in (
        "all_proxy", "ALL_PROXY", "http_proxy", "https_proxy",
        "HTTP_PROXY", "HTTPS_PROXY",
    ):
        os.environ.pop(var, None)


_apply_hf_env()


def _preload_cuda_libs() -> None:
    """RTLD_GLOBAL-load the nvidia wheel libs so the CUDA EP's dlopen resolves.

    Idempotent per process. Programmatic equivalent of run.sh's LD_LIBRARY_PATH
    discovery — necessary because the dynamic loader ignores post-startup
    os.environ changes (see module docstring).
    """
    global _cuda_libs_preloaded
    if _cuda_libs_preloaded:
        return
    _cuda_libs_preloaded = True
    pattern = str(Path(sys.prefix) / "lib" / "**" / "nvidia" / "*" / "lib")
    lib_dirs = sorted(glob.glob(pattern, recursive=True))
    loaded: set[str] = set()
    # Repeat to fixpoint: a later pass satisfies deps that failed in earlier ones.
    for _ in range(4):
        newly_loaded = 0
        for lib_dir in lib_dirs:
            for so in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                if so in loaded:
                    continue
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    continue
                loaded.add(so)
                newly_loaded += 1
        if not newly_loaded:
            break
    if not loaded:
        logger.warning("no nvidia wheel libs under %s — CUDA EP cannot load", pattern)
        return
    # For child processes only; this process's loader has already started.
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        [*lib_dirs, os.environ.get("LD_LIBRARY_PATH", "")]
    ).rstrip(":")


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
    """Cast to float32 and L2-normalize rows (fastembed emits float64, already
    near-unit norms; the cast can perturb them by ~1e-7)."""
    m = mat.astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


class Embedder:
    """Lazy-loading jina-v2-base-zh embedder with deterministic device fallback.

    Load flow: preload CUDA libs -> CUDA-session attempt -> probe embed ->
    nvidia-smi PID check. Any failure (EP unavailable, lib load error, OOM,
    silent CPU fallback) retries once on CPU and is logged; `device` then
    reports "cpu". Both devices failing raises the underlying error.
    """

    def __init__(self, cache_dir: str | os.PathLike[str] = DEFAULT_CACHE_DIR) -> None:
        self.cache_dir = Path(cache_dir)
        self._device: str | None = None
        self._model = None

    @property
    def device(self) -> str | None:
        """Device actually in use: "cuda", "cpu", or None before first load."""
        return self._device

    def _new_model(self, providers: list[str]):
        from fastembed import TextEmbedding  # imported after _preload_cuda_libs()

        return TextEmbedding(
            model_name=MODEL_NAME,
            cache_dir=str(self.cache_dir),
            providers=providers,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            _preload_cuda_libs()
            model = self._new_model(["CUDAExecutionProvider", "CPUExecutionProvider"])
            probe = np.stack(list(model.embed(["device probe"])))
            vram = _pid_vram_on_gpu(os.getpid())
            if vram is None:
                raise RuntimeError(
                    "CUDA session created but nvidia-smi shows no VRAM for this "
                    "pid — silent CPU fallback (T1 trap), refusing to claim GPU"
                )
            self._device = "cuda"
            logger.info("embedder on GPU: pid=%s vram=%s", os.getpid(), vram)
        except Exception as exc:  # noqa: BLE001 — spec: ANY CUDA failure (OOM, EP, lib load) falls back
            logger.warning(
                "CUDA init failed (%s: %s) — falling back to CPU embedder",
                type(exc).__name__,
                exc,
            )
            model = self._new_model(["CPUExecutionProvider"])
            probe = np.stack(list(model.embed(["device probe"])))
            self._device = "cpu"
        assert probe.shape == (1, DIM), (
            f"{MODEL_NAME} embedding shape {probe.shape}, want (1, {DIM}) "
            "(task-1-scaffold.md pinned dim=768)"
        )
        self._model = model

    def _rebuild_on_cpu(self, exc: Exception) -> None:
        logger.warning(
            "GPU embed failed (%s: %s) — rebuilding embedder on CPU",
            type(exc).__name__,
            exc,
        )
        self._model = self._new_model(["CPUExecutionProvider"])
        self._device = "cpu"

    def embed_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed documents -> (n, 768) float32, rows L2-normalized.

        Callers pass title + "\\n\\n" + content per document (see module
        docstring). A CUDA-time failure (e.g. OOM) rebuilds on CPU once and
        retries; the fallback is logged and `device` flips to "cpu".
        """
        self._ensure_loaded()
        if not texts:
            return np.empty((0, DIM), dtype=np.float32)
        # ponytail: small in-process batches (EMBED_BATCH_SIZE) — the default
        # 256 needs a ~3 GiB fused-MatMul workspace that OOMs a 6 GB card;
        # parallel must stay None (0 means all-cores -> per-core worker
        # processes each opening a CUDA session = guaranteed OOM).
        kwargs = {"batch_size": EMBED_BATCH_SIZE}
        try:
            mat = np.stack(list(self._model.embed(texts, **kwargs)))
        except Exception as exc:
            if self._device != "cuda":
                raise
            self._rebuild_on_cpu(exc)
            mat = np.stack(list(self._model.embed(texts, **kwargs)))
        return _l2_normalized(mat)

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed one raw query (no prefix) -> (768,) float32, unit norm."""
        return self.embed_batch([text])[0]
