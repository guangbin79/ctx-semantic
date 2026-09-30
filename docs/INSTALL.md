# Installation Reference — ctx-semantic

Fixed, copy-paste-able install guide for a new system. For architecture and
design records see `docs/agents/knowledge/`; this file only covers getting a
working build.

Two inference profiles share one code path and one GGUF model:

- **cpu (default)** — abetlen's prebuilt ~24 MB cpu wheel. No NVIDIA GPU, no
  CUDA runtime libs, no cmake toolchain; installs in well under a minute on a
  normal pipe and imports with no loader-path setup.
- **cuda (opt-in)** — the pinned 1.7 GB cu124 wheel, materialized by the
  script (16-way ranged fetch + sha256) and repaired on-host. Needs an NVIDIA
  GPU (~2 GB VRAM) and a C++ build chain.

Switching profiles = re-running `scripts/uv-sync.sh` with the other argument.

## Prerequisites

| Requirement | Profile | Notes |
|---|---|---|
| Linux x86_64, Python ≥ 3.10 | both | venv is created by uv (`uv venv`), system python only bootstraps |
| [uv](https://docs.astral.sh/uv/) | both | package manager; also provides the sdist build env (cuda repair) |
| git, curl, sha256sum, awk, wc | both | used by `scripts/uv-sync.sh` |
| NVIDIA GPU + CUDA 12.x driver | cuda only | model needs ~2 GB VRAM (peak 1966 MiB at batch 32 on a 6 GB card). No GPU / no nvidia-smi → use the cpu profile (slower, correct) |
| C++ build chain: cmake, ninja, gcc/g++ | cuda only | the cu124 wheel's `libggml-cpu` is rebuilt from the sdist on EVERY host (see "Why the repair step" below). The cpu profile needs no compiler at all — the cpu wheel is prebuilt |

Network: hosts that cannot reach pypi.org need `UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple` (or any PyPI mirror) exported even when every wheel is already cached — uv still fetches dependency metadata live, and without a reachable index `uv-sync.sh` fails (observed as F3: syncs with all wheels on disk still died on the metadata fetch).

## Quick start

```sh
git clone git@github.com:guangbin79/ctx-semantic.git   # ANY location works
cd ctx-semantic
./scripts/uv-sync.sh          # cpu profile (default): ~24 MB wheel, no CUDA libs, no cmake
```

GPU opt-in (switch at any time; re-running with the other argument switches):

```sh
./scripts/uv-sync.sh --cuda   # cuda profile: 1.7 GB cu124 wheel + libggml-cpu host repair
```

Then, for either profile, the one manual asset (GGUF, 639 MB — exact commands
in the Assets table below):

```sh
mkdir -p ~/ctx-semantic/models
```

`uv-sync.sh` is the ONLY supported setup entry point. It must run BEFORE any
bare `uv run` / `uv lock` on a fresh clone. No argument = cpu profile:
`uv sync` (base+dev) + a sha256-verified fetch of the cpu wheel from the
`CPU_WHEEL_*` pin (installed imperatively — the abetlen index serves no
digests, so the lock cannot hash-pin it) + an embedded import probe. `--cuda` = the full cuda pipeline:
materialize the cu124 wheel (16-way ranged fetch + sha256 verify) →
`uv sync --extra cuda` → imperative wheel install (the lock records the path
arm as a requires-dist mapping only, so the sync itself never installs it) →
sdist fetch + sha256 verify → rebuild `libggml-cpu` (`GGML_CUDA=OFF`) →
atomic swap → import probe. Second cuda run hits the "already host-built"
marker fast path.

## Assets (gitignored — bring your own)

| Asset | Location | How to obtain |
|---|---|---|
| cpu wheel (24 MB) | `models/wheels/` (gitignored; verified fetch) | sha256-verified fetch + imperative install — `uv sync` covers base+dev only. Pin (source of truth: `CPU_WHEEL_*` constants in `scripts/uv-sync.sh`): `https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl` · sha256 `d172f3d3c8cdd194c3c47c71cb077ed6e61354a2d0f939ceeac0c8fd29999596` · 23,912,624 bytes |
| cu124 wheel (1.7 GB) | `<repo>/models/wheels/` | auto-fetched + sha256-verified by `uv-sync.sh --cuda` |
| sdist tarball | `<repo>/models/wheels/` | auto-fetched + sha256-verified by `uv-sync.sh --cuda` |
| GGUF model (639 MB) | `~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf` (exact filename — the loader opens it by name) | one-time fetch, both profiles, see snippet below. sha256 pinned in-repo: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| vector store | `~/ctx-semantic/data/vectors.db` | optional; created on demand, deletion = safe full rebuild |

Note the split: the cu124 wheel is per-checkout (repo-relative pin), the GGUF
is home-anchored (`~/ctx-semantic/models`, shared across checkouts — matches
`embedder.DEFAULT_CACHE_DIR`). The cpu wheel is never materialized locally —
uv resolves it straight from the index.

Fetching the GGUF (the HF repo file name is mixed-case and already matches
the local target — the `mv` only moves it into `~/ctx-semantic/models`, no
rename; a lowercase filename from older instructions 404s):

```sh
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Qwen/Qwen3-Embedding-0.6B-GGUF Qwen3-Embedding-0.6B-Q8_0.gguf --local-dir /tmp/gguf
mv /tmp/gguf/Qwen3-Embedding-0.6B-Q8_0.gguf ~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf
echo "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439  $HOME/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf" | sha256sum -c
```

(direct huggingface.co works too when the mirror is unnecessary; `HF_ENDPOINT` and no `all_proxy` is the verified combo on this network)

## Why the repair step runs on every host (cuda profile only)

The pinned cu124 wheel's `libggml-cpu.so` was compiled `-march native` on
Zen4 CI runners and executes AVX-VNNI — SIGILL on e.g. Tiger Lake. The script
rebuilds that one library from the same-version sdist on the local CPU and
swaps it in atomically (`cp` to temp + `mv`). Harmless where the stock lib
would work, mandatory where it would not. A full CUDA source build is not
possible via pip (nvcc wheels ship only ptxas + headers). The cpu wheel has
no such problem — no repair, no compiler.

## Verification (copy-paste; no loader-path setup in either profile)

```sh
# FTS5 available in the venv
uv run --extra cpu python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)')"

# llama-cpp-python pinned version — plain import, the cpu wheel has no CUDA deps
uv run --extra cpu python -c "import llama_cpp; print(llama_cpp.__version__)"        # -> 0.3.35

# model loads and embeds on CPU
uv run --extra cpu python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)"                                              # -> ...-gguf-q8 cpu
```

cuda profile (after `./scripts/uv-sync.sh --cuda`) — the cu124 wheel dlopens
`libcudart.so.12` at import, but the embedder preloads the nvidia-wheel libs
in-process (ctypes `RTLD_GLOBAL`) first, so the same probes run with no env
prefix:

```sh
uv run --extra cuda python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)"                                              # -> ...-gguf-q8 cuda

uv run --extra cuda pytest -q                                             # full suite, cuda assertions live
```

(On a cuda install, a bare `import llama_cpp` with no ctx_semantic import
first fails — the wheel needs the CUDA libs and only the embedder preloads
them in-process. That is the wheel's architecture, not a config error; the
script's own cuda probe uses the same preload. The cpu wheel has no such
dependency.)

## Registering as an opencode MCP server

`~/.config/opencode/opencode.json` (per-host, manual). Default = cpu profile:

```json
{
  "mcp": {
    "ctx-semantic": {
      "type": "local",
      "command": ["/path/to/ctx-semantic/run.sh"]
    }
  }
}
```

GPU opt-in: install the cuda profile first (`./scripts/uv-sync.sh --cuda`),
then pass the profile through the `environment` block — `run.sh` reads
`CTX_SEMANTIC_PROFILE` (default `cpu`):

```json
{
  "mcp": {
    "ctx-semantic": {
      "type": "local",
      "command": ["/path/to/ctx-semantic/run.sh"],
      "environment": {
        "CTX_SEMANTIC_PROFILE": "cuda"
      }
    }
  }
}
```

## Daily operations

```sh
./run.sh                                       # MCP stdio server (spawned by opencode); CTX_SEMANTIC_PROFILE=cuda ./run.sh for GPU
uv run --extra cpu python -m ctx_semantic.warmup --all     # pre-embed every content DB (per-DB fault tolerant)
uv run --extra cpu python -m ctx_semantic.warmup --prune   # drop dead-path rows + retired model keys
uv run --extra cpu python -m ctx_semantic.harness          # recall gates; exit 2 = corpus churned, re-sample CASES
```

(`--extra cpu` matches the default install; on a cuda-profile install swap in
`--extra cuda`. Any exact sync — a bare `uv run`/`uv sync`, or an `--extra`
selection different from the synced state — strips the extra-installed cuda
wheel: always re-run `scripts/uv-sync.sh` to switch profiles. `run.sh`'s
cuda arm boots with `--no-sync` and fails fast when the wheel is missing.)

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv lock`/`uv run`: "path … does not exist" | cu124 wheel not materialized yet — run `scripts/uv-sync.sh --cuda` first (the cpu profile never needs it) |
| First query on a cuda install fails loud with a dlopen traceback (`libcudart.so.12` / `libcublas*`) | the embedder's ctypes preload found the nvidia wheels missing/broken (e.g. a bare `uv sync` stripped them). Diagnose: `uv run --extra cuda python -c "from ctx_semantic.embedder import _find_nvidia_libs; print(_find_nvidia_libs())"` — any `None` value means that wheel lib is gone → re-run `scripts/uv-sync.sh --cuda`. Cannot happen on a cpu install: there the same failed preload only logs a warning and inference runs on CPU |
| Server boots, then the FIRST query fails `ModuleNotFoundError: llama_cpp` (cuda profile) | the cuda wheel was stripped by a re-sync (bare `uv run`/`uv sync` or a different `--extra` on this venv). `run.sh`'s cuda arm now fails fast at boot on the missing wheel; older entrypoints surface it at first query (lazy import) — re-run `scripts/uv-sync.sh --cuda` |
| "preload failed" warning in server logs on the cpu profile | expected, not an error — no nvidia wheels exist on that profile; inference is on CPU |
| SIGILL in libggml-cpu | cuda profile: repair step was skipped/bypassed (bare `uv sync` relinks the wheel's stock lib) — re-run `scripts/uv-sync.sh --cuda` |
| `FAIL <db>: SchemaDrift …` from warmup --all | a content DB from an incompatible context-mode version; others still warm, exit 1 |
| harness exits 2 (gold drift) | live corpus re-indexed since CASES were authored — re-sample per `ctx_semantic/harness.py` header |
| silent "(no results)" for a symlinked project | projhash realpath divergence — see `ctx_semantic/projhash.py` docstring; pin via `$CTX_SEMANTIC_DB` |

## Version bump contract

On a llama-cpp-python bump, the pin source of truth is the constant blocks in
`scripts/uv-sync.sh`: `CPU_WHEEL_URL`/`_SHA256`/`_BYTES` (cpu profile) and
`WHEEL_URL`/`_SHA256` + `SDIST_SHA256` (cuda profile). Move BOTH blocks
together with the `==0.3.35` version pins in `pyproject.toml`
`[project.optional-dependencies]`, the version-embedded path in
`[tool.uv.sources]`, and `uv.lock` (`uv lock` after editing). The cpu
constants drive the cpu wheel's verified fetch + install (the index serves
no digests, so the pin is enforced by the script, not the lock); the cuda
constants drive the ranged fetch + sha256 verify exactly as before.
