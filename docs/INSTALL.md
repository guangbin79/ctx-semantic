# Installation Reference — ctx-semantic

Fixed, copy-paste-able install guide for a new system. For architecture and
design records see `docs/agents/knowledge/`; this file only covers getting a
working build.

## Prerequisites

| Requirement | Notes |
|---|---|
| Linux x86_64, Python ≥ 3.10 | venv is created by uv (`uv venv`), system python only bootstraps |
| [uv](https://docs.astral.sh/uv/) | package manager; also provides the sdist build env |
| git, curl, sha256sum, awk, wc | used by `scripts/uv-sync.sh` |
| C++ build chain: cmake, ninja, gcc/g++ | required — the wheel's `libggml-cpu` is rebuilt from the sdist on EVERY host (see "Why the repair step" below) |
| NVIDIA GPU + CUDA 12.x driver | optional but expected; model needs ~2 GB VRAM (peak 1966 MiB at batch 32 on a 6 GB card). No GPU / no nvidia-smi → automatic CPU fallback (slower, correct) |

## Quick start

```sh
git clone git@github.com:guangbin79/ctx-semantic.git   # ANY location works
cd ctx-semantic
./scripts/uv-sync.sh          # wheel fetch + sha256 verify + uv sync + libggml-cpu repair
mkdir -p ~/ctx-semantic/models
cp <from-old-host>/Qwen3-Embedding-0.6B-Q8_0.gguf ~/ctx-semantic/models/
```

`uv-sync.sh` is the ONLY supported setup entry point. It must run BEFORE any
bare `uv run` / `uv lock`: the `[tool.uv.sources]` wheel pin is repo-relative
(`models/wheels/…`, gitignored) and only exists after the script materializes
it. Order inside the script: derive wheel path + version from the pyproject
pin → basename cross-check vs the pinned URL → fetch (16-way ranged, sha256)
→ `uv sync` → sdist fetch + sha256 verify → rebuild `libggml-cpu`
(`GGML_CUDA=OFF`) → atomic swap. Second run hits the "already host-built"
marker fast path.

## Assets (gitignored — bring your own)

| Asset | Location | How to obtain |
|---|---|---|
| cu124 wheel (1.7 GB) | `<repo>/models/wheels/` | auto-fetched + sha256-verified by `uv-sync.sh` |
| sdist tarball | `<repo>/models/wheels/` | auto-fetched + sha256-verified by `uv-sync.sh` |
| GGUF model (639 MB) | `~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf` (exact filename — the loader opens it by name) | one-time fetch, see snippet below. sha256 pinned in-repo: `06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439` |
| vector store | `~/ctx-semantic/data/vectors.db` | optional; created on demand, deletion = safe full rebuild |

Note the split: the wheel is per-checkout (repo-relative pin), the GGUF is
home-anchored (`~/ctx-semantic/models`, shared across checkouts — matches
`embedder.DEFAULT_CACHE_DIR`).

Fetching the GGUF (HF repo file is lowercase — rename to the exact name in the table):

```sh
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
    Qwen/Qwen3-Embedding-0.6B-GGUF qwen3-embedding-0.6b-q8_0.gguf --local-dir /tmp/gguf
mv /tmp/gguf/qwen3-embedding-0.6b-q8_0.gguf ~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf
echo "06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439  ~/ctx-semantic/models/Qwen3-Embedding-0.6B-Q8_0.gguf" | sha256sum -c
```

(direct huggingface.co works too when the mirror is unnecessary; `HF_ENDPOINT` and no `all_proxy` is the verified combo on this network)

## Why the repair step runs on every host

The pinned cu124 wheel's `libggml-cpu.so` was compiled `-march native` on
Zen4 CI runners and executes AVX-VNNI — SIGILL on e.g. Tiger Lake. The script
rebuilds that one library from the same-version sdist on the local CPU and
swaps it in atomically (`cp` to temp + `mv`). Harmless where the stock lib
would work, mandatory where it would not. A full CUDA source build is not
possible via pip (nvcc wheels ship only ptxas + headers).

## Verification (copy-paste; GPU probes need the LD prefix)

```sh
# FTS5 available in the venv
uv run python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)')"

# llama-cpp-python pinned version
LD_LIBRARY_PATH="$(find .venv/lib -type d -name lib -path '*nvidia/*' | tr '\n' ':')" \
uv run python -c "import llama_cpp; print(llama_cpp.__version__)"        # -> 0.3.35

# model loads and embeds on GPU
LD_LIBRARY_PATH="$(find .venv/lib -type d -name lib -path '*nvidia/*' | tr '\n' ':')" \
uv run python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)"                                              # -> qwen/qwen3-embedding-0.6b-gguf-q8 cuda

uv run pytest -q                                                          # 157 passed (with LD) / 152+5 skip (without)
```

Even `import llama_cpp` needs the nvidia wheel lib dirs on `LD_LIBRARY_PATH`
(it dlopens `libllama.so` → `libcudart.so.12`). `run.sh` exports this
automatically and resolves its `CTX_DIR` script-relative, so it works from any
checkout (including worktrees).

## Registering as an opencode MCP server

`~/.config/opencode/opencode.json` (per-host, manual):

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

## Daily operations

```sh
./run.sh                                       # MCP stdio server (spawned by opencode)
uv run python -m ctx_semantic.warmup --all     # pre-embed every content DB (per-DB fault tolerant)
uv run python -m ctx_semantic.warmup --prune   # drop dead-path rows + retired model keys
uv run python -m ctx_semantic.harness          # recall gates; exit 2 = corpus churned, re-sample CASES
```

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv lock`/`uv run`: "path … does not exist" | wheel not materialized yet — run `scripts/uv-sync.sh` first |
| `libcudart.so.12: cannot open shared object file` | missing `LD_LIBRARY_PATH` nvidia lib dirs — use `run.sh`, or prefix as in the probes |
| SIGILL in libggml-cpu | repair step was skipped/bypassed (bare `uv sync` relinks the wheel's stock lib) — re-run `scripts/uv-sync.sh` |
| `FAIL <db>: SchemaDrift …` from warmup --all | a content DB from an incompatible context-mode version; others still warm, exit 1 |
| harness exits 2 (gold drift) | live corpus re-indexed since CASES were authored — re-sample per `ctx_semantic/harness.py` header |
| silent "(no results)" for a symlinked project | projhash realpath divergence — see `ctx_semantic/projhash.py` docstring; pin via `$CTX_SEMANTIC_DB` |

## Version bump contract

On a llama-cpp-python bump, three places move together (enforced fail-fast by
the script's basename cross-check where possible): `WHEEL_URL` +
`WHEEL_SHA256` + `SDIST_SHA256` in `scripts/uv-sync.sh`, the
`[tool.uv.sources]` path pin in `pyproject.toml`, and `uv.lock`
(`uv lock` after editing the pin).
