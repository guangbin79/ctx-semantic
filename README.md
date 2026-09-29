# ctx-semantic

Hybrid semantic recall sidecar for the context-mode knowledge base.

context-mode already stores session knowledge (decisions, errors, plans, notes)
in SQLite with FTS5 full-text search. This sidecar adds a semantic layer:
documents are embedded with llama-cpp-python (Qwen3-Embedding-0.6B GGUF
Q8_0, dim 1024 — see docs/agents/knowledge), vectors are kept
alongside the FTS5 store, and
an MCP server (python `mcp` SDK) exposes hybrid recall — exact keyword match
merged with dense nearest-neighbor — to coding agents. It runs as a plain
subprocess (`run.sh`), no daemon.

Two inference profiles, same model and same embeddings either way: **cpu is
the default** (~24 MB wheel, no NVIDIA GPU, no CUDA runtime) and **cuda is
opt-in** (pinned 1.7 GB cu124 wheel + on-host `libggml-cpu` repair, ~2 GB
VRAM).

## Setup

```sh
git clone git@github.com:guangbin79/ctx-semantic.git   # any location
cd ctx-semantic
./scripts/uv-sync.sh       # cpu profile (default): ~24 MB wheel, no CUDA libs, no compiler
# GPU opt-in: ./scripts/uv-sync.sh --cuda
```

Plus the one manual asset (GGUF model) — full prerequisites, asset table,
MCP registration, and troubleshooting: **[docs/INSTALL.md](docs/INSTALL.md)**.

Entry point: `./run.sh` (executes `python -m ctx_semantic.server` inside the
uv environment with `--extra "${CTX_SEMANTIC_PROFILE:-cpu}"`; script-relative,
works from any checkout).

## Verification probes

```sh
# FTS5 available in the venv
uv run --extra cpu python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)')"

# llama-cpp-python pinned version — plain import, no loader-path setup
uv run --extra cpu python -c "import llama_cpp; print(llama_cpp.__version__)"    # -> 0.3.35

# embedding model loads and embeds on CPU
uv run --extra cpu python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)
"                                                                                # -> ... cpu

# cuda profile (after ./scripts/uv-sync.sh --cuda): the embedder preloads the
# nvidia-wheel libs in-process (ctypes RTLD_GLOBAL) before importing
# llama_cpp, so the same probe needs no env prefix
uv run --extra cuda python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)
"                                                                                # -> ... cuda
```

## Drift probe

context-mode upgrades can change the DB layout or project-hash scheme
this sidecar reverse-depends on. Gates: `open_db` fails loud on schema
drift (the error text carries the full fix kit), the server refuses to
start on it (preflight), and an mtime heuristic warns on stale DBs.
Health check from anywhere:

```sh
uv run --project ~/ctx-semantic python -m ctx_semantic.probe [--project DIR]
```

Exit codes: `0` PASS/SKIP · `1` DRIFT (fix kit printed) · `2` STALE.
Run it after every `ctx_upgrade`.

## Environment notes

- The inference profile is picked by `CTX_SEMANTIC_PROFILE` (`cpu` default,
  `cuda` opt-in) and must be installed first: `./scripts/uv-sync.sh`
  (`--cuda` for GPU). Switching profiles = re-running the script with the
  other argument.
- The embedder preloads the nvidia-wheel libs in-process (ctypes
  `RTLD_GLOBAL`) before importing llama_cpp — no loader-path export is used
  or needed in either profile. Preload-failure semantics: on a cpu install
  there are no nvidia wheels, so the preload logs a warning and inference
  runs on CPU (expected, benign); on a cuda install with broken/missing
  nvidia wheels the first query fails loud and diagnosably — see
  docs/INSTALL.md Troubleshooting.

## Testing

```sh
uv run --extra cpu pytest -q                # cpu profile (default install)
uv run --extra cpu pytest --cov=ctx_semantic --cov-report=term-missing
# cuda profile install: swap --extra cpu -> --extra cuda
```
