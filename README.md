# ctx-semantic

Hybrid semantic recall sidecar for the context-mode knowledge base.

context-mode already stores session knowledge (decisions, errors, plans, notes)
in SQLite with FTS5 full-text search. This sidecar adds a semantic layer:
documents are embedded with llama-cpp-python (Qwen3-Embedding-0.6B GGUF
Q8_0, dim 1024, CUDA — see docs/agents/knowledge), vectors are kept
alongside the FTS5 store, and
an MCP server (python `mcp` SDK) exposes hybrid recall — exact keyword match
merged with dense nearest-neighbor — to coding agents. It runs as a plain
subprocess (`run.sh`), no daemon.

## Setup

```sh
cd ~/ctx-semantic
./scripts/uv-sync.sh       # uv sync + llama-cpp wheel fetch + host CPU-lib repair
```

Entry point: `./run.sh` (executes `python -m ctx_semantic.server` inside the
uv environment).

## Verification probes

```sh
# FTS5 available in the venv
uv run python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)')"

# llama-cpp-python pinned version (import dlopens the CUDA libs — needs the
# nvidia lib dirs on LD_LIBRARY_PATH, like the probe below)
LD_LIBRARY_PATH="$(find .venv/lib -type d -name lib -path '*nvidia/*' | tr '\n' ':')" \
uv run python -c "import llama_cpp; print(llama_cpp.__version__)"

# embedding model loads and embeds on GPU (nvidia wheel libs on LD_LIBRARY_PATH)
LD_LIBRARY_PATH="$(find .venv/lib -type d -name lib -path '*nvidia/*' | tr '\n' ':')" \
uv run python -c "
from ctx_semantic.embedder import Embedder, MODEL_NAME
e = Embedder(); e.embed_query('probe')
print(MODEL_NAME, e.device)
"
```

## Environment notes

- GPU inference needs the nvidia-wheel lib dirs on LD_LIBRARY_PATH — `run.sh`
  sets this up automatically.


## Testing

```sh
uv run pytest -q                # plain run
uv run pytest --cov=ctx_semantic --cov-report=term-missing
```
