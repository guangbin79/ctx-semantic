# ctx-semantic

Hybrid semantic recall sidecar for the context-mode knowledge base.

context-mode already stores session knowledge (decisions, errors, plans, notes)
in SQLite with FTS5 full-text search. This sidecar adds a semantic layer:
documents are embedded with fastembed (jinaai/jina-embeddings-v2-base-zh,
dim 768, mixed Chinese-English — see docs/evidence; bge-m3 is not in
fastembed's supported list), vectors are kept alongside the FTS5 store, and
an MCP server (python `mcp` SDK) exposes hybrid recall — exact keyword match
merged with dense nearest-neighbor — to coding agents. It runs as a plain
subprocess (`run.sh`), no daemon.

## Setup

```sh
cd ~/ctx-semantic
uv sync --locked          # create .venv from uv.lock
```

Entry point: `./run.sh` (executes `python -m ctx_semantic.server` inside the
uv environment).

## Verification probes

```sh
# FTS5 available in the venv
uv run python -c "import sqlite3; con=sqlite3.connect(':memory:'); con.execute('CREATE VIRTUAL TABLE t USING fts5(x)')"

# embedding model loads and embeds (add LD_LIBRARY_PATH for GPU)
LD_LIBRARY_PATH="$(find .venv/lib -type d -name lib -path '*nvidia/*' | tr '\n' ':')" \
uv run python -c "
from fastembed import TextEmbedding
m = TextEmbedding('jinaai/jina-embeddings-v2-base-zh', cache_dir='models', providers=['CUDAExecutionProvider'])
print(next(iter(m.embed(['ping', 'pong']))).shape)
"
# onnxruntime execution providers (GPU vs CPU)
uv run python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
```

## Environment notes

- HF downloads in this network need: `HF_ENDPOINT=https://hf-mirror.com`,
  `HF_HUB_DISABLE_XET=1` (mirror cannot serve the Xet CAS protocol), and no
  `all_proxy` (its `socks://` scheme is rejected by httpx; `https_proxy=http://…` is fine).
- GPU inference needs the nvidia-wheel lib dirs on LD_LIBRARY_PATH — `run.sh`
  sets this up automatically.

```

## Testing

```sh
uv run pytest -q                # plain run
uv run pytest --cov=ctx_semantic --cov-report=term-missing
```
