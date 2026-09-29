#!/bin/sh
# Entry point for the ctx-semantic MCP sidecar server.
CTX_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# llama-cpp-python dlopens CUDA runtime/cuBLAS from the nvidia pip wheels; the
# system loader does not see site-packages, so export their lib dirs.
NVLIB=$(find "$CTX_DIR"/.venv/lib -type d -name lib -path '*nvidia/*' 2>/dev/null | paste -sd: -)
LD_LIBRARY_PATH="${NVLIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
exec uv run --project "$CTX_DIR" python -m ctx_semantic.server
