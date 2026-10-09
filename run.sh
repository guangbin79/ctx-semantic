#!/bin/sh
# Entry point for the ctx-semantic MCP sidecar server.
CTX_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# Inference profile: cpu (default) or cuda. The embedder preloads the nvidia
# wheel libs in-process (ctypes RTLD_GLOBAL, ctx_semantic.embedder) before
# importing llama_cpp, so no loader-path export is needed in either profile.
PROFILE="${CTX_SEMANTIC_PROFILE:-cpu}"
case "$PROFILE" in
    cpu|cuda) ;;
    *) echo "ctx-semantic: CTX_SEMANTIC_PROFILE must be 'cpu' or 'cuda', got '$PROFILE'" >&2; exit 1 ;;
esac
if [ "$PROFILE" = cuda ]; then
    # cuda is script-gated: uv-sync.sh --cuda installs the wheel uv's lock
    # cannot express, so uv run's implicit exact sync would strip it on any
    # re-sync triggered elsewhere (bare `uv run`, a different --extra). Run
    # without sync and fail fast at boot when the cuda wheel is missing.
    # OR semantics: either mirror location satisfies the check (a two-operand
    # ls would AND them — any unmatched glob literal makes ls exit 2).
    ls "$CTX_DIR"/.venv/lib/python3*/site-packages/llama_cpp/lib/libggml-cuda* >/dev/null 2>&1 \
        || ls "$CTX_DIR"/.venv/lib/python3*/site-packages/llama_cpp_python.libs/libggml-cuda* >/dev/null 2>&1 \
        || {
        echo "ctx-semantic: cuda profile set but the cuda wheel is missing — run ./scripts/uv-sync.sh --cuda first" >&2
        exit 1
    }
    exec uv run --no-sync --project "$CTX_DIR" python -m ctx_semantic.server
fi
exec uv run --no-sync --project "$CTX_DIR" --extra cpu python -m ctx_semantic.server
