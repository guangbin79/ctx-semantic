#!/bin/sh
# Entry point for the ctx-semantic MCP sidecar server.
CTX_DIR=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
# Inference profile: cpu (default) or cuda. The embedder preloads the nvidia
# wheel libs in-process (ctypes RTLD_GLOBAL, ctx_semantic.embedder) before
# importing llama_cpp, so no loader-path export is needed in either profile.
exec uv run --project "$CTX_DIR" --extra "${CTX_SEMANTIC_PROFILE:-cpu}" python -m ctx_semantic.server
