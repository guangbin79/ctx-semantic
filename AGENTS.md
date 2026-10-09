# AGENTS.md — ctx-semantic agent constraints

## Forbidden
- `uv sync`, `uv add`, `uv remove`, and bare `uv run` (without `--no-sync`): every
  implicit exact sync strips the manually installed CUDA wheel from `.venv`.

## Environment changes — single entry point
- `./scripts/uv-sync.sh` (cpu, default) · `./scripts/uv-sync.sh --cuda` (GPU)

## Boot failure "cuda wheel is missing"
- Re-run `./scripts/uv-sync.sh --cuda`, then start again.

## Why
- The cu124 wheel cannot be expressed in `uv.lock` — it can only be re-installed
  by hand after each sync, which is what `uv-sync.sh` automates.
