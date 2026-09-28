#!/bin/sh
# uv sync + host repair for llama-cpp-python's CPU kernels.
#
# Why (T1 spike finding, ~/.omo/evidence/ctx-semantic/spike-qwen3-llama-cpp.out):
# the pinned cu124 prebuilt wheel's libggml-cpu.so was compiled with
# -march native on Zen4 CI runners and executes AVX-VNNI (VEX vpdpbusd) —
# SIGILL on this host's i7-11800H (Tiger Lake). The GPU kernels
# (libggml-cuda.so) are unaffected. Fix: after uv sync, rebuild libggml-cpu
# from the SAME 0.3.35 sdist on THIS host with GGML_CUDA=OFF and swap it in.
# A full CUDA source build is not possible here: pip's nvidia nvcc wheels
# ship only ptxas + headers, no nvcc.
#
# The script also materializes the wheel itself: uv cannot range-read GitHub
# release assets (falls back to streaming the whole 1.7GB single-stream,
# ~0.4MB/s on this host), but curl -r against the same signed-redirect URL
# works, so the fetch is 16 parallel ranged chunks + sha256 verification
# (the hash must equal both the constant below and uv.lock's pin — keep all
# three in sync on version bumps).
set -eu
cd "$(dirname "$0")/.."

WHEEL_URL=https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu124/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl
WHEEL_SHA256=62fc788a4ecee5a579f40708ccfe0c4d566dd5c9959497bdc0544dbba94d82aa
WHEEL_DIR="${CTX_SEMANTIC_MODELS:-$HOME/ctx-semantic}/models/wheels"
WHEEL="$WHEEL_DIR/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl"

mkdir -p "$WHEEL_DIR"
if ! echo "$WHEEL_SHA256  $WHEEL" | sha256sum -c --status 2>/dev/null; then
    echo "fetching wheel (16-way ranged GET): $WHEEL_URL"
    size=$(curl -sIL "$WHEEL_URL" | awk 'tolower($1)=="content-length:"{l=$2} END{print l}' | tr -d "\r")
    [ -n "$size" ] || { echo "FAIL: no content-length from $WHEEL_URL"; exit 1; }
    parts=16
    chunk=$(( (size + parts - 1) / parts ))
    tmp="$WHEEL.part"
    rm -rf "$tmp"; mkdir "$tmp"
    i=0
    while [ "$i" -lt "$parts" ]; do
        start=$(( i * chunk )); end=$(( start + chunk - 1 ))
        [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        curl -sfL -r "$start-$end" -o "$tmp/$i" "$WHEEL_URL" &
        i=$(( i + 1 ))
    done
    wait
    i=0
    while [ "$i" -lt "$parts" ]; do cat "$tmp/$i"; i=$(( i + 1 )); done > "$WHEEL"
    rm -rf "$tmp"
    echo "$WHEEL_SHA256  $WHEEL" | sha256sum -c || exit 1
fi

uv sync "$@"

# --- host repair of libggml-cpu ---
LIB=""
for d in .venv/lib/python3*/site-packages/llama_cpp/lib; do LIB="$d"; done
[ -n "$LIB" ] || { echo "FAIL: llama_cpp not installed in .venv"; exit 1; }
MARKER="$LIB/.host-cpu-lib.sha256"
cur=$(sha256sum "$LIB/libggml-cpu.so" | cut -d" " -f1)
if [ -f "$MARKER" ] && [ "$cur" = "$(head -1 "$MARKER" | cut -d' ' -f1)" ]; then
    echo "libggml-cpu already host-built (sha256 $cur)"
    exit 0
fi

echo "rebuilding libggml-cpu from the 0.3.35 sdist on this host (GGML_CUDA=OFF)..."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
uv venv -q "$T/venv"
CMAKE_ARGS="-DGGML_CUDA=OFF" uv pip install -q --python "$T/venv/bin/python" \
    --no-binary llama-cpp-python "llama-cpp-python==0.3.35"
SRC=""
for d in "$T"/venv/lib/python3*/site-packages/llama_cpp/lib; do SRC="$d"; done
want=$(ls "$LIB" | grep "^libggml-cpu" | sort)
got=$(ls "$SRC" | grep "^libggml-cpu" | sort)
[ "$want" = "$got" ] || { echo "FAIL: soname set mismatch wheel[$want] vs sdist[$got]"; exit 1; }
# rm first: site-packages files are hardlinked into uv's cache — cp onto the
# existing inode would corrupt the shared cache entry.
rm -f "$LIB"/libggml-cpu.so*
cp "$SRC"/libggml-cpu.so* "$LIB"/
sha256sum "$LIB"/libggml-cpu.so* | tee "$MARKER"
echo "repair done: wheel GPU libs + host CPU kernels"
