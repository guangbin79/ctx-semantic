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
# works, so the fetch is 16 parallel ranged chunks + sha256 verification.
#
# Bump contract: WHEEL_URL + WHEEL_SHA256 + SDIST_SHA256 here, the
# [tool.uv.sources] path pin in pyproject.toml, and uv.lock move together.
# WHEEL/VER/SDIST below are DERIVED from the pyproject pin (single source of
# truth — an env-var override here could fetch to a place uv never reads).
set -eu
cd "$(dirname "$0")/.."

WHEEL_URL=https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu124/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl
WHEEL_SHA256=62fc788a4ecee5a579f40708ccfe0c4d566dd5c9959497bdc0544dbba94d82aa
SDIST_SHA256=1139dbb54509074b70893fab8554e3b079aa9f4d312058ce4018ef0019e3de12

WHEEL=$(grep -E '^llama-cpp-python[[:space:]]*=' pyproject.toml | grep -o '"[^"]*\.whl"' | tr -d '"')
[ -n "$WHEEL" ] || { echo "FAIL: cannot read llama-cpp-python pin from pyproject.toml"; exit 1; }
WHEEL_DIR=$(dirname "$WHEEL")
VER=$(basename "$WHEEL" | cut -d- -f2)
[ -n "$VER" ] || { echo "FAIL: cannot parse version from $(basename "$WHEEL")"; exit 1; }
[ "$(basename "$WHEEL")" = "$(basename "$WHEEL_URL")" ] || {
    echo "FAIL: pyproject pin ($(basename "$WHEEL")) and WHEEL_URL ($(basename "$WHEEL_URL")) disagree — bump contract broken"
    exit 1
}
SDIST="$WHEEL_DIR/llama_cpp_python-$VER.tar.gz"

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
    # per-part size check before concatenation: catches a failed/short chunk
    # AND a Range-ignoring server (every part would be `size` bytes) with a
    # precise message, instead of an opaque final hash mismatch.
    i=0
    while [ "$i" -lt "$parts" ]; do
        start=$(( i * chunk )); end=$(( start + chunk - 1 ))
        [ "$end" -ge "$size" ] && end=$(( size - 1 ))
        asz=$(wc -c < "$tmp/$i" 2>/dev/null || echo 0)
        [ "$asz" -eq $(( end - start + 1 )) ] || {
            echo "FAIL: part $i is ${asz}B, want $(( end - start + 1 ))B (bad chunk or Range-ignoring server)"
            exit 1
        }
        i=$(( i + 1 ))
    done
    i=0
    while [ "$i" -lt "$parts" ]; do cat "$tmp/$i"; i=$(( i + 1 )); done > "$WHEEL"
    rm -rf "$tmp"
    echo "$WHEEL_SHA256  $WHEEL" | sha256sum -c || exit 1
fi

uv sync "$@"

# --- host repair of libggml-cpu ---
LIB=""
for d in .venv/lib/python3*/site-packages/llama_cpp/lib; do [ -d "$d" ] && LIB="$d"; done
[ -n "$LIB" ] || { echo "FAIL: llama_cpp not installed in .venv"; exit 1; }
[ -f "$LIB/libggml-cpu.so" ] || { echo "FAIL: $LIB/libggml-cpu.so missing"; exit 1; }
MARKER="$LIB/.host-cpu-lib.sha256"
cur=$(sha256sum "$LIB/libggml-cpu.so" | cut -d" " -f1)
if [ -f "$MARKER" ] && [ "$cur" = "$(head -1 "$MARKER" | cut -d' ' -f1)" ]; then
    echo "libggml-cpu already host-built (sha256 $cur)"
    exit 0
fi

echo "rebuilding libggml-cpu from the $VER sdist on this host (GGML_CUDA=OFF)..."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# sdist: fetched once and sha256-verified like the wheel — `uv pip` does not
# apply the lock's hashes, so without this the lib swapped into the venv
# would come from whatever PyPI serves that day.
if ! echo "$SDIST_SHA256  $SDIST" | sha256sum -c --status 2>/dev/null; then
    sdist_url=$(curl -s "https://pypi.org/pypi/llama-cpp-python/$VER/json" \
        | grep -o '"url": *"[^"]*\.tar\.gz"' | grep -o 'https://[^"]*' | head -1)
    [ -n "$sdist_url" ] || { echo "FAIL: no sdist URL from PyPI JSON for $VER"; exit 1; }
    curl -sfL -o "$SDIST" "$sdist_url"
fi
echo "$SDIST_SHA256  $SDIST" | sha256sum -c || exit 1

uv venv -q "$T/venv"
CMAKE_ARGS="-DGGML_CUDA=OFF" uv pip install -q --python "$T/venv/bin/python" "$SDIST"
SRC=""
for d in "$T"/venv/lib/python3*/site-packages/llama_cpp/lib; do [ -d "$d" ] && SRC="$d"; done
[ -n "$SRC" ] || { echo "FAIL: sdist build produced no llama_cpp lib dir"; exit 1; }
want=$(ls "$LIB" | grep "^libggml-cpu" | sort)
got=$(ls "$SRC" | grep "^libggml-cpu" | sort)
[ -n "$want" ] && [ "$want" = "$got" ] || { echo "FAIL: soname set mismatch wheel[$want] vs sdist[$got]"; exit 1; }
# cp to fresh inodes, then mv into place: rename(2) swaps each directory
# entry atomically (no window with libggml-cpu absent) and never writes
# through uv's cache hardlinks (writing through the old entry would
# corrupt the shared cache).
for so in "$SRC"/libggml-cpu.so*; do
    n=$(basename "$so")
    cp "$so" "$LIB/.host-new.$n"
    mv -f "$LIB/.host-new.$n" "$LIB/$n"
done
sha256sum "$LIB"/libggml-cpu.so* | tee "$MARKER"
echo "repair done: wheel GPU libs + host CPU kernels"
