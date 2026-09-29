#!/bin/sh
# Dual-profile uv sync for llama-cpp-python (SHAPE=1: pyproject extras
# cpu/cuda). No argument = CPU default profile; --cuda = opt-in CUDA profile.
#
# CPU default: `uv sync --extra cpu` pulls the ~24MB manylinux cpu wheel from
# abetlen's cpu index — no CUDA libs, no cmake, imports with no
# LD_LIBRARY_PATH (spike-cpu-wheel.out, cpu-wheel-verdict=OK). The sdist
# CMAKE fallback branch is consciously omitted: the SIGILL trigger below is
# false for the cpu wheel. Reintroduce it only if a future cpu wheel SIGILLs
# (CMAKE_ARGS="-DGGML_CUDA=off" uv pip install --no-binary llama-cpp-python).
#
# CUDA profile: the pinned cu124 prebuilt wheel's libggml-cpu.so was compiled
# with -march native on Zen4 CI runners and executes AVX-VNNI (VEX vpdpbusd) —
# SIGILL on this host's i7-11800H (Tiger Lake). The GPU kernels
# (libggml-cuda.so) are unaffected. Fix: after installing the wheel, rebuild
# libggml-cpu from the SAME 0.3.35 sdist on THIS host with GGML_CUDA=OFF and
# swap it in. A full CUDA source build is not possible here: pip's nvidia
# nvcc wheels ship only ptxas + headers, no nvcc.
#
# The script also materializes the cuda wheel: uv cannot range-read GitHub
# release assets (falls back to streaming the whole 1.7GB single-stream,
# ~0.4MB/s on this host), but curl -r against the same signed-redirect URL
# works, so the fetch is 16 parallel ranged chunks + sha256 verification.
# The wheel is then installed imperatively (uv pip install) after
# `uv sync --extra cuda`: the lock records the cuda path arm as a
# requires-dist mapping only, so the sync itself never installs it (see the
# inline comment at the install step).
#
# Bump contract: when bumping llama-cpp-python, bump BOTH constant blocks
# below together — CPU_WHEEL_URL/_SHA256/_BYTES and WHEEL_URL/_SHA256 +
# SDIST_SHA256 — plus the ==0.3.35 version pins in pyproject.toml
# [project.optional-dependencies] and uv.lock. The script constants are the
# pin source of truth: the cpu constants document what the abetlen index must
# serve (uv resolves the cpu profile via the index), the cuda constants drive
# the ranged fetch + sha256 verify exactly as before.
set -eu
cd "$(dirname "$0")/.."

# --- cpu profile constants (documented pin; the abetlen cpu index serves
# this exact wheel — spike-cpu-wheel.out) ---
CPU_WHEEL_URL=https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35/llama_cpp_python-0.3.35-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl
CPU_WHEEL_SHA256=d172f3d3c8cdd194c3c47c71cb077ed6e61354a2d0f939ceeac0c8fd29999596
CPU_WHEEL_BYTES=23912624

# --- cuda profile constants (drive the materialize + sha256 gate below) ---
WHEEL_URL=https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-cu124/llama_cpp_python-0.3.35-py3-none-manylinux_2_35_x86_64.whl
WHEEL_SHA256=62fc788a4ecee5a579f40708ccfe0c4d566dd5c9959497bdc0544dbba94d82aa
SDIST_SHA256=1139dbb54509074b70893fab8554e3b079aa9f4d312058ce4018ef0019e3de12

case "${1:-}" in
    "")     MODE=cpu ;;
    --cuda) MODE=cuda ;;
    *)      echo "usage: $0 [--cuda]  (no argument = cpu default profile)"; exit 2 ;;
esac

WHEEL="models/wheels/$(basename "$WHEEL_URL")"
VER=$(basename "$WHEEL_URL" | cut -d- -f2)
SDIST="models/wheels/llama_cpp_python-$VER.tar.gz"

# Embedded import self-check. Runs after sync in both profiles. The cuda arm
# preloads the nvidia-wheel libs via the embedder's own discovery/CDLL
# machinery: the cu124 wheel dlopens libcudart at import, so a bare import
# cannot succeed in-process without the preload (embedder.py; LD_LIBRARY_PATH
# is deliberately not used anywhere in this script).
probe() {
    if [ "$MODE" = cpu ]; then
        v=$(uv run --no-sync python -c 'import llama_cpp; print(llama_cpp.__version__)') || {
            echo "FAIL: cpu import probe could not import llama_cpp"; exit 1; }
    else
        v=$(uv run --no-sync python -c '
import ctypes
from ctx_semantic.embedder import _find_nvidia_libs
libs = _find_nvidia_libs()
missing = [k for k, p in libs.items() if p is None]
assert not missing, "nvidia wheel libs missing after sync: %s" % missing
for p in libs.values():
    ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
import llama_cpp
print(llama_cpp.__version__)
') || { echo "FAIL: cuda import probe failed (traceback above)"; exit 1; }
    fi
    [ "$v" = "0.3.35" ] || { echo "FAIL: import probe got '$v', want 0.3.35"; exit 1; }
    echo "self-check OK: llama_cpp 0.3.35 imports with no LD_LIBRARY_PATH ($MODE profile)"
}

if [ "$MODE" = cpu ]; then
    uv sync --extra cpu
    probe
    exit 0
fi

# --- cuda profile: materialize the wheel (ranged fetch + verify) ---
mkdir -p models/wheels
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

uv sync --extra cuda

# The lock records the cuda path arm as a requires-dist mapping only — uv
# never reads path-wheel METADATA at lock time, so there is no [[package]]
# entry and `uv sync --extra cuda` silently skips the wheel itself (it only
# installs the nvidia wheels + base/dev). Install the verified wheel
# imperatively, after sync (a later sync would strip it — same order rule
# as the repair step).
echo "installing materialized wheel: $WHEEL"
uv pip install -q --python .venv/bin/python "$WHEEL"

# --- host repair of libggml-cpu ---
LIB=""
for d in .venv/lib/python3*/site-packages/llama_cpp/lib; do [ -d "$d" ] && LIB="$d"; done
[ -n "$LIB" ] || { echo "FAIL: llama_cpp not installed in .venv"; exit 1; }
[ -f "$LIB/libggml-cpu.so" ] || { echo "FAIL: $LIB/libggml-cpu.so missing"; exit 1; }
MARKER="$LIB/.host-cpu-lib.sha256"
cur=$(sha256sum "$LIB/libggml-cpu.so" | cut -d" " -f1)
if [ -f "$MARKER" ] && [ "$cur" = "$(head -1 "$MARKER" | cut -d' ' -f1)" ]; then
    echo "libggml-cpu already host-built (sha256 $cur)"
    probe
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

probe
