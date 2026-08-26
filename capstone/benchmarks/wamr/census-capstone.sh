#!/usr/bin/env bash
# Compile every WAMR core source for capstone64 and report what fails, by CAUSE.
#
# Mirrors micropython/census-capstone.sh, and for the same reason: the useful
# question at this stage is not "does it link" but "is anything here hostile to
# capabilities", and a per-file census answers that before a line of porting glue
# is written. A count alone would not -- the causes are what separate "needs a
# freestanding libc" from "needs the object model changed", and those differ by
# months.
#
# BASELINE is asserted, not printed and forgotten: if the number of files that
# compile goes DOWN this exits non-zero, the way the musl survey does.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../../tests/capstone-test-env.sh"

SRC=${WAMR_SRC_DIR:-$CAPSTONE_TMP_ROOT/wamr-src/wasm-micro-runtime}
[[ -d "$SRC/core" ]] || { echo "no WAMR source at $SRC -- run fetch-wamr.sh" >&2; exit 2; }

# 27 of 27 at f73410e2: the whole interpreter core, with the freestanding shim in
# adapted/include and one upstream patch. Never lower it to make a run pass.
BASELINE_OK=${WAMR_BASELINE_OK:-27}

INC=(-I"$SCRIPT_DIR/port" -I"$SCRIPT_DIR/adapted/include"
     -I"$SRC/core/shared/utils" -I"$SRC/core/shared/platform/include"
     -I"$SRC/core/shared/mem-alloc" -I"$SRC/core/iwasm/include"
     -I"$SRC/core/iwasm/common" -I"$SRC/core/iwasm/interpreter" -I"$SRC/core")

# BUILD_TARGET_RISCV64_LP64 explicitly: WAMR's config.h detects the target from
# __riscv, and this toolchain defines __capstone instead, so autodetection fails
# with "Build target isn't set" rather than picking something wrong.
# -nostdinc is LOAD-BEARING. Without it the driver still searches /usr/include,
# the host stdio.h wins over adapted/include/, and the census reports libc failures
# that are really include-order failures -- which is what it did on the first run
# with the shim in place.
RESOURCE_DIR=$("$CAPSTONE_LLVM_BIN/clang" -print-resource-dir)
FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin -nostdinc -isystem "$RESOURCE_DIR/include"
       -O1 -w -DBUILD_TARGET_RISCV64_LP64
       # ONE coherent configuration, not the union of all of them. Compiling every
       # file meant compiling the classic AND the fast interpreter, which exclude
       # each other -- the "no member named 'operand'" and "block_addr_cache"
       # failures were that, not a port problem. Classic interpreter only, no AOT,
       # no WASI, no threads: the smallest thing that can run a module.
       -DWASM_ENABLE_INTERP=1 -DWASM_ENABLE_FAST_INTERP=0
       -DWASM_ENABLE_AOT=0 -DWASM_ENABLE_JIT=0
        # Defaults to 1 in core/config.h even with AOT off, and its init builds an
        # entry table for a path this configuration does not have.
        -DWASM_ENABLE_QUICK_AOT_ENTRY=0
       -DWASM_ENABLE_LIBC_WASI=0 -DWASM_ENABLE_LIBC_BUILTIN=1
       -DWASM_ENABLE_MULTI_MODULE=0 -DWASM_ENABLE_SHARED_MEMORY=0
       -DWASM_ENABLE_THREAD_MGR=0 -DWASM_ENABLE_MEMORY_PROFILING=0
       -DWASM_ENABLE_GLOBAL_HEAP_POOL=1
       # The runtime build asserts these two are exactly these names
       # (wasm_runtime_common.c:45-56), so the CMake normally sets them.
       -DBH_MALLOC=wasm_runtime_malloc -DBH_FREE=wasm_runtime_free)

declare -A CAUSE
ok=0; bad=0
for f in "$SRC"/core/shared/mem-alloc/*.c "$SRC"/core/shared/mem-alloc/ems/*.c \
         "$SRC"/core/shared/utils/*.c "$SRC"/core/iwasm/common/*.c \
         "$SRC"/core/iwasm/interpreter/*.c; do
  case "$(basename "$f")" in
    wasm_interp_fast.c|wasm_mini_loader.c) continue ;;   # nicht in dieser Konfiguration
  esac
  [[ -e "$f" ]] || continue
  if "$CAPSTONE_LLVM_BIN/clang" "${FLAGS[@]}" "${INC[@]}" -c "$f" -o /dev/null >/tmp/wamr-census.err 2>&1; then
    ok=$((ok+1)); continue
  fi
  bad=$((bad+1))
  msg=$(grep -m1 'error:' /tmp/wamr-census.err | sed 's|.*error: ||')
  case "$msg" in
    *"undeclared function"*) c="freestanding libc: $(sed "s/.*undeclared function '\([^']*\)'.*/\1/" <<<"$msg")" ;;
    *"Cannot select"*|*"i128"*|*capability*) c="CAPABILITY: $msg" ;;
    *) c="other: ${msg:0:56}" ;;
  esac
  CAUSE["$c"]=$(( ${CAUSE["$c"]:-0} + 1 ))
done

echo "WAMR census at $(git -C "$SRC" rev-parse --short HEAD)"
echo "  compiled $ok, failed $bad"
echo "failures by cause:"
for k in "${!CAUSE[@]}"; do printf '  %3d  %s\n' "${CAUSE[$k]}" "$k"; done | sort -rn
if (( ok < BASELINE_OK )); then
  echo "REGRESSION: $ok compiled, baseline is $BASELINE_OK" >&2; exit 1
fi
