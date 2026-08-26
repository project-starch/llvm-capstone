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

# 15 of 29 at f73410e2, with every failure a missing libc declaration and none of
# them capability-related. Raise this when the libc shim lands; never lower it to
# make a run pass.
BASELINE_OK=${WAMR_BASELINE_OK:-15}

INC=(-I"$SCRIPT_DIR/port"
     -I"$SRC/core/shared/utils" -I"$SRC/core/shared/platform/include"
     -I"$SRC/core/shared/mem-alloc" -I"$SRC/core/iwasm/include"
     -I"$SRC/core/iwasm/common" -I"$SRC/core/iwasm/interpreter" -I"$SRC/core")

# BUILD_TARGET_RISCV64_LP64 explicitly: WAMR's config.h detects the target from
# __riscv, and this toolchain defines __capstone instead, so autodetection fails
# with "Build target isn't set" rather than picking something wrong.
FLAGS=(-target capstone64-unknown-elf -Xclang -target-feature -Xclang +m
       -ffreestanding -fno-builtin -O1 -w -DBUILD_TARGET_RISCV64_LP64)

declare -A CAUSE
ok=0; bad=0
for f in "$SRC"/core/shared/mem-alloc/*.c "$SRC"/core/shared/mem-alloc/ems/*.c \
         "$SRC"/core/shared/utils/*.c "$SRC"/core/iwasm/common/*.c \
         "$SRC"/core/iwasm/interpreter/*.c; do
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
