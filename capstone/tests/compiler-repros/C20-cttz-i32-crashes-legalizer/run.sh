#!/usr/bin/env bash
# C-20: __builtin_ctz on a 32-bit value crashes the Capstone backend.
#
#   ./run.sh          reproduce; prints VERDICT
#   ./run.sh workaround   show that -mattr=+zbb avoids it
#
# No board, no QEMU, no build of anything but this one file.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C20-cttz-i32-crashes-legalizer
CLANG=llvm/cmake-build-debug/bin/clang
[[ -x $CLANG ]] || { echo "no clang at $CLANG"; exit 1; }
F=(-target capstone64-unknown-elf -ffreestanding -nostdlibinc -std=c99 -O0 -w)
O=/tmp/capstone/c20.o

run() {  # $1 label, rest = extra flags
  local label=$1; shift
  rm -f "$O"                      # a stale object from an earlier arm reads as a pass
  printf "  %-22s " "$label"
  if out=$("$CLANG" "${F[@]}" "$@" -c "$D/src/cttz.c" -o "$O" 2>&1); then
    echo "compiles, $(stat -c%s "$O") bytes"
  else
    echo "CRASH: $(echo "$out" | grep -oE 'Assertion .*failed' | head -1 | cut -c1-64)"
  fi
}

if [[ ${1:-} == workaround ]]; then
  run "no attributes"
  run "+zbb" -Xclang -target-feature -Xclang +zbb
  echo
  echo "  +zbb avoids it, but NO build in this tree passes +zbb, and whether the"
  echo "  silicon implements Zbb is not established here. Do not adopt it as a fix"
  echo "  for a board build on the strength of this script."
  exit 0
fi

rm -f "$O"
if "$CLANG" "${F[@]}" -c "$D/src/cttz.c" -o "$O" 2>/dev/null; then
  echo "  VERDICT: C-20 ABSENT -- the file compiles"
  exit 0
else
  echo "  VERDICT: C-20 PRESENT -- __builtin_ctz crashes the backend"
  exit 1
fi
