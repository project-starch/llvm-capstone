#!/usr/bin/env bash
# C-21: a ternary yielding one of two __int128 CONSTANTS cannot be selected.
# No board, no QEMU, one file.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C21-i128-select-of-constants
CLANG=llvm/cmake-build-debug/bin/clang
O=/tmp/capstone/c21.o
rc=0
for opt in -O0 -O1 -Os; do
  rm -f "$O"                        # a stale object from another arm reads as a pass
  printf "  %-4s " "$opt"
  if out=$("$CLANG" -target capstone64-unknown-elf -ffreestanding -nostdlibinc \
             -std=c99 "$opt" -w -c "$D/src/sel.c" -o "$O" 2>&1); then
    echo "compiles, $(stat -c%s "$O") bytes"
  else
    echo "CANNOT SELECT: $(echo "$out" | grep -oE 'i128 = CapstoneISD::[A-Z_]+' | head -1)"
    rc=1
  fi
done
[[ $rc == 0 ]] && echo "  VERDICT: C-21 ABSENT" || echo "  VERDICT: C-21 PRESENT"
exit $rc
