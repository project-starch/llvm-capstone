#!/usr/bin/env bash
# C-23: an i128 whose high half carries information is computed on the low 64
# bits only; the register holding the high half is never read.
# No board, no QEMU. Exit 1 if the miscompile is present.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
D=capstone/tests/compiler-repros/C23-i128-high-half-silently-dropped
CLANG=llvm/cmake-build-debug/bin/clang

ASM=$("$CLANG" -target capstone64-unknown-elf -Xclang -target-feature -Xclang +m \
        -ffreestanding -nostdlibinc -fno-builtin -O1 -S "$D/src/halves.c" -o - 2>&1) || {
  echo "  ERROR: compile failed"; echo "$ASM" | tail -5; exit 2; }

body() { echo "$ASM" | sed -n "/^$1:/,/^\.Lfunc_end/p"; }

# Argument registers, confirmed from control_returns_b: out=a0, a=a1, b=a2.
# a2 is the ONLY place the high half can come from.
reads_a2() { body "$1" | grep -qE '\ba2\b'; }

echo "  --- positive control ---"
if reads_a2 control_returns_b; then
  echo "  control_returns_b reads a2: yes (detector works)"
else
  echo "  ERROR: the control does not read a2 -- the detector is broken or the"
  echo "         calling convention changed. No conclusion can be drawn."
  body control_returns_b | sed 's/^/    /'
  exit 2
fi

fail=0
for fn in store_assembled eq_full; do
  echo "  --- $fn ---"
  body "$fn" | grep -vE '^\s*\.' | sed 's/^/    /'
  if reads_a2 "$fn"; then
    echo "    reads a2 (high half): yes"
  else
    echo "    reads a2 (high half): NO -- b is silently discarded"
    fail=1
  fi
done

if [ "$fail" -eq 0 ]; then
  echo "  VERDICT: C-23 ABSENT -- both functions read the high half"
  exit 0
fi
echo "  VERDICT: C-23 PRESENT -- i128 is computed on its low 64 bits only."
echo "           This is SILENT: no diagnostic, no crash, wrong value."
exit 1
